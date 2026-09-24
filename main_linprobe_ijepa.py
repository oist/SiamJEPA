# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright (c) 2026 Makoto Yamada and contributors.
# All rights reserved.
#
# Linear probing following the I-JEPA evaluation protocol
# (Assran et al., 2023, Appendix "Linear evaluation"):
#   * frozen target (EMA) encoder, final LayerNorm applied, patch tokens average-pooled
#   * two feature variants, best one reported:
#       - "last":  average-pooled patch tokens of the last block
#       - "cat4":  concatenation of the average-pooled patch tokens of the last 4 blocks
#   * LARS, batch size 16384, 50 epochs, lr divided by 10 every 15 epochs
#   * sweep of reference lr {0.01, 0.05, 0.001} x weight decay {0.0005, 0.0}
#   * RandomResizedCrop + horizontal flip
#
# All 2 x 3 x 2 = 12 linear heads are trained simultaneously on top of a single
# forward pass of the frozen encoder, so the full sweep costs one probe job.
# Each head is BatchNorm1d(affine=False) + Linear, as in the MAE linear probe.
# Reference lr is scaled as lr = ref_lr * batch_size / 256 (MAE/MSN convention).

import argparse
import datetime
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.distributed as dist
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from timm.models.layers import trunc_normal_

import util.misc as misc
from util.pos_embed import interpolate_pos_embed
from util.lars import LARS
from util.crop import RandomResizedCrop
import util.experiment_tracking as exptrack

import models_vit


def get_args_parser():
    parser = argparse.ArgumentParser('I-JEPA-protocol linear probing', add_help=False)
    parser.add_argument('--batch_size', default=512, type=int, help='per-GPU batch size')
    parser.add_argument('--accum_iter', default=8, type=int,
                        help='effective batch = batch_size * accum_iter * #gpus (16384 = 512*8*4)')
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--lr_step', default=15, type=int, help='divide lr by 10 every N epochs')
    parser.add_argument('--ref_lrs', default='0.01,0.05,0.001', type=str)
    parser.add_argument('--weight_decays', default='0.0005,0.0', type=str)
    parser.add_argument('--n_last_blocks', default=4, type=int)
    parser.add_argument('--model', default='vit_base_patch16', type=str)
    parser.add_argument('--finetune', required=True, help='pretrain checkpoint')
    parser.add_argument('--use_ema', action='store_true', help='probe the EMA/teacher encoder')
    parser.add_argument('--data_path', default='/home/pj26000049/ku60000347/Python/Dataset/ImageNet/', type=str)
    parser.add_argument('--nb_classes', default=1000, type=int)
    parser.add_argument('--output_dir', default='./output_dir_linprobe_ijepa')
    parser.add_argument('--resume_dir', default='', help='existing run dir to resume from')
    parser.add_argument('--eval_freq', default=5, type=int)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://')
    parser.add_argument('--local_rank', '--local-rank', type=int, default=-1)
    return parser


class FrozenEncoder(nn.Module):
    """Returns {'last': [B, D], 'cat4': [B, n*D]} of LN'd, average-pooled patch tokens."""

    def __init__(self, vit, n_last_blocks):
        super().__init__()
        self.vit = vit
        self.n = n_last_blocks

    @torch.no_grad()
    def forward(self, x):
        v = self.vit
        B = x.shape[0]
        x = v.patch_embed(x)
        x = torch.cat((v.cls_token.expand(B, -1, -1), x), dim=1)
        x = v.pos_drop(x + v.pos_embed)
        outs = []
        for i, blk in enumerate(v.blocks):
            x = blk(x)
            if i >= len(v.blocks) - self.n:
                outs.append(v.norm(x)[:, 1:, :].mean(dim=1))
        return {'last': outs[-1].float(), 'cat4': torch.cat(outs, dim=-1).float()}


class Heads(nn.Module):
    def __init__(self, configs, dims, nb_classes):
        super().__init__()
        self.configs = configs
        self.heads = nn.ModuleList()
        for c in configs:
            lin = nn.Linear(dims[c['feat']], nb_classes)
            trunc_normal_(lin.weight, std=0.01)
            nn.init.zeros_(lin.bias)
            self.heads.append(nn.Sequential(nn.BatchNorm1d(dims[c['feat']], affine=False, eps=1e-6), lin))

    def forward(self, feats):
        return [h(feats[c['feat']]) for c, h in zip(self.configs, self.heads)]


def head_name(c):
    return f"{c['feat']}_lr{c['ref_lr']:g}_wd{c['wd']:g}"


def set_lr(optimizer, epoch, args):
    factor = 0.1 ** (epoch // args.lr_step)
    for g in optimizer.param_groups:
        g['lr'] = g['base_lr'] * factor


@torch.no_grad()
def evaluate(encoder, heads, loader, device, n_heads):
    heads.eval()
    correct1 = torch.zeros(n_heads, device=device, dtype=torch.float64)
    correct5 = torch.zeros(n_heads, device=device, dtype=torch.float64)
    total = torch.zeros(1, device=device, dtype=torch.float64)
    for images, target in loader:
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        with torch.cuda.amp.autocast():
            feats = encoder(images)
        for i, out in enumerate(heads(feats)):
            top5 = out.topk(5, dim=1).indices
            hit = top5.eq(target[:, None])
            correct1[i] += hit[:, 0].sum()
            correct5[i] += hit.any(dim=1).sum()
        total += target.shape[0]
    if dist.is_initialized():
        for t in (correct1, correct5, total):
            dist.all_reduce(t)
    heads.train()
    return (100 * correct1 / total).tolist(), (100 * correct5 / total).tolist(), int(total.item())


def main(args):
    misc.init_distributed_mode(args)
    device = torch.device(args.device)
    world = misc.get_world_size()
    rank = misc.get_rank()

    repo_dir = os.path.dirname(os.path.realpath(__file__))
    git_info = exptrack.get_git_info(repo_dir)
    if args.resume_dir:
        args.output_dir = args.resume_dir
    else:
        ckpt_path = Path(args.finetune)
        encoder_tag = "ema" if args.use_ema else "student"
        tag = f"ijepaprobe_{ckpt_path.parent.name}_{ckpt_path.stem}_{encoder_tag}"
        args.output_dir = exptrack.make_run_dir(
            args.output_dir, tag, git_info, misc.is_main_process(), extra={"args": vars(args)})
    print('resolved output_dir: {}'.format(args.output_dir))
    print("{}".format(args).replace(', ', ',\n'))

    torch.manual_seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    cudnn.benchmark = True

    transform_train = transforms.Compose([
        RandomResizedCrop(224, interpolation=3),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    transform_val = transforms.Compose([
        transforms.Resize(256, interpolation=3),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    dataset_train = datasets.ImageFolder(os.path.join(args.data_path, 'train'), transform=transform_train)
    dataset_val = datasets.ImageFolder(os.path.join(args.data_path, 'val'), transform=transform_val)
    sampler_train = torch.utils.data.DistributedSampler(dataset_train, num_replicas=world, rank=rank, shuffle=True)
    # exact sharded eval: disjoint strided subsets, no padding duplicates
    val_subset = torch.utils.data.Subset(dataset_val, list(range(rank, len(dataset_val), world)))
    loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train, batch_size=args.batch_size,
        num_workers=args.num_workers, pin_memory=True, drop_last=True)
    loader_val = torch.utils.data.DataLoader(
        val_subset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True, drop_last=False)

    # ---- frozen encoder (keeps the pretrained final LayerNorm, no fc_norm) ----
    vit = models_vit.__dict__[args.model](num_classes=args.nb_classes, global_pool=False)
    checkpoint_model = torch.load(args.finetune, map_location='cpu')['model']
    if args.use_ema:
        prefix = 'ema_model.'
        checkpoint_model = {k[len(prefix):]: v for k, v in checkpoint_model.items() if k.startswith(prefix)}
        assert checkpoint_model, "--use_ema set but no 'ema_model.*' keys found"
    for k in ['head.weight', 'head.bias']:
        checkpoint_model.pop(k, None)
    interpolate_pos_embed(vit, checkpoint_model)
    msg = vit.load_state_dict(checkpoint_model, strict=False)
    print(msg)
    assert set(msg.missing_keys) == {'head.weight', 'head.bias'}, msg.missing_keys
    encoder = FrozenEncoder(vit, args.n_last_blocks).to(device).eval()
    for p in encoder.parameters():
        p.requires_grad = False

    # ---- 12 heads, one LARS param group per head ----
    D = vit.embed_dim
    dims = {'last': D, 'cat4': D * args.n_last_blocks}
    ref_lrs = [float(x) for x in args.ref_lrs.split(',')]
    wds = [float(x) for x in args.weight_decays.split(',')]
    configs = [{'feat': f, 'ref_lr': lr, 'wd': wd} for f in ('last', 'cat4') for lr in ref_lrs for wd in wds]
    names = [head_name(c) for c in configs]
    heads = Heads(configs, dims, args.nb_classes).to(device)

    eff_batch = args.batch_size * args.accum_iter * world
    param_groups = []
    for c, h in zip(configs, heads.heads):
        base_lr = c['ref_lr'] * eff_batch / 256
        param_groups.append({'params': list(h.parameters()), 'lr': base_lr, 'base_lr': base_lr,
                             'weight_decay': c['wd']})
    optimizer = LARS(param_groups)
    print(f"effective batch size: {eff_batch}")
    for n, g in zip(names, optimizer.param_groups):
        print(f"  head {n}: base lr {g['base_lr']:.4g}, wd {g['weight_decay']:g}")

    heads_ddp = torch.nn.parallel.DistributedDataParallel(heads, device_ids=[args.gpu]) if args.distributed else heads
    criterion = nn.CrossEntropyLoss()

    start_epoch = 0
    ckpt_file = os.path.join(args.output_dir, 'heads_checkpoint.pth')
    if os.path.exists(ckpt_file):
        ck = torch.load(ckpt_file, map_location='cpu')
        heads.load_state_dict(ck['heads'])
        optimizer.load_state_dict(ck['optimizer'])
        start_epoch = ck['epoch'] + 1
        print(f"resumed from {ckpt_file} at epoch {start_epoch}")

    print(f"Start training for {args.epochs} epochs")
    t0 = time.time()
    for epoch in range(start_epoch, args.epochs):
        sampler_train.set_epoch(epoch)
        set_lr(optimizer, epoch, args)
        heads.train()
        optimizer.zero_grad()
        loss_sum, n_steps = 0.0, 0
        for it, (images, target) in enumerate(loader_train):
            images = images.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            with torch.cuda.amp.autocast():
                feats = encoder(images)
            update = (it + 1) % args.accum_iter == 0
            ctx = heads_ddp.no_sync() if (args.distributed and not update) else _null()
            with ctx:
                outs = heads_ddp(feats)
                loss = sum(criterion(o, target) for o in outs)
                if not math.isfinite(loss.item()):
                    raise RuntimeError(f"non-finite loss at epoch {epoch} iter {it}")
                (loss / args.accum_iter).backward()
            if update:
                optimizer.step()
                optimizer.zero_grad()
            loss_sum += loss.item() / len(outs)
            n_steps += 1
            if it % 100 == 0:
                print(f"Epoch [{epoch}] it {it}/{len(loader_train)} mean-head loss {loss.item() / len(outs):.4f} "
                      f"lr(max) {max(g['lr'] for g in optimizer.param_groups):.4g}")

        log = {'epoch': epoch, 'train_loss_mean_head': loss_sum / max(n_steps, 1),
               'elapsed': str(datetime.timedelta(seconds=int(time.time() - t0)))}
        if (epoch + 1) % args.eval_freq == 0 or epoch + 1 == args.epochs:
            acc1, acc5, n_val = evaluate(encoder, heads, loader_val, device, len(configs))
            log['n_val'] = n_val
            log['test_acc1'] = dict(zip(names, acc1))
            log['test_acc5'] = dict(zip(names, acc5))
            best = int(np.argmax(acc1))
            log['best_head'] = names[best]
            log['best_acc1'] = acc1[best]
            for feat in ('last', 'cat4'):
                idx = [i for i, c in enumerate(configs) if c['feat'] == feat]
                j = max(idx, key=lambda i: acc1[i])
                log[f'best_{feat}_head'] = names[j]
                log[f'best_{feat}_acc1'] = acc1[j]
            print(f"* epoch {epoch}: best {names[best]} {acc1[best]:.3f} | "
                  f"last {log['best_last_acc1']:.3f} | cat4 {log['best_cat4_acc1']:.3f}")

        if misc.is_main_process():
            with open(os.path.join(args.output_dir, 'log.txt'), 'a', encoding='utf-8') as f:
                f.write(json.dumps(log) + '\n')
            torch.save({'heads': heads.state_dict(), 'optimizer': optimizer.state_dict(),
                        'epoch': epoch, 'names': names}, ckpt_file + '.tmp')
            os.replace(ckpt_file + '.tmp', ckpt_file)
        if dist.is_initialized():
            dist.barrier()

    print('Training time {}'.format(str(datetime.timedelta(seconds=int(time.time() - t0)))))


class _null:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


if __name__ == '__main__':
    args = get_args_parser().parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
