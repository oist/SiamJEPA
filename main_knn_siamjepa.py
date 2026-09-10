# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright (c) 2026 Makoto Yamada and contributors.
# All rights reserved.

# This file is based on the Meta MAE / Facebook DINO implementations and has
# been substantially modified for the SiamJEPA project.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# --------------------------------------------------------
# References:
# MAE:  https://github.com/facebookresearch/mae
# DINO: https://github.com/facebookresearch/dino (eval_knn.py weighted k-NN classifier)
# --------------------------------------------------------

import argparse
import json
import os
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import util.misc as misc
from util.pos_embed import interpolate_pos_embed
import util.experiment_tracking as exptrack

import models_vit


def get_args_parser():
    parser = argparse.ArgumentParser('SiamJEPA weighted k-NN evaluation', add_help=False)
    parser.add_argument('--batch_size', default=512, type=int,
                        help='Batch size per GPU for feature extraction')
    parser.add_argument('--model', default='vit_base_patch16', type=str, metavar='MODEL')

    parser.add_argument('--finetune', required=True,
                        help='checkpoint to evaluate (a SiamJEPA pretrain checkpoint-N.pth)')
    parser.add_argument('--use_ema', action='store_true',
                        help='Evaluate the EMA/teacher encoder instead of the student encoder '
                             '(same semantics as the flag in main_linprobe_siamjepa.py).')
    parser.add_argument('--global_pool', action='store_true')
    parser.set_defaults(global_pool=True)
    parser.add_argument('--cls_token', action='store_false', dest='global_pool',
                        help='Use class token instead of global pool for the k-NN feature')

    parser.add_argument('--data_path', default='/home/pj26000049/ku60000347/Python/Dataset/ImageNet/', type=str)
    parser.add_argument('--nb_classes', default=1000, type=int)

    parser.add_argument('--nb_knn', default=[10, 20, 100, 200], type=int, nargs='+',
                        help='Numbers of neighbors to evaluate, DINO-style (reports each k).')
    parser.add_argument('--temperature', default=0.07, type=float,
                        help='Temperature for the weighted k-NN vote.')
    parser.add_argument('--knn_chunk_size', default=20, type=int,
                        help='Query chunk size for the k-NN similarity matmul (memory/speed tradeoff).')

    parser.add_argument('--output_dir', default='./output_dir_knn',
                        help='path where to save, empty for no saving. '
                             'A run-specific subdirectory (job id/timestamp + git commit + '
                             'evaluated checkpoint) is created under this path for each run.')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed
    parser.add_argument('--world_size', default=1, type=int)
    parser.add_argument('--local_rank', '--local-rank', type=int, default=-1)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://')
    return parser


@torch.no_grad()
def extract_features(model, data_loader, device):
    all_features, all_labels = [], []
    for imgs, targets in data_loader:
        imgs = imgs.to(device, non_blocking=True)
        with torch.cuda.amp.autocast():
            feats = model.forward_features(imgs)
        feats = F.normalize(feats.float(), dim=-1)
        all_features.append(feats.cpu())
        all_labels.append(targets.clone())
    return torch.cat(all_features), torch.cat(all_labels)


def all_gather_equal(tensor):
    """Gather equal-length tensors from every rank (safe here because both
    loaders use DistributedSampler(shuffle=False, drop_last=False), which
    pads every shard to the same length)."""
    world_size = misc.get_world_size()
    if world_size == 1:
        return tensor
    tensor = tensor.cuda()
    gathered = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(gathered, tensor)
    return torch.cat([g.cpu() for g in gathered], dim=0)


@torch.no_grad()
def knn_classifier(train_features, train_labels, test_features, test_labels, k, T,
                    num_classes=1000, chunk_size=20, device='cuda'):
    """DINO-style weighted k-NN vote (facebookresearch/dino, eval_knn.py)."""
    train_features = train_features.to(device)
    train_labels = train_labels.to(device)
    test_features = test_features.to(device)
    test_labels = test_labels.to(device)

    train_features_t = train_features.t()  # [dim, N_train]
    top1, top5, total = 0.0, 0.0, 0
    num_test = test_features.shape[0]
    retrieval_one_hot = torch.zeros(chunk_size * k, num_classes, device=device)

    for start in range(0, num_test, chunk_size):
        end = min(start + chunk_size, num_test)
        features = test_features[start:end]
        targets = test_labels[start:end]
        bs = targets.shape[0]

        similarity = torch.mm(features, train_features_t)
        distances, indices = similarity.topk(k, largest=True, sorted=True)
        candidates = train_labels.view(1, -1).expand(bs, -1)
        retrieved_neighbors = torch.gather(candidates, 1, indices)

        retrieval_one_hot.resize_(bs * k, num_classes).zero_()
        retrieval_one_hot.scatter_(1, retrieved_neighbors.reshape(-1, 1), 1)
        distances_transform = distances.clone().div_(T).exp_()
        probs = torch.sum(
            retrieval_one_hot.view(bs, -1, num_classes) * distances_transform.view(bs, -1, 1),
            dim=1,
        )
        _, predictions = probs.sort(1, True)

        correct = predictions.eq(targets.view(-1, 1))
        top1 += correct[:, :1].sum().item()
        top5 += correct[:, :min(5, k)].sum().item()
        total += bs

    return top1 * 100.0 / total, top5 * 100.0 / total


def main(args):
    misc.init_distributed_mode(args)

    repo_dir = os.path.dirname(os.path.realpath(__file__))
    git_info = exptrack.get_git_info(repo_dir)

    if args.output_dir:
        ckpt_path = Path(args.finetune)
        ckpt_tag = f"{ckpt_path.parent.name}_{ckpt_path.stem}"
        encoder_tag = "ema" if args.use_ema else "student"
        tag = f"knn_{ckpt_tag}_{encoder_tag}"
        args.output_dir = exptrack.make_run_dir(
            args.output_dir, tag, git_info, misc.is_main_process(),
            extra={"args": vars(args)},
        )

    print('job dir: {}'.format(repo_dir))
    print('git commit: {short_commit} (branch {branch}, dirty={dirty})'.format(**git_info))
    print('resolved output_dir: {}'.format(args.output_dir))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)
    torch.manual_seed(args.seed + misc.get_rank())
    cudnn.benchmark = True

    # deterministic, unaugmented eval transform for both banks (same as linprobe's val transform)
    transform_eval = transforms.Compose([
        transforms.Resize(256, interpolation=3),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    dataset_train = datasets.ImageFolder(os.path.join(args.data_path, 'train'), transform=transform_eval)
    dataset_val = datasets.ImageFolder(os.path.join(args.data_path, 'val'), transform=transform_eval)
    print(dataset_train)
    print(dataset_val)

    num_tasks = misc.get_world_size()
    global_rank = misc.get_rank()
    sampler_train = torch.utils.data.DistributedSampler(
        dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=False)
    sampler_val = torch.utils.data.DistributedSampler(
        dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train, batch_size=args.batch_size,
        num_workers=args.num_workers, pin_memory=args.pin_mem, drop_last=False)
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val, batch_size=args.batch_size,
        num_workers=args.num_workers, pin_memory=args.pin_mem, drop_last=False)

    model = models_vit.__dict__[args.model](num_classes=args.nb_classes, global_pool=args.global_pool)

    checkpoint = torch.load(args.finetune, map_location='cpu')
    print("Load pre-trained checkpoint from: %s" % args.finetune)
    checkpoint_model = checkpoint['model']
    if args.use_ema:
        prefix = 'ema_model.'
        ema_keys = {k[len(prefix):]: v for k, v in checkpoint_model.items() if k.startswith(prefix)}
        assert ema_keys, f"--use_ema set but no '{prefix}*' keys found in checkpoint['model']"
        print(f"Using EMA/teacher encoder weights ({len(ema_keys)} keys with prefix '{prefix}')")
        checkpoint_model = ema_keys

    interpolate_pos_embed(model, checkpoint_model)
    msg = model.load_state_dict(checkpoint_model, strict=False)
    print(msg)
    if args.global_pool:
        assert set(msg.missing_keys) == {'head.weight', 'head.bias', 'fc_norm.weight', 'fc_norm.bias'}
    else:
        assert set(msg.missing_keys) == {'head.weight', 'head.bias'}

    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    print(f"Extracting train features ({len(dataset_train)} images)...")
    t0 = time.time()
    train_features, train_labels = extract_features(model, data_loader_train, device)
    train_features = all_gather_equal(train_features)
    train_labels = all_gather_equal(train_labels)
    print(f"Train features: {tuple(train_features.shape)}, took {time.time() - t0:.1f}s")

    print(f"Extracting val features ({len(dataset_val)} images)...")
    t0 = time.time()
    val_features, val_labels = extract_features(model, data_loader_val, device)
    val_features = all_gather_equal(val_features)
    val_labels = all_gather_equal(val_labels)
    print(f"Val features: {tuple(val_features.shape)}, took {time.time() - t0:.1f}s")

    if misc.is_main_process():
        results = {}
        for k in args.nb_knn:
            top1, top5 = knn_classifier(
                train_features, train_labels, val_features, val_labels,
                k=k, T=args.temperature, num_classes=args.nb_classes,
                chunk_size=args.knn_chunk_size, device=device)
            print(f"k={k}: top1={top1:.2f}%  top5={top5:.2f}%")
            results[f"k{k}"] = {"top1": top1, "top5": top5}

        if args.output_dir:
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps({"knn_results": results}) + "\n")

    if misc.get_world_size() > 1:
        dist.barrier()


if __name__ == '__main__':
    args = get_args_parser().parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
