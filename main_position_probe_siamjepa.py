# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright (c) 2026 Makoto Yamada and contributors.
# All rights reserved.
#
# This file is based on the Meta MAE implementation and has been
# substantially modified for the SiamJEPA project.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# --------------------------------------------------------
# Position-decodability probe.
#
# Tests whether a frozen encoder's final-layer PATCH tokens (not the
# pooled/CLS feature used by linprobe/kNN) still encode which of the
# 14x14 grid positions they came from. Motivation: the pre-fix
# forward_encoder shuffled patch order before the teacher/EMA branch was
# compared against the predictor's output, so the training signal never
# rewarded keeping patch content tied to a specific grid position. If
# that pushed the model toward a more DINO/BYOL-style "global content,
# position-agnostic" representation, position-decoding accuracy should
# be measurably lower for a buggy-code checkpoint than a corrected one
# at a comparable point in training.
#
# Method: extract every patch token from the last transformer block
# (no pooling, no final norm -- global_pool checkpoints only ever
# trained a norm for the pooled feature, not per-patch use) for a set
# of train images and a disjoint set of held-out images. Every image
# contributes the same 196 (position, token) pairs since patch order is
# a fixed raster scan out of patch_embed. Fit a single linear layer
# (768 -> 196) by cross-entropy to predict position from the token, on
# the train images' patches; report top-1/top-5 accuracy on the held-out
# images' patches. Chance level is 1/196 ~= 0.51%.
# --------------------------------------------------------

import argparse
import json
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import util.misc as misc
from util.pos_embed import interpolate_pos_embed
import util.experiment_tracking as exptrack

import models_vit


def get_args_parser():
    parser = argparse.ArgumentParser('SiamJEPA position-decodability probe', add_help=False)
    parser.add_argument('--batch_size', default=256, type=int,
                        help='Batch size per GPU/process for feature extraction')
    parser.add_argument('--model', default='vit_base_patch16', type=str, metavar='MODEL')

    parser.add_argument('--finetune', required=True,
                        help='checkpoint to evaluate (a SiamJEPA pretrain checkpoint-N.pth)')
    parser.add_argument('--use_ema', action='store_true',
                        help='Evaluate the EMA/teacher encoder instead of the student encoder '
                             '(same semantics as the flag in main_linprobe_siamjepa.py). This is '
                             'the branch forward_encoder (and the pre-fix shuffle bug) applies to.')
    parser.add_argument('--global_pool', action='store_true')
    parser.set_defaults(global_pool=True)

    parser.add_argument('--data_path', default='/home/pj26000049/ku60000347/Python/Dataset/ImageNet/', type=str)
    parser.add_argument('--num_train_images', default=2000, type=int,
                        help='Images used to fit the position-probe classifier.')
    parser.add_argument('--num_val_images', default=500, type=int,
                        help='Disjoint held-out images used to report accuracy.')
    parser.add_argument('--probe_epochs', default=30, type=int)
    parser.add_argument('--probe_lr', default=1e-3, type=float)
    parser.add_argument('--probe_batch_size', default=4096, type=int,
                        help='Minibatch size (in patches, not images) for training the linear probe.')

    parser.add_argument('--output_dir', default='./output_dir_posprobe',
                        help='path where to save, empty for no saving. '
                             'A run-specific subdirectory (job id/timestamp + git commit + '
                             'evaluated checkpoint) is created under this path for each run.')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed (kept for parity with the other eval scripts; --device cpu skips all of it)
    parser.add_argument('--world_size', default=1, type=int)
    parser.add_argument('--local_rank', '--local-rank', type=int, default=-1)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://')
    return parser


@torch.no_grad()
def extract_patch_tokens(model, x):
    """Return every patch token (no CLS, no pooling, no final norm) from the
    last transformer block: shape [B, num_patches, embed_dim]."""
    B = x.shape[0]
    x = model.patch_embed(x)
    cls_tokens = model.cls_token.expand(B, -1, -1)
    x = torch.cat((cls_tokens, x), dim=1)
    x = x + model.pos_embed
    x = model.pos_drop(x)
    for blk in model.blocks:
        x = blk(x)
    return x[:, 1:, :]


@torch.no_grad()
def extract_all_patches(model, data_loader, device, num_images):
    """Collect [n_images, num_patches, embed_dim] token features, stopping
    once num_images images have been seen. Labels are just arange(num_patches)
    repeated per image (fixed raster order out of patch_embed)."""
    feats = []
    seen = 0
    for imgs, _ in data_loader:
        imgs = imgs.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type):
            tokens = extract_patch_tokens(model, imgs)
        feats.append(tokens.float().cpu())
        seen += imgs.shape[0]
        if seen >= num_images:
            break
    feats = torch.cat(feats, dim=0)[:num_images]
    return feats  # [n_images, num_patches, embed_dim]


def make_position_dataset(patch_feats):
    """[n_images, P, D] -> (X [n_images*P, D], y [n_images*P]) with y = position id."""
    n_images, num_patches, embed_dim = patch_feats.shape
    X = patch_feats.reshape(n_images * num_patches, embed_dim)
    y = torch.arange(num_patches).repeat(n_images)
    return X, y


def train_position_probe(train_X, train_y, val_X, val_y, num_patches, embed_dim,
                          epochs, lr, batch_size, device):
    probe = nn.Linear(embed_dim, num_patches).to(device)
    opt = torch.optim.Adam(probe.parameters(), lr=lr)
    train_X = train_X.to(device)
    train_y = train_y.to(device)
    val_X = val_X.to(device)
    val_y = val_y.to(device)

    n = train_X.shape[0]
    for epoch in range(epochs):
        perm = torch.randperm(n, device=device)
        total_loss = 0.0
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            xb, yb = train_X[idx], train_y[idx]
            logits = probe(xb)
            loss = F.cross_entropy(logits, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item() * xb.shape[0]
        print(f"  probe epoch {epoch}: train_loss={total_loss / n:.4f}")

    probe.eval()
    with torch.no_grad():
        val_logits = probe(val_X)
        top1 = (val_logits.argmax(dim=-1) == val_y).float().mean().item() * 100
        top5 = (val_logits.topk(5, dim=-1).indices == val_y.unsqueeze(-1)).any(dim=-1).float().mean().item() * 100
    return top1, top5


def main(args):
    misc.init_distributed_mode(args)

    repo_dir = os.path.dirname(os.path.realpath(__file__))
    git_info = exptrack.get_git_info(repo_dir)

    if args.output_dir:
        ckpt_path = Path(args.finetune)
        ckpt_tag = f"{ckpt_path.parent.name}_{ckpt_path.stem}"
        encoder_tag = "ema" if args.use_ema else "student"
        tag = f"posprobe_{ckpt_tag}_{encoder_tag}"
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

    transform_eval = transforms.Compose([
        transforms.Resize(256, interpolation=3),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    dataset_val = datasets.ImageFolder(os.path.join(args.data_path, 'val'), transform=transform_eval)
    print(dataset_val)

    total_needed = args.num_train_images + args.num_val_images
    indices = list(range(min(total_needed, len(dataset_val))))
    subset = torch.utils.data.Subset(dataset_val, indices)
    loader = torch.utils.data.DataLoader(
        subset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=args.pin_mem)

    model = models_vit.__dict__[args.model](num_classes=1000, global_pool=args.global_pool)

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

    num_patches = model.patch_embed.num_patches
    embed_dim = model.pos_embed.shape[-1]
    print(f"num_patches={num_patches}  embed_dim={embed_dim}")

    print(f"Extracting patch tokens for {total_needed} images...")
    t0 = time.time()
    all_feats = extract_all_patches(model, loader, device, total_needed)
    print(f"Extracted {all_feats.shape}, took {time.time() - t0:.1f}s")

    train_feats = all_feats[:args.num_train_images]
    val_feats = all_feats[args.num_train_images:args.num_train_images + args.num_val_images]

    train_X, train_y = make_position_dataset(train_feats)
    val_X, val_y = make_position_dataset(val_feats)
    print(f"train patches: {train_X.shape}, val patches: {val_X.shape}")

    top1, top5 = train_position_probe(
        train_X, train_y, val_X, val_y, num_patches, embed_dim,
        args.probe_epochs, args.probe_lr, args.probe_batch_size, device)

    chance = 100.0 / num_patches
    print(f"position-probe top1={top1:.2f}%  top5={top5:.2f}%  (chance level ~{chance:.2f}%)")

    if args.output_dir and misc.is_main_process():
        with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
            f.write(json.dumps({
                "position_probe_results": {
                    "top1": top1, "top5": top5, "chance_level": chance,
                    "num_patches": num_patches,
                    "num_train_images": args.num_train_images,
                    "num_val_images": args.num_val_images,
                }
            }) + "\n")


if __name__ == '__main__':
    args = get_args_parser().parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
