# Copyright (c) 2026 Makoto Yamada and contributors.
# All rights reserved.
#
# Test-time robustness of a trained linear probe: patch shuffling, removing
# the position embedding, and dropping patch tokens.
#
# Loads a finished linear-probe run (output_dir_linprobe/<run>/checkpoint-N.pth,
# which holds the frozen encoder *and* the trained BN+linear head) and
# evaluates ImageNet val top-1 under each condition. No training is involved.
#
# Conditions (result keys):
#   "<g>"        cut the image into a g x g grid and randomly permute the tiles
#                ("1" = clean). g = 14 tiles are exactly the 16x16 ViT patches,
#                so the input is a pure permutation of the patch tokens: an
#                encoder that ignored its position embedding would be exactly
#                invariant to it (self-attention is permutation equivariant and
#                the probe mean-pools the patch tokens). Coarser grids keep
#                local structure inside each tile.
#   "nopos"      zero the position embedding of the patch tokens (the CLS
#                token's is kept): content stays in place, but the encoder is
#                told nothing about where each patch is.
#   "keep<r>"    keep a random fraction r of the patch tokens and drop the rest
#                before the first block (as in the MAE encoder, dropped tokens
#                take no part in attention); the probe mean-pools the survivors.
#   "blockdrop<a>" drop the patch tokens of one contiguous square block covering
#                a fraction ~a of the image at a random location.
#   "occ<a>"     occlude the same block in pixel space (filled with 0 after
#                normalisation, i.e. the dataset mean colour); all tokens are
#                still fed to the encoder -- the closest to a real occluder.
#   "occc<a>"    as occ<a>, but the block sits at the image centre.
#   "occvis<a>"  as occ<a> (occluded tokens go through the encoder), but only the
#                visible tokens are mean-pooled: separates the harm done through
#                attention (occvis vs. blockdrop) from dilution of the pooled
#                feature by occluded tokens (occ vs. occvis).
#   "ctx<k>"     clean image through the encoder, but only k random patch tokens
#                are mean-pooled: class information carried by a few tokens
#                with the full attention context intact. (The head was trained
#                on 196-token means, so read it as a ranking across encoders.)
#   Blocks are aligned to the 16x16 patch grid; a is realised as an s x s patch
#   square with s = round(sqrt(a * 196)) (0.25/0.5/0.75 -> 7/10/12 patches).
#
# Every random choice (tile permutation, kept tokens, block location) is drawn
# per image from a fixed seed, so all encoders are evaluated on exactly the
# same inputs; blockdrop<a> and occ<a> share the same block.

import argparse
import glob
import json
import os
import re
import sys
import time
from pathlib import Path

# This script lives in probes/; add the repo root (its parent) to sys.path so
# `models_vit`, `util`, and `timm` resolve regardless of invocation directory.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms

import util.experiment_tracking as exptrack
import models_vit


def get_args_parser():
    parser = argparse.ArgumentParser('Test-time robustness of a linear probe', add_help=False)
    parser.add_argument('--probe_dir', required=True, help='finished output_dir_linprobe/<run> directory')
    parser.add_argument('--grids', default='1,2,4,7,14', type=str,
                        help='tiles per side (1 = clean); empty string to skip')
    parser.add_argument('--no_pos', action='store_true', default=True,
                        help='also evaluate with the patch position embedding zeroed')
    parser.add_argument('--skip_no_pos', action='store_false', dest='no_pos')
    parser.add_argument('--keep_ratios', default='0.75,0.5,0.25,0.1', type=str,
                        help='fractions of patch tokens to keep; empty string to skip')
    parser.add_argument('--block_areas', default='', type=str,
                        help='image fractions for blockdrop/occ/occc, e.g. 0.25,0.5,0.75; empty to skip')
    parser.add_argument('--skip_center', action='store_true', help='skip occc<a>')
    parser.add_argument('--occ_vispool', action='store_true', help='also evaluate occvis<a>')
    parser.add_argument('--ctx_pool_ks', default='', type=str,
                        help='numbers of in-context tokens to pool, e.g. 1,4,16,49; empty to skip')
    parser.add_argument('--num_val_images', default=10000, type=int, help='evenly strided subset of val')
    parser.add_argument('--model', default='vit_base_patch16', type=str)
    parser.add_argument('--nb_classes', default=1000, type=int)
    parser.add_argument('--data_path', default='/home/pj26000049/ku60000347/Python/Dataset/ImageNet/', type=str)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--num_workers', default=32, type=int)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--output_dir', default='./output_dir_shuffle')
    parser.add_argument('--device', default='cpu')
    return parser


def shuffle_tiles(x, perms, g):
    """x: [B, C, H, W]; perms: [B, g*g] -- output tile i is input tile perms[i]."""
    if g == 1:
        return x
    B, C, H, W = x.shape
    t = H // g
    tiles = x.reshape(B, C, g, t, g, t).permute(0, 2, 4, 1, 3, 5).reshape(B, g * g, C, t, t)
    tiles = torch.gather(tiles, 1, perms[:, :, None, None, None].expand(-1, -1, C, t, t))
    return tiles.reshape(B, g, g, C, t, t).permute(0, 3, 1, 4, 2, 5).reshape(B, C, H, W)


def occlude(x, blocks, s, patch=16):
    """Zero an s x s patch block per image; blocks: [B, 2] top-left (row, col) in patches."""
    x = x.clone()
    for b, (r, c) in enumerate(blocks.tolist()):
        x[b, :, r * patch:(r + s) * patch, c * patch:(c + s) * patch] = 0
    return x


def block_keep_idx(blocks, s, side):
    """Indices of the patch tokens outside each s x s block, [B, side*side - s*s]."""
    grid = torch.arange(side * side).reshape(side, side)
    out = []
    for r, c in blocks.tolist():
        inside = torch.zeros(side, side, dtype=torch.bool)
        inside[r:r + s, c:c + s] = True
        out.append(grid[~inside])
    return torch.stack(out)


def encode_tokens(model, x):
    """Final-block tokens [B, 1+L, D] of the frozen encoder (no pooling)."""
    B = x.shape[0]
    x = model.patch_embed(x)
    x = torch.cat((model.cls_token.expand(B, -1, -1), x), dim=1) + model.pos_embed
    x = model.pos_drop(x)
    for blk in model.blocks:
        x = blk(x)
    return x


def pool_head(model, tokens, sel=None):
    """Mean-pool the patch tokens (all, or those in sel [B, K]) and apply the probe."""
    assert model.global_pool, 'token pooling conditions need a global-pool probe'
    patches = tokens[:, 1:, :]
    if sel is not None:
        patches = torch.gather(patches, 1, sel[:, :, None].expand(-1, -1, patches.shape[-1]))
    return model.head(model.fc_norm(patches.mean(dim=1)))


def probe_forward(model, x, zero_patch_pos=False, keep_idx=None):
    """models_vit.VisionTransformer.forward, with the patch position embedding
    optionally zeroed and/or only the patch tokens in keep_idx [B, K] kept."""
    B = x.shape[0]
    x = model.patch_embed(x)
    pos = model.pos_embed
    if zero_patch_pos:
        pos = torch.cat([pos[:, :1], torch.zeros_like(pos[:, 1:])], dim=1)
    cls = model.cls_token.expand(B, -1, -1) + pos[:, :1]
    x = x + pos[:, 1:]
    if keep_idx is not None:
        x = torch.gather(x, 1, keep_idx[:, :, None].expand(-1, -1, x.shape[-1]))
    x = model.pos_drop(torch.cat((cls, x), dim=1))
    for blk in model.blocks:
        x = blk(x)
    if model.global_pool:
        outcome = model.fc_norm(x[:, 1:, :].mean(dim=1))
    else:
        outcome = model.norm(x)[:, 0]
    return model.head(outcome)


class IndexedSubset(torch.utils.data.Dataset):
    def __init__(self, base, indices):
        self.base, self.indices = base, indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        img, target = self.base[self.indices[i]]
        return img, target, i


def latest_checkpoint(probe_dir):
    ckpts = glob.glob(os.path.join(probe_dir, 'checkpoint-*.pth'))
    assert ckpts, f'no checkpoint-*.pth in {probe_dir}'
    return max(ckpts, key=lambda p: int(re.search(r'checkpoint-(\d+)\.pth$', p).group(1)))


def main(args):
    # leave the DataLoader workers their own cores (all 240 threads on the login
    # node trips its CPU-time limit)
    torch.set_num_threads(max(1, len(os.sched_getaffinity(0)) - args.num_workers))
    device = torch.device(args.device)
    grids = [int(g) for g in args.grids.split(',') if g]
    keep_ratios = [float(r) for r in args.keep_ratios.split(',') if r]
    block_areas = [float(a) for a in args.block_areas.split(',') if a]
    ctx_ks = [int(k) for k in args.ctx_pool_ks.split(',') if k]
    assert all(224 % g == 0 for g in grids), grids

    # ---- trained probe: frozen encoder + BN/linear head, exactly as saved ----
    ckpt_path = latest_checkpoint(args.probe_dir)
    ckpt = torch.load(ckpt_path, map_location='cpu')
    state = ckpt['model']
    global_pool = 'fc_norm.weight' in state
    model = models_vit.__dict__[args.model](num_classes=args.nb_classes, global_pool=global_pool)
    model.head = nn.Sequential(nn.BatchNorm1d(model.head.in_features, affine=False, eps=1e-6), model.head)
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    print(f'loaded {ckpt_path} (probe epoch {ckpt.get("epoch")}, global_pool={global_pool})')
    L = model.patch_embed.num_patches

    # ---- evaluation data: evenly strided val subset, shared randomness ----
    transform_val = transforms.Compose([
        transforms.Resize(256, interpolation=3),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    dataset_val = datasets.ImageFolder(os.path.join(args.data_path, 'val'), transform=transform_val)
    stride = max(1, len(dataset_val) // args.num_val_images)
    indices = list(range(0, len(dataset_val), stride))[:args.num_val_images]
    loader = torch.utils.data.DataLoader(
        IndexedSubset(dataset_val, indices), batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, drop_last=False)
    gen = torch.Generator().manual_seed(args.seed)
    perms = {g: torch.stack([torch.randperm(g * g, generator=gen) for _ in indices]) for g in grids}
    keeps = {r: torch.stack([torch.randperm(L, generator=gen)[:max(1, round(r * L))].sort().values
                             for _ in indices]) for r in keep_ratios}

    side = int(L ** 0.5)
    bgen = torch.Generator().manual_seed(args.seed + 1)  # separate stream: earlier conditions unchanged
    blocks = {}
    for a in block_areas:
        s = int(round((a * L) ** 0.5))
        blocks[a] = (s, torch.randint(0, side - s + 1, (len(indices), 2), generator=bgen))
    cgen = torch.Generator().manual_seed(args.seed + 2)  # separate stream again
    ctx_sel = {k: torch.stack([torch.randperm(L, generator=cgen)[:k] for _ in indices]) for k in ctx_ks}

    conditions = [(str(g), dict(grid=g)) for g in grids]
    if args.no_pos:
        conditions.append(('nopos', dict(zero_patch_pos=True)))
    conditions += [(f'keep{r:g}', dict(keep=r)) for r in keep_ratios]
    conditions += [(f'blockdrop{a:g}', dict(blockdrop=a)) for a in block_areas]
    conditions += [(f'occ{a:g}', dict(occ=a)) for a in block_areas]
    if not args.skip_center:
        conditions += [(f'occc{a:g}', dict(occc=a)) for a in block_areas]
    if args.occ_vispool:
        conditions += [(f'occvis{a:g}', dict(occvis=a)) for a in block_areas]
    conditions += [(f'ctx{k}', dict(ctx=k)) for k in ctx_ks]
    names = [c for c, _ in conditions]
    print('conditions:', names)

    probe_dir = Path(args.probe_dir.rstrip('/'))
    git_info = exptrack.get_git_info(os.path.dirname(os.path.realpath(__file__)))
    run_dir = exptrack.make_run_dir(args.output_dir, f'shuffle_{probe_dir.name}', git_info, True,
                                    extra={'args': vars(args), 'probe_checkpoint': ckpt_path})
    print('resolved output_dir:', run_dir)

    correct1 = {c: 0 for c in names}
    correct5 = {c: 0 for c in names}
    n, t0 = 0, time.time()
    with torch.no_grad():
        for it, (images, target, idx) in enumerate(loader):
            images, target = images.to(device), target.to(device)
            cache = {}  # encoder outputs shared between conditions of this batch

            def occ_tokens(a):
                if ('occ', a) not in cache:
                    s, bl = blocks[a]
                    cache[('occ', a)] = encode_tokens(model, occlude(images, bl[idx], s))
                return cache[('occ', a)]

            for name, cond in conditions:
                if 'ctx' in cond:
                    if 'clean' not in cache:
                        cache['clean'] = encode_tokens(model, images)
                    out = pool_head(model, cache['clean'], ctx_sel[cond['ctx']][idx].to(device))
                elif 'occvis' in cond:
                    s, bl = blocks[cond['occvis']]
                    out = pool_head(model, occ_tokens(cond['occvis']),
                                    block_keep_idx(bl[idx], s, side).to(device))
                elif 'grid' in cond:
                    out = model(shuffle_tiles(images, perms[cond['grid']][idx].to(device), cond['grid']))
                elif 'keep' in cond:
                    out = probe_forward(model, images, keep_idx=keeps[cond['keep']][idx].to(device))
                elif 'blockdrop' in cond:
                    s, bl = blocks[cond['blockdrop']]
                    out = probe_forward(model, images, keep_idx=block_keep_idx(bl[idx], s, side).to(device))
                elif 'occ' in cond:
                    out = pool_head(model, occ_tokens(cond['occ'])) if model.global_pool else \
                        model(occlude(images, blocks[cond['occ']][1][idx], blocks[cond['occ']][0]))
                elif 'occc' in cond:
                    s = blocks[cond['occc']][0]
                    ctr = torch.full((images.shape[0], 2), (side - s) // 2, dtype=torch.long)
                    out = model(occlude(images, ctr, s))
                else:
                    out = probe_forward(model, images, zero_patch_pos=True)
                hit = out.topk(5, dim=1).indices.eq(target[:, None])
                correct1[name] += hit[:, 0].sum().item()
                correct5[name] += hit.any(dim=1).sum().item()
            n += target.shape[0]
            if it % 20 == 0:
                print(f'[{n}/{len(indices)}] {time.time() - t0:.0f}s  ' +
                      '  '.join(f'{c}={100 * correct1[c] / n:.2f}' for c in names))

    res = {'probe_dir': str(probe_dir), 'probe_checkpoint': ckpt_path, 'num_val_images': n,
           'top1': {c: 100 * correct1[c] / n for c in names},
           'top5': {c: 100 * correct5[c] / n for c in names}}
    print('shuffle_robustness_results', json.dumps(res))
    with open(os.path.join(run_dir, 'log.txt'), 'a') as f:
        f.write(json.dumps({'shuffle_robustness_results': res}) + '\n')


if __name__ == '__main__':
    main(get_args_parser().parse_args())
