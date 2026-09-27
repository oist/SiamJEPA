# [CLS] -> patch attention maps of the last encoder block (as in DINO).
#
# The encoder blocks use fused attention (F.scaled_dot_product_attention),
# which does not return the weights, so for the last block they are
# recomputed here from its qkv projection.
#
# Writes two figures:
#   <out>        rows = images, columns = models; head-averaged attention
#   <out>_heads  one image, rows = models, columns = the individual heads
#
#   python probes/visualize_cls_attention.py --ckpt_file ckpts.txt \
#       --classes n02099601 ... --num_images 8 --img_size 448 --out cls_attn.pdf
#
# Checkpoint specs / image selection are the same as visualize_pca_features.py.
import argparse
import os
import sys

import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from visualize_pca_features import MEAN, STD, load_encoder, pick_images  # noqa: E402


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', action='append', default=[])
    p.add_argument('--ckpt_file', default=None)
    p.add_argument('--model', default='vit_base_patch16')
    p.add_argument('--data_path', default='/home/pj26000049/ku60000347/Python/Dataset/ImageNet/val')
    p.add_argument('--images', nargs='*', default=None)
    p.add_argument('--classes', nargs='*', default=None)
    p.add_argument('--num_images', type=int, default=8)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--img_size', type=int, default=224)
    p.add_argument('--heads_image', type=int, default=0, help='image index for the per-head figure')
    p.add_argument('--out', default='cls_attention.pdf')
    p.add_argument('--dpi', type=int, default=200)
    return p.parse_args()


@torch.no_grad()
def cls_attention(model, imgs):
    """-> [N, heads, L] attention of the [CLS] query over the patch tokens."""
    x = model.patch_embed(imgs)
    x = torch.cat((model.cls_token.expand(x.shape[0], -1, -1), x), dim=1) + model.pos_embed
    for blk in model.blocks[:-1]:
        x = blk(x)
    blk = model.blocks[-1]
    attn = blk.attn
    B, N, C = x.shape
    qkv = attn.qkv(blk.norm1(x)).reshape(B, N, 3, attn.num_heads, C // attn.num_heads)
    q, k = qkv.permute(2, 0, 3, 1, 4)[:2]
    w = ((q @ k.transpose(-2, -1)) * attn.scale).softmax(dim=-1)
    return w[:, :, 0, 1:]


def upsample(maps, grid, size):
    """[..., L] -> [..., size, size], each map min-max normalized."""
    shape = maps.shape[:-1]
    m = maps.reshape(-1, 1, grid, grid)
    lo = m.amin(dim=(2, 3), keepdim=True)
    hi = m.amax(dim=(2, 3), keepdim=True)
    m = (m - lo) / (hi - lo + 1e-12)
    m = F.interpolate(m, size=(size, size), mode='bilinear', align_corners=False)
    return m.reshape(*shape, size, size)


def style(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def main():
    args = get_args()
    if args.ckpt_file:
        with open(args.ckpt_file) as f:
            args.ckpt += [l.strip() for l in f if l.strip() and not l.lstrip().startswith('#')]
    assert args.ckpt, "give --ckpt and/or --ckpt_file"
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    tf = T.Compose([T.Resize(int(args.img_size * 256 / 224), interpolation=T.InterpolationMode.BICUBIC),
                    T.CenterCrop(args.img_size), T.ToTensor()])
    paths = pick_images(args)
    raw = torch.stack([tf(Image.open(p).convert('RGB')) for p in paths])
    imgs = T.Normalize(MEAN, STD)(raw)
    N, grid = len(paths), args.img_size // 16

    labels, mean_maps, head_maps = [], [], []
    for spec in args.ckpt:
        label, model = load_encoder(spec, args)
        a = cls_attention(model, imgs)  # [N, H, L]
        labels.append(label)
        mean_maps.append(upsample(a.mean(1), grid, args.img_size))
        head_maps.append(upsample(a[args.heads_image], grid, args.img_size))
        print(f"{label}: {spec.rsplit('=', 1)[1]}", flush=True)

    # head-averaged maps, all images
    ncol = 1 + len(labels)
    fig, axes = plt.subplots(N, ncol, figsize=(2.0 * ncol, 2.0 * N), squeeze=False)
    for i in range(N):
        axes[i, 0].imshow(raw[i].permute(1, 2, 0).numpy())
        for j, m in enumerate(mean_maps):
            axes[i, j + 1].imshow(m[i].numpy(), cmap='inferno', vmin=0, vmax=1)
        for ax in axes[i]:
            style(ax)
    axes[0, 0].set_title('Image', fontsize=12)
    for j, label in enumerate(labels):
        axes[0, j + 1].set_title(label, fontsize=12)
    plt.tight_layout(pad=0.2, w_pad=0.1, h_pad=0.1)
    base = os.path.splitext(args.out)[0]
    fig.savefig(args.out, dpi=args.dpi, bbox_inches='tight')
    fig.savefig(base + '.png', dpi=args.dpi, bbox_inches='tight')
    plt.close(fig)

    # per-head maps, one image
    H = head_maps[0].shape[0]
    fig, axes = plt.subplots(len(labels), H + 1, figsize=(1.6 * (H + 1), 1.6 * len(labels)), squeeze=False)
    for r, (label, hm) in enumerate(zip(labels, head_maps)):
        axes[r, 0].imshow(raw[args.heads_image].permute(1, 2, 0).numpy())
        axes[r, 0].set_ylabel(label, fontsize=9, rotation=0, ha='right', va='center')
        for h in range(H):
            axes[r, h + 1].imshow(hm[h].numpy(), cmap='inferno', vmin=0, vmax=1)
            if r == 0:
                axes[r, h + 1].set_title(f'head {h}', fontsize=9)
        for ax in axes[r]:
            style(ax)
    plt.tight_layout(pad=0.2, w_pad=0.1, h_pad=0.1)
    fig.savefig(base + '_heads.pdf', dpi=args.dpi, bbox_inches='tight')
    fig.savefig(base + '_heads.png', dpi=args.dpi, bbox_inches='tight')
    with open(base + '_images.txt', 'w') as f:
        f.write('\n'.join(paths) + '\n')
    print(f"wrote {args.out}, {base}_heads.pdf (+ .png, image list)")


if __name__ == '__main__':
    main()
