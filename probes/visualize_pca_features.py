# PCA visualization of frozen patch features (as in DINOv2 / VISReg figures).
#
# For each checkpoint: take the final-layer (post-norm) patch tokens of a set
# of images, fit a 3-component PCA jointly over all images' patches, map the
# components to RGB, and upsample to the image. Components of every model are
# matched to the first model's (Hungarian on |correlation|, with sign flips),
# so the same color means the same direction across columns.
#
#   python probes/visualize_pca_features.py \
#       --ckpt "JEPA-like=/path/checkpoint-399.pth:ema" \
#       --ckpt "SiamJEPA=/path/checkpoint-399.pth:ema" \
#       --num_images 8 --img_size 224 --out pca_features.pdf
#
# CPU is fine (a few images per model). ":ema" loads the EMA/teacher encoder
# (ema_model.* keys), otherwise the student encoder.
import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import models_vit  # noqa: E402
from util.pos_embed import interpolate_pos_embed  # noqa: E402

MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', action='append', default=[],
                   help='"Label=/path/checkpoint.pth[:ema]"; repeat per model (first = reference)')
    p.add_argument('--ckpt_file', default=None,
                   help='file with one --ckpt spec per line (after any --ckpt given directly); '
                        'blank lines and lines starting with # are skipped')
    p.add_argument('--model', default='vit_base_patch16')
    p.add_argument('--data_path', default='/home/pj26000049/ku60000347/Python/Dataset/ImageNet/val')
    p.add_argument('--images', nargs='*', default=None, help='explicit image paths')
    p.add_argument('--num_images', type=int, default=8, help='random val images if --images not given')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--img_size', type=int, default=224,
                   help='input resolution; >224 interpolates the positional embedding '
                        '(finer patch grid)')
    p.add_argument('--remove_pos_mean', type=int, default=0, metavar='K',
                   help='subtract each grid position\'s mean feature, estimated on K other '
                        'random val images, before the PCA (removes the positional component '
                        'that otherwise dominates the top PCs of position-heavy models)')
    p.add_argument('--pca_mode', default='joint', choices=['joint', 'per_image'],
                   help='joint: one PCA over all images (same color = same direction across '
                        'images); per_image: a PCA per image (shows parts within an image)')
    p.add_argument('--fg_mask', action='store_true',
                   help='per_image only: split foreground/background on the first PC (sign '
                        'chosen so the image border is background, DINOv2-style) and color only '
                        'the foreground with a PCA fitted on it')
    p.add_argument('--fg_thresh', type=float, default=0.5,
                   help='foreground = min-max-normalized first PC above this')
    p.add_argument('--classes', nargs='*', default=None,
                   help='draw the random images from these ImageNet synsets only')
    p.add_argument('--clip_pct', type=float, default=1.0,
                   help='per-component percentile clip before min-max scaling to [0, 1]')
    p.add_argument('--out', default='pca_features.pdf')
    p.add_argument('--dpi', type=int, default=200)
    return p.parse_args()


def pick_images(args, n=None, seed=None, use_explicit=True):
    if use_explicit and args.images:
        return args.images
    rng = np.random.RandomState(args.seed if seed is None else seed)
    classes = args.classes if (args.classes and use_explicit) else sorted(os.listdir(args.data_path))
    paths = []
    for c in rng.choice(classes, n or args.num_images, replace=n is not None and n > len(classes)):
        files = sorted(os.listdir(os.path.join(args.data_path, c)))
        paths.append(os.path.join(args.data_path, c, files[rng.randint(len(files))]))
    return paths


def load_encoder(spec, args):
    label, rest = spec.rsplit('=', 1)  # labels may contain '=' (paths do not)
    label = label.replace('\\n', '\n')  # a literal \n in the label breaks the column title
    use_ema = rest.endswith(':ema')
    path = rest[:-4] if use_ema else rest
    model = models_vit.__dict__[args.model](num_classes=1000, global_pool=False, img_size=args.img_size)
    sd = torch.load(path, map_location='cpu')
    sd = sd.get('model', sd)  # our checkpoints wrap the state dict; public ones (e.g. DINO) do not
    if use_ema:
        sd = {k[len('ema_model.'):]: v for k, v in sd.items() if k.startswith('ema_model.')}
        assert sd, f"{label}: no ema_model.* keys in {path}"
    interpolate_pos_embed(model, sd)
    msg = model.load_state_dict(sd, strict=False)
    assert set(msg.missing_keys) <= {'head.weight', 'head.bias'}, (label, msg.missing_keys)
    return label, model.eval()


@torch.no_grad()
def patch_features(model, imgs):
    x = model.patch_embed(imgs)
    cls = model.cls_token.expand(x.shape[0], -1, -1)
    x = torch.cat((cls, x), dim=1) + model.pos_embed
    for blk in model.blocks:
        x = blk(x)
    x = model.norm(x)
    return x[:, 1:]  # [N, L, C]


def pca3(feats):
    """feats [M, C] -> projections [M, 3] onto the top-3 principal directions."""
    f = feats - feats.mean(0, keepdim=True)
    _, _, V = torch.linalg.svd(f, full_matrices=False)
    return f @ V[:3].T


def align_to(ref, proj):
    """Permute / sign-flip proj's 3 columns to best match ref's (by correlation)."""
    r = (ref - ref.mean(0)) / ref.std(0)
    p = (proj - proj.mean(0)) / proj.std(0)
    corr = (r.T @ p) / r.shape[0]  # [3 ref, 3 proj]
    rows, cols = linear_sum_assignment(-corr.abs().numpy())
    out = torch.empty_like(proj)
    for i, j in zip(rows, cols):
        out[:, i] = proj[:, j] * torch.sign(corr[i, j])
    return out


def to_rgb(proj, clip_pct, ref_rows=None):
    r = proj if ref_rows is None else proj[ref_rows]
    lo = torch.quantile(r, clip_pct / 100, dim=0)
    hi = torch.quantile(r, 1 - clip_pct / 100, dim=0)
    return ((proj - lo) / (hi - lo)).clamp(0, 1)


def pca_fit(feats, k=3):
    mean = feats.mean(0, keepdim=True)
    _, _, V = torch.linalg.svd(feats - mean, full_matrices=False)
    return mean, V[:k]


def border_mask(grid):
    m = torch.zeros(grid, grid, dtype=torch.bool)
    m[0, :] = m[-1, :] = m[:, 0] = m[:, -1] = True
    return m.flatten()


def per_image_rgb(feats, args, grid, ref=None):
    """feats [N, L, C] -> (rgb [N, L, 3], projections, foreground masks).
    ref: the reference model's (projections, fg masks) to align components to."""
    border = border_mask(grid)
    rgbs, projs, fgs = [], [], []
    for i, f in enumerate(feats):
        fg = torch.ones(f.shape[0], dtype=torch.bool)
        if args.fg_mask:
            mean, V = pca_fit(f, 1)
            p1 = ((f - mean) @ V.T)[:, 0]
            if p1[border].mean() > p1[~border].mean():
                p1 = -p1  # border = background
            p1 = (p1 - p1.min()) / (p1.max() - p1.min())
            cand = p1 > args.fg_thresh
            if cand.sum() >= 8:
                fg = cand
        mean, V = pca_fit(f[fg])
        proj = (f - mean) @ V.T
        if ref is not None:
            proj = align_to(ref[0][i], proj)
        rgb = to_rgb(proj, args.clip_pct, ref_rows=fg)
        rgb[~fg] = 1.0  # background white
        rgbs.append(rgb)
        projs.append(proj)
        fgs.append(fg)
    return torch.stack(rgbs), projs, fgs


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
    N = len(paths)
    grid = args.img_size // 16
    if args.remove_pos_mean:
        mean_paths = pick_images(args, n=args.remove_pos_mean, seed=args.seed + 1000, use_explicit=False)
        mean_imgs = T.Normalize(MEAN, STD)(
            torch.stack([tf(Image.open(p).convert('RGB')) for p in mean_paths]))

    ref = None
    columns = []
    for spec in args.ckpt:
        label, model = load_encoder(spec, args)
        feats = patch_features(model, imgs)  # [N, L, C]
        if args.remove_pos_mean:
            pos_mean = torch.cat([patch_features(model, b) for b in mean_imgs.split(32)]).mean(0)
            feats = feats - pos_mean
        if args.pca_mode == 'per_image':
            rgb, projs, fgs = per_image_rgb(feats, args, grid, ref)
            if ref is None:
                ref = (projs, fgs)
            rgb = rgb.reshape(N, grid, grid, 3).permute(0, 3, 1, 2)
        else:
            proj = pca3(feats.reshape(-1, feats.shape[-1]))
            if ref is None:
                ref = proj
            else:
                proj = align_to(ref, proj)
            rgb = to_rgb(proj, args.clip_pct).reshape(N, grid, grid, 3).permute(0, 3, 1, 2)
        rgb = F.interpolate(rgb, size=(args.img_size, args.img_size), mode='bilinear',
                            align_corners=False)
        columns.append((label, rgb.permute(0, 2, 3, 1).numpy()))
        print(f"{label}: {spec.rsplit('=', 1)[1]}", flush=True)

    ncol = 1 + len(columns)
    fig, axes = plt.subplots(N, ncol, figsize=(2.0 * ncol, 2.0 * N), squeeze=False)
    for i in range(N):
        axes[i, 0].imshow(raw[i].permute(1, 2, 0).numpy())
        for j, (label, rgb) in enumerate(columns):
            axes[i, j + 1].imshow(rgb[i])
        for ax in axes[i]:
            ax.set_xticks([])
            ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
    axes[0, 0].set_title('Image', fontsize=12)
    for j, (label, _) in enumerate(columns):
        axes[0, j + 1].set_title(label, fontsize=12)
    plt.tight_layout(pad=0.2, w_pad=0.1, h_pad=0.1)
    fig.savefig(args.out, dpi=args.dpi, bbox_inches='tight')
    png = os.path.splitext(args.out)[0] + '.png'
    if png != args.out:
        fig.savefig(png, dpi=args.dpi, bbox_inches='tight')
    with open(os.path.splitext(args.out)[0] + '_images.txt', 'w') as f:
        f.write('\n'.join(paths) + '\n')
    print(f"wrote {args.out} (+ .png, image list)")


if __name__ == '__main__':
    main()
