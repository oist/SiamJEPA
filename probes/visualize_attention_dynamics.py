# [CLS] attention over the course of pretraining, next to the KL (Sim-1) curve.
#
# Top: train_loss_sim1 per epoch (from each run's log.txt; the free-bit floor
# is the horizontal dashed line) with the visualized epochs marked.
# Below: head-averaged last-block [CLS]->patch attention of the EMA/teacher
# encoder at those epochs; one row per (run, image).
#
#   python probes/visualize_attention_dynamics.py \
#       --run "λ=0.01=./output_dir_siamjepa/<run_dir_a>" \
#       --run "λ=1e-5=./output_dir_siamjepa/<run_dir_b>" \
#       --epochs 5 25 50 75 100 150 200 300 399 --images a.JPEG b.JPEG \
#       --out attention_dynamics.pdf
#
# A run spec may also carry an epoch offset, its own epochs to visualize, and
# a last epoch for its KL curve: "Label=DIR@OFFSET#e1,e2,...<LAST" (epochs may
# also be separated by ':', which survives `pjsub -x`). With
# --concat, all runs are laid out on one total-epoch axis (e.g. the two stages
# of the RST curriculum), one row per image.
import argparse
import re
import json
import os
import sys

import torch
import torchvision.transforms as T
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from visualize_pca_features import MEAN, STD, load_encoder  # noqa: E402
from visualize_cls_attention import cls_attention, upsample  # noqa: E402


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--run', action='append', required=True, help='"Label=/path/to/run_dir"')
    p.add_argument('--epochs', type=int, nargs='+', default=None,
                   help='epochs to visualize (runs may override with #e1,e2,...)')
    p.add_argument('--concat', action='store_true',
                   help='one total-epoch timeline for all runs (run offsets via @OFFSET)')
    p.add_argument('--images', nargs='+', required=True)
    p.add_argument('--model', default='vit_base_patch16')
    p.add_argument('--img_size', type=int, default=448)
    p.add_argument('--free_bit', type=float, default=0.1)
    p.add_argument('--out', default='attention_dynamics.pdf')
    p.add_argument('--dpi', type=int, default=200)
    return p.parse_args()


def main():
    args = get_args()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    tf = T.Compose([T.Resize(int(args.img_size * 256 / 224), interpolation=T.InterpolationMode.BICUBIC),
                    T.CenterCrop(args.img_size), T.ToTensor()])
    raw = torch.stack([tf(Image.open(p).convert('RGB')) for p in args.images])
    imgs = T.Normalize(MEAN, STD)(raw)
    grid = args.img_size // 16

    runs = []
    for spec in args.run:
        label, rest = spec.rsplit('=', 1)
        m = re.match(r'([^@#<]+)(?:@(-?\d+))?(?:#([\d,:]+))?(?:<(\d+))?$', rest)
        run_dir, offset, ep_list, last = m.groups()
        offset = int(offset or 0)
        epochs = [int(e) for e in re.split('[,:]', ep_list)] if ep_list else args.epochs
        with open(os.path.join(run_dir, 'log.txt')) as f:
            log = [json.loads(l) for l in f if l.strip().startswith('{')]
        kl = {r['epoch'] + offset: r['train_loss_sim1'] for r in log
              if last is None or r['epoch'] <= int(last)}
        maps = {}
        for e in epochs:
            ckpt = os.path.join(run_dir, f'checkpoint-{e}.pth')
            _, model = load_encoder(f"{label}={ckpt}:ema", args)
            maps[e + offset] = upsample(cls_attention(model, imgs).mean(1), grid, args.img_size)  # [N, S, S]
            print(f"{label} epoch {e} (total {e + offset}): loaded {ckpt}", flush=True)
        runs.append((label, kl, maps))

    n_img = len(args.images)
    if args.concat:
        # columns = every (run, total epoch), in timeline order; one row per image
        columns = sorted(((e, k) for k, (_, _, maps) in enumerate(runs) for e in maps))
        rows = [(None, i) for i in range(n_img)]
    else:
        columns = [(e, None) for e in sorted(runs[0][2])]
        rows = [(k, i) for k in range(len(runs)) for i in range(n_img)]
    n_ep, nrows = len(columns), len(rows)
    # layout in units of one image cell: KL plot, an empty spacer row (room for
    # the KL x-axis and the column titles), then the image grid
    u = 1.55
    spacer = 0.8 if args.concat else 0.65
    fig = plt.figure(figsize=(u * (n_ep + 1), u * (1.6 + spacer + nrows) * 1.03))
    outer = GridSpec(3, 1, figure=fig, height_ratios=[1.6, spacer, nrows], hspace=0)
    top = outer[0].subgridspec(1, n_ep + 1, wspace=0.04)
    gs = outer[2].subgridspec(nrows, n_ep + 1, hspace=0.06, wspace=0.04)

    ax = fig.add_subplot(top[0, 1:])
    colors = plt.get_cmap('tab10').colors
    for k, (label, kl, _) in enumerate(runs):
        ep = sorted(kl)
        ax.plot(ep, [kl[e] for e in ep], color=colors[k], lw=1.6, label=label)
    ax.axhline(args.free_bit, color='0.4', ls='--', lw=1, label='free-bit floor')
    for e, _ in columns:
        ax.axvline(e, color='0.8', lw=0.8, zorder=0)
    ax.set_yscale('log')
    ax.set_xlim(0, max(max(kl) for _, kl, _ in runs))
    if args.concat and len(runs) > 1:
        for _, kl, _ in runs[1:]:
            ax.axvline(min(kl), color='k', lw=1, ls=':')
    ax.set_xlabel('epoch', fontsize=10)
    ax.set_ylabel('loss$_{\\mathrm{sim1}}$ (KL)', fontsize=10)
    ax.legend(fontsize=8.5, ncol=len(runs) + 1, loc='upper right')
    ax.tick_params(labelsize=9)

    for r, (k_row, i) in enumerate(rows):
        a0 = fig.add_subplot(gs[r, 0])
        a0.imshow(raw[i].permute(1, 2, 0).numpy())
        if k_row is not None:
            a0.set_ylabel(runs[k_row][0], fontsize=9)
        a0.set_xticks([]); a0.set_yticks([])
        for j, (e, k_col) in enumerate(columns):
            k = k_row if k_row is not None else k_col
            a = fig.add_subplot(gs[r, j + 1])
            a.imshow(runs[k][2][e][i].numpy(), cmap='inferno', vmin=0, vmax=1)
            a.set_xticks([]); a.set_yticks([])
            if r == 0:
                title = f'ep {e}' if k_col is None else f'ep {e}\n{runs[k][0]}'
                a.set_title(title, fontsize=8, color=colors[k] if k_col is not None else 'k')

    base = os.path.splitext(args.out)[0]
    fig.savefig(args.out, dpi=args.dpi, bbox_inches='tight')
    fig.savefig(base + '.png', dpi=args.dpi, bbox_inches='tight')
    print(f"wrote {args.out} (+ .png)")


if __name__ == '__main__':
    main()
