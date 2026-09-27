# How much does SiamJEPA's stochastic latent z change the predictor's output?
#
# For each image, two disjoint views are masked exactly as in training (the
# same masks for every checkpoint). The predictor gets view1's context plus a
# latent z sampled S times from either the posterior (which also sees view2's
# [CLS]) or the prior (view1 only). For every predicted position we measure
#   spread_j = 1 - || mean_s normalize(pred_{s,j}) ||   (0: z has no effect)
# and the prediction error of the mean prediction against the EMA teacher,
#   err_j = 1 - cos(mean_s pred_{s,j}, teacher_j).
# Also logs KL(posterior || prior) and both entropies per image.
#
#   python probes/visualize_latent_uncertainty.py --ckpt_file ckpts.txt \
#       --images a.JPEG b.JPEG ... --samples 64 --out latent_uncertainty.pdf
#
# ckpt specs: "Label=/path/checkpoint.pth" (full SiamJEPA checkpoints; the
# student predictor/posterior/prior are needed, so no ":ema").
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import models_siamjepa  # noqa: E402

MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', action='append', default=[])
    p.add_argument('--ckpt_file', default=None)
    p.add_argument('--images', nargs='+', required=True)
    p.add_argument('--mask_ratio', type=float, default=0.75)
    p.add_argument('--samples', type=int, default=64, help='latent samples per image')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--out', default='latent_uncertainty.pdf')
    p.add_argument('--dpi', type=int, default=200)
    return p.parse_args()


def load_model(spec, mask_ratio):
    label, path = spec.rsplit('=', 1)
    label = label.replace('\\n', '\n')
    # these checkpoints were trained with view2 reusing view1's ids_restore;
    # only the view1 predictor is used here, which is the same either way
    model = models_siamjepa.siamjepa_vit_base_patch16(
        kl_scale=0.01, beta=0.99, mask_ratio=mask_ratio, fix_view2_restore=False)
    sd = torch.load(path, map_location='cpu')
    model.load_state_dict(sd.get('model', sd), strict=True)
    return label, model.eval()


@torch.no_grad()
def encode_views(model, imgs, k1, k2):
    x = model.patch_embed(imgs) + model.pos_embed[:, 1:, :]
    D = x.shape[-1]
    cls = (model.cls_token + model.pos_embed[:, :1, :]).expand(x.shape[0], -1, -1)
    x1 = torch.cat((cls, torch.gather(x, 1, k1.unsqueeze(-1).expand(-1, -1, D))), 1)
    x2 = torch.cat((cls, torch.gather(x, 1, k2.unsqueeze(-1).expand(-1, -1, D))), 1)
    x12 = torch.cat((x1, x2), 0)
    for blk in model.blocks:
        x12 = blk(x12)
    return model.norm(x12).chunk(2, 0)


@torch.no_grad()
def analyze(model, img, ids_restore, len_keep, samples):
    ids_shuffle = torch.argsort(ids_restore, dim=1)
    k1, k2 = ids_shuffle[:, :len_keep], ids_shuffle[:, len_keep:2 * len_keep]
    h, p = encode_views(model, img, k1, k2)
    teacher = model.ema_model.forward_encoder(img, mask_ratio=0)[0][:, 1:]

    h_ca3 = model.ca3(h)
    h_cls = model.ca3(model.projector(h[:, 0]))
    post_logits = model.to_posterior(torch.cat([h_cls, model.projector(p[:, 0])], -1)).float().clamp(-20, 20)
    prior_logits = model.to_prior(h_cls).float().clamp(-20, 20)
    out = {}
    _, kl = model.kl_loss(post_logits, prior_logits)
    out['kl'] = kl.item()
    for name, logits in (('posterior', post_logits), ('prior', prior_logits)):
        dist = model.make_dist(logits)
        out[f'entropy_{name}'] = dist.entropy().mean().item()
        z = dist.sample((samples,)).squeeze(1)  # [S, stoch, discrete]
        pred = model.forward_predictor(h_ca3.expand(samples, -1, -1),
                                       ids_restore.expand(samples, -1), z)  # [S, L, D]
        pn = F.normalize(pred.float(), dim=-1)
        spread = 1 - pn.mean(0).norm(dim=-1)  # [L]
        err = 1 - F.cosine_similarity(pred.float().mean(0), teacher[0].float(), dim=-1)
        out[f'spread_{name}'] = spread
        out[f'err_{name}'] = err
    keep1 = torch.zeros(ids_restore.shape[1], dtype=torch.bool)
    keep1[k1[0]] = True
    keep2 = torch.zeros_like(keep1)
    keep2[k2[0]] = True
    out['keep1'], out['keep2'] = keep1, keep2
    return out


def to_map(v, grid, size):
    m = v.reshape(1, 1, grid, grid)
    return F.interpolate(m, size=(size, size), mode='nearest')[0, 0].numpy()


def main():
    args = get_args()
    if args.ckpt_file:
        with open(args.ckpt_file) as f:
            args.ckpt += [l.strip() for l in f if l.strip() and not l.lstrip().startswith('#')]
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    tf = T.Compose([T.Resize(256, interpolation=T.InterpolationMode.BICUBIC), T.CenterCrop(224), T.ToTensor()])
    raw = torch.stack([tf(Image.open(p).convert('RGB')) for p in args.images])
    imgs = T.Normalize(MEAN, STD)(raw)
    N, grid, L = len(args.images), 14, 196
    len_keep = int(L * (1 - args.mask_ratio))

    # one pair of disjoint training-style masks per image, shared by all checkpoints
    masks = []
    for i in range(N):
        torch.manual_seed(args.seed + i)
        _, _, _, _, ids_restore = models_siamjepa.SiamJEPA.random_masking_dual(
            None, torch.zeros(1, L, 1), args.mask_ratio)
        masks.append(ids_restore)

    results, labels = [], []
    for spec in args.ckpt:
        label, model = load_model(spec, args.mask_ratio)
        torch.manual_seed(args.seed)
        res = [analyze(model, imgs[i:i + 1], masks[i], len_keep, args.samples) for i in range(N)]
        results.append(res)
        labels.append(label)
        print(label.replace('\n', ' '), ' '.join(
            f"[img{i}: KL={r['kl']:.3f} H_post={r['entropy_posterior']:.2f} H_prior={r['entropy_prior']:.2f} "
            f"spread_post={r['spread_posterior'][~r['keep1']].mean():.4f} "
            f"spread_prior={r['spread_prior'][~r['keep1']].mean():.4f}]" for i, r in enumerate(res)), flush=True)

    # shared color scale for spread (over masked positions of all models / both latents)
    vmax = max(float(r[f'spread_{n}'][~r['keep1']].max()) for res in results for r in res
               for n in ('posterior', 'prior'))
    emax = max(float(r[f'err_{n}'][~r['keep1']].max()) for res in results for r in res
               for n in ('posterior',))

    def context_img(i, keep1):
        img = raw[i].permute(1, 2, 0).numpy().copy()
        m = to_map(keep1.float(), grid, 224)[..., None]
        return img * (0.25 + 0.75 * m)

    def masked_map(v, keep1):
        m = to_map(v, grid, 224)
        return np.ma.masked_where(to_map(keep1.float(), grid, 224) > 0.5, m)

    cmap = plt.get_cmap('viridis').copy()
    cmap.set_bad(color='0.85')
    base = os.path.splitext(args.out)[0]

    # summary: rows = images, columns = context + posterior spread per model
    ncol = 1 + len(labels)
    fig, axes = plt.subplots(N, ncol, figsize=(2.0 * ncol, 2.0 * N), squeeze=False)
    for i in range(N):
        keep1 = results[0][i]['keep1']
        axes[i, 0].imshow(context_img(i, keep1))
        for j, res in enumerate(results):
            im = axes[i, j + 1].imshow(masked_map(res[i]['spread_posterior'], keep1), cmap=cmap, vmin=0, vmax=vmax)
        for ax in axes[i]:
            ax.set_xticks([]); ax.set_yticks([])
    axes[0, 0].set_title('view-1 context', fontsize=11)
    for j, label in enumerate(labels):
        axes[0, j + 1].set_title(label, fontsize=11)
    fig.colorbar(im, ax=axes, fraction=0.015, pad=0.01, label='spread over z (posterior)')
    fig.savefig(args.out, dpi=args.dpi, bbox_inches='tight')
    fig.savefig(base + '.png', dpi=args.dpi, bbox_inches='tight')
    plt.close(fig)

    # detail: per model, columns = context, posterior spread, prior spread, error
    fig, axes = plt.subplots(N, 1 + 3 * len(labels), figsize=(1.7 * (1 + 3 * len(labels)), 1.8 * N), squeeze=False)
    for i in range(N):
        keep1 = results[0][i]['keep1']
        axes[i, 0].imshow(context_img(i, keep1))
        for j, res in enumerate(results):
            r = res[i]
            c = 1 + 3 * j
            axes[i, c].imshow(masked_map(r['spread_posterior'], keep1), cmap=cmap, vmin=0, vmax=vmax)
            axes[i, c + 1].imshow(masked_map(r['spread_prior'], keep1), cmap=cmap, vmin=0, vmax=vmax)
            axes[i, c + 2].imshow(masked_map(r['err_posterior'], keep1), cmap='magma', vmin=0, vmax=emax)
            if i == 0:
                short = labels[j].replace('\n', ' ')
                axes[0, c].set_title(f'{short}\nspread (post.)', fontsize=7)
                axes[0, c + 1].set_title('spread (prior)', fontsize=7)
                axes[0, c + 2].set_title('error (post. mean)', fontsize=7)
        for ax in axes[i]:
            ax.set_xticks([]); ax.set_yticks([])
    axes[0, 0].set_title('view-1 context', fontsize=7)
    fig.savefig(base + '_detail.pdf', dpi=args.dpi, bbox_inches='tight')
    fig.savefig(base + '_detail.png', dpi=args.dpi, bbox_inches='tight')
    plt.close(fig)

    summary = {}
    for label, res in zip(labels, results):
        summary[label.replace('\n', ' ')] = {
            'kl': float(np.mean([r['kl'] for r in res])),
            'entropy_posterior': float(np.mean([r['entropy_posterior'] for r in res])),
            'entropy_prior': float(np.mean([r['entropy_prior'] for r in res])),
            'spread_posterior_masked': float(np.mean([r['spread_posterior'][~r['keep1']].mean() for r in res])),
            'spread_prior_masked': float(np.mean([r['spread_prior'][~r['keep1']].mean() for r in res])),
            'err_posterior_masked': float(np.mean([r['err_posterior'][~r['keep1']].mean() for r in res])),
            'err_prior_masked': float(np.mean([r['err_prior'][~r['keep1']].mean() for r in res])),
        }
    with open(base + '_summary.json', 'w') as f:
        json.dump({'images': args.images, 'vmax_spread': vmax, 'models': summary}, f, indent=1)
    print(json.dumps(summary, indent=1))
    print(f"wrote {args.out}, {base}_detail.pdf, {base}_summary.json")


if __name__ == '__main__':
    main()
