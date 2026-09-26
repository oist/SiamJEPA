# Checks that the speed/memory changes do not change what is computed.
# Compares this checkout's models_siamjepa.py against an older checkout
# (--old_repo) on a real checkpoint. Single GPU:
#
#   python bench/check_speedup_equivalence.py --old_repo ../SiamJEPA-dev \
#       --checkpoint ../SiamJEPA-dev/output_dir_siamjepa/<run>/checkpoint-399.pth
import argparse
import importlib.util
import os
import sys

import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import models_siamjepa as new_mod  # noqa: E402
import models_vit  # noqa: E402
from timm.models.vision_transformer import Attention, CrossAttention  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument('--old_repo', required=True)
p.add_argument('--checkpoint', required=True)
p.add_argument('--batch', default=16, type=int)
args = p.parse_args()

spec = importlib.util.spec_from_file_location(
    'models_siamjepa_old', os.path.join(args.old_repo, 'models_siamjepa.py'))
old_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(old_mod)

dev = torch.device('cuda')
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
results = []


def report(name, ok, detail):
    results.append(ok)
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}", flush=True)


def rel(a, b):
    return ((a - b).abs().max() / b.abs().max().clamp_min(1e-12)).item()


# ---------------------------------------------------------------- 1. SDPA
torch.manual_seed(0)
att = Attention(768, num_heads=12, qkv_bias=True).to(dev).eval()
x = torch.randn(8, 50, 768, device=dev)
with torch.no_grad():
    B, N, C = x.shape
    qkv = att.qkv(x).reshape(B, N, 3, 12, C // 12).permute(2, 0, 3, 1, 4)
    a = ((qkv[0] @ qkv[1].transpose(-2, -1)) * att.scale).softmax(-1)
    ref = att.proj((a @ qkv[2]).transpose(1, 2).reshape(B, N, C))
    out = att(x)
report("Attention SDPA == manual (fp32)", rel(out, ref) < 1e-5, f"max rel diff {rel(out, ref):.2e}")

catt = CrossAttention(768, num_heads=16, qkv_bias=True).to(dev).eval()
kvx = torch.randn(8, 51, 768, device=dev)
with torch.no_grad():
    kv = catt.kv(kvx).reshape(B, 51, 2, 16, C // 16).permute(2, 0, 3, 1, 4)
    q = catt.q(x).reshape(B, N, 1, 16, C // 16).permute(2, 0, 3, 1, 4)[0]
    a = ((q @ kv[0].transpose(-2, -1)) * catt.scale).softmax(-1)
    ref = catt.proj((a @ kv[1]).transpose(1, 2).reshape(B, N, C))
    out = catt(x, kvx)
report("CrossAttention SDPA == manual (fp32)", rel(out, ref) < 1e-5, f"max rel diff {rel(out, ref):.2e}")

# ------------------------------------------------------- 2. masking statistics
L, D, mr = 196, 8, 0.75
len_keep = int(L * (1 - mr))
xs = torch.zeros(2048, L, D, device=dev)
freq = {'new': torch.zeros(3, L, device=dev), 'old': torch.zeros(3, L, device=dev)}
inv_ok = True
reps = 10
for _ in range(reps):
    for name, fn in (('new', lambda t: new_mod.SiamJEPA.random_masking_dual(None, t, mr)),
                     ('old', lambda t: old_mod.SiamJEPA.random_masking_dual(None, t, mr))):
        _, _, m1, m2, ids_restore = fn(xs)
        keep1, keep2 = (m1 == 0), (m2 == 0)
        tgt = m1 * m2
        if name == 'new':
            inv_ok &= bool((keep1.sum(1) == len_keep).all() and (keep2.sum(1) == len_keep).all())
            inv_ok &= bool(~(keep1 & keep2).any())
            inv_ok &= bool((tgt.sum(1) == L - 2 * len_keep).all())
            # masked-in-both set must contain a full block_size x block_size square
            bsz = int((L - 2 * len_keep) ** 0.5)
            t2 = tgt.view(-1, 1, 14, 14)
            pooled = torch.nn.functional.avg_pool2d(t2, bsz, stride=1)
            inv_ok &= bool((pooled.flatten(1).max(1).values == 1).all())
        freq[name] += torch.stack([keep1.float().mean(0), keep2.float().mean(0), tgt.mean(0)])
report("masking invariants (sizes, disjoint views, square block masked in both)", inv_ok, "")
fd = (freq['new'] - freq['old']).abs().max().item() / reps
report("masking per-position frequencies new ~= old", fd < 0.03,
       f"max |freq diff| {fd:.4f} over {reps * 2048} samples (sampling noise ~0.01)")

# ----------------------------------------- 3. full forward / backward / EMA
ckpt = torch.load(args.checkpoint, map_location='cpu')['model']
old = old_mod.siamjepa_vit_base_patch16(kl_scale=0.01, beta=0.99, mask_ratio=0.75).to(dev)
new = new_mod.siamjepa_vit_base_patch16(kl_scale=0.01, beta=0.99, mask_ratio=0.75).to(dev)
old.load_state_dict(ckpt, strict=True)
new.load_state_dict(ckpt, strict=True)  # drops stale ema_model.* keys
report("old full-EMA checkpoint loads strictly into encoder-only EMA model", True, "")

n_ema_old = sum(p.numel() for p in old.ema_model.parameters())
n_ema_new = sum(p.numel() for p in new.ema_model.parameters())
report("EMA params", n_ema_new < n_ema_old,
       f"{n_ema_old / 1e6:.1f}M -> {n_ema_new / 1e6:.1f}M "
       f"({(n_ema_old - n_ema_new) * 4 / 2**20:.0f} MiB fp32 saved)")

torch.manual_seed(1)
imgs = torch.randn(args.batch, 3, 224, 224, device=dev)
with torch.no_grad():
    fixed = new_mod.SiamJEPA.random_masking_dual(
        None, new.patch_embed(imgs) + new.pos_embed[:, 1:, :], 0.75)
_, _, fm1, fm2, fids = fixed


def fixed_masking(self_model):
    def f(x, mask_ratio):
        N, L, D = x.shape
        ids_shuffle = torch.argsort(fids, dim=1)
        k1, k2 = ids_shuffle[:, :len_keep], ids_shuffle[:, len_keep:2 * len_keep]
        return (torch.gather(x, 1, k1.unsqueeze(-1).repeat(1, 1, D)),
                torch.gather(x, 1, k2.unsqueeze(-1).repeat(1, 1, D)), fm1, fm2, fids)
    return f


old.random_masking_dual = fixed_masking(old)
new.random_masking_dual = fixed_masking(new)
old.train()
new.train()

outs = {}
for name, m in (('old', old), ('new', new)):
    torch.manual_seed(123)  # same posterior rsample
    loss, _, s1, s2 = m(imgs)
    loss.backward()
    outs[name] = (loss.detach(), s1.detach(), s2.detach(),
                  {n: p.grad.detach().clone() for n, p in m.named_parameters() if p.grad is not None})
lo, ln = outs['old'], outs['new']
report("loss (fp32) new == old", rel(ln[0], lo[0]) < 1e-4,
       f"old={lo[0].item():.6f} new={ln[0].item():.6f} (sim1 {lo[1].item():.5f}/{ln[1].item():.5f}, "
       f"sim2 {lo[2].item():.6f}/{ln[2].item():.6f})")
gd = max(rel(ln[3][n], lo[3][n]) for n in lo[3] if n in ln[3] and lo[3][n].abs().max() > 0)
report("gradients (fp32) new == old", gd < 1e-3 and set(lo[3]) == set(ln[3]),
       f"max rel grad diff {gd:.2e} over {len(lo[3])} tensors")

with torch.no_grad():
    for m in (old, new):
        for n, p_ in m.named_parameters():
            if p_.grad is not None:
                p_.add_(p_.grad, alpha=-1e-2)
        m.beta = 0.99
        m.update_ema_model()
    oe = dict(old.ema_model.named_parameters())
    ed = max(rel(p_, oe[n]) for n, p_ in new.ema_model.named_parameters())
report("EMA update (encoder) new == old", ed < 1e-6, f"max rel diff {ed:.2e}")

# ------------------------------------------------------------- 4. bf16 drift
new.zero_grad(set_to_none=True)
with torch.no_grad():
    torch.manual_seed(123)
    lf, _, s1f, s2f = new(imgs)
    for pred_bf16 in (False, True):
        for m in new.modules():
            if isinstance(m, new_mod.CSABlock):
                m.force_fp32 = not pred_bf16
        torch.manual_seed(123)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            lb, _, s1b, s2b = new(imgs)
        d = abs(lb.item() - lf.item()) / abs(lf.item())
        report(f"bf16 loss close to fp32 (predictor {'bf16' if pred_bf16 else 'fp32'})", d < 1e-2,
               f"fp32={lf.item():.5f} bf16={lb.item():.5f} rel diff {d:.2e} "
               f"(sim2 {s2f.item():.5f}/{s2b.item():.5f})")
for m in new.modules():
    if isinstance(m, new_mod.CSABlock):
        m.force_fp32 = True

# ------------------------------------------- 5. linprobe --use_ema loading path
sd = new.state_dict()
ema_keys = {k[len('ema_model.'):]: v for k, v in sd.items() if k.startswith('ema_model.')}
vit = models_vit.vit_base_patch16(num_classes=1000, global_pool=True)
msg = vit.load_state_dict(ema_keys, strict=False)
report("linprobe --use_ema: encoder-only EMA keys load into ViT-B",
       set(msg.missing_keys) == {'head.weight', 'head.bias', 'fc_norm.weight', 'fc_norm.bias'},
       f"{len(ema_keys)} keys, missing={msg.missing_keys}")

print(f"\n{sum(results)}/{len(results)} checks passed")
sys.exit(0 if all(results) else 1)
