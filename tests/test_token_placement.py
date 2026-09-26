# Regression tests for patch-position bookkeeping in SiamJEPA.
#
# Both bugs found so far (the shuffled teacher in forward_encoder, and view2's
# context tokens being placed in view1's predictor slots) left the loss and
# every tensor shape looking normal, so they are checked here directly.
#
#   python tests/test_token_placement.py      (or: pytest tests/)
# CPU only, ~1 min.
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import models_siamjepa  # noqa: E402

GRID = 14
L = GRID * GRID
MASK_RATIO = 0.75


def small_model(**kwargs):
    # ViT-B width (the tagged-token checks below assume D=768); depth is cut
    # to keep this fast
    return models_siamjepa.SiamJEPA(
        embed_dim=768, depth=1, num_heads=12,
        decoder_embed_dim=768, decoder_depth=1, decoder_num_heads=12,
        kl_scale=0.01, beta=0.99, mask_ratio=MASK_RATIO, **kwargs)


def test_random_masking_dual_invariants():
    torch.manual_seed(0)
    N = 256
    len_keep = int(L * (1 - MASK_RATIO))
    x = torch.arange(L).float().view(1, L, 1).repeat(N, 1, 4)
    x1, x2, mask1, mask2, ids_restore = models_siamjepa.SiamJEPA.random_masking_dual(
        None, x, MASK_RATIO)
    keep1, keep2 = mask1 == 0, mask2 == 0

    assert (keep1.sum(1) == len_keep).all() and (keep2.sum(1) == len_keep).all()
    assert not (keep1 & keep2).any(), "the two views must not share visible patches"
    # the gathered tokens are exactly the patches the masks mark as visible
    for xv, keep in ((x1, keep1), (x2, keep2)):
        got = xv[..., 0].long().sort(1).values
        want = keep.nonzero()[:, 1].view(N, len_keep)
        assert torch.equal(got, want)
    # the masked-in-both (target) set contains a full square block
    tgt = (mask1 * mask2).view(N, 1, GRID, GRID)
    bsz = int((L - 2 * len_keep) ** 0.5)
    pooled = torch.nn.functional.avg_pool2d(tgt, bsz, stride=1)
    assert (pooled.flatten(1).max(1).values == 1).all()


def _predictor_inputs(model):
    """Run model.forward() with position-tagged encoder outputs and return, per
    view, the predictor-block input (channel 0 = grid position + 1 of the token
    placed in each slot, -1 for mask tokens) plus both views' visible masks."""
    N, D = 4, 768
    captured = []
    with torch.no_grad():
        model.ca3[0].weight.copy_(torch.eye(D))
        model.ca3[0].bias.zero_()
        model.decoder_embed_deter.weight.copy_(torch.eye(D))
        model.decoder_embed_deter.bias.zero_()
        model.decoder_pos_embed.zero_()
        model.mask_token.fill_(-1.0)
    masks = {}

    def tagged_encoder_dual(imgs, mask_ratio):
        _, _, mask1, mask2, ids_restore = model.random_masking_dual(
            torch.zeros(N, L, D), mask_ratio)
        ids_shuffle = torch.argsort(ids_restore, dim=1)
        len_keep = int(L * (1 - mask_ratio))
        views = []
        for ids_keep in (ids_shuffle[:, :len_keep], ids_shuffle[:, len_keep:2 * len_keep]):
            h = torch.zeros(N, 1 + len_keep, D)
            h[:, 1:, 0] = ids_keep.float() + 1  # each token carries its true grid position
            views.append(h)
        masks['keep1'], masks['keep2'] = mask1 == 0, mask2 == 0
        return views[0], views[1], mask1, mask2, ids_restore

    model.forward_encoder_dual = tagged_encoder_dual
    hook = model.decoder_blocks[0].register_forward_pre_hook(
        lambda mod, args: captured.append(args[0][:, 1:, 0].clone()))
    model.eval()
    with torch.no_grad():
        model(torch.randn(N, 3, 224, 224))
    hook.remove()
    assert len(captured) == 2  # view1 predictor call, then view2
    return captured, masks


def _placement_ok(slots, keep):
    positions = torch.arange(L).float().expand_as(slots) + 1
    return torch.equal(slots[keep], positions[keep]) and bool((slots[~keep] == -1).all())


def test_each_view_placed_at_its_own_positions():
    torch.manual_seed(0)
    (slots1, slots2), masks = _predictor_inputs(small_model())
    assert _placement_ok(slots1, masks['keep1']), "view1 tokens not at their own grid slots"
    assert _placement_ok(slots2, masks['keep2']), "view2 tokens not at their own grid slots"


def test_legacy_mode_reproduces_old_view2_placement():
    # fix_view2_restore=False is kept only to reproduce pre-fix runs: view2's
    # tokens land in view1's slots. If this starts passing _placement_ok, the
    # legacy switch no longer does what its name says.
    torch.manual_seed(0)
    (slots1, slots2), masks = _predictor_inputs(small_model(fix_view2_restore=False))
    assert _placement_ok(slots1, masks['keep1'])
    assert not _placement_ok(slots2, masks['keep2'])
    assert bool((slots2[masks['keep1']] > 0).all()), "legacy: view2 tokens should sit in view1 slots"


def test_teacher_keeps_patch_order():
    # forward_encoder (EMA teacher) must return tokens in grid order unless
    # Random Shuffle Teacher is requested explicitly
    torch.manual_seed(0)
    imgs = torch.randn(2, 3, 224, 224)
    for shuffle, expect_same in ((False, True), (True, False)):
        model = small_model(shuffle_teacher=shuffle).eval()
        with torch.no_grad():
            z, _, ids_restore = model.ema_model.forward_encoder(imgs, mask_ratio=0)
            # reference: the same blocks applied to the unshuffled token sequence
            x = model.ema_model.patch_embed(imgs) + model.ema_model.pos_embed[:, 1:, :]
            cls = (model.ema_model.cls_token + model.ema_model.pos_embed[:, :1, :]).expand(2, -1, -1)
            x = torch.cat((cls, x), dim=1)
            for blk in model.ema_model.blocks:
                x = blk(x)
            ref = model.ema_model.norm(x)
        same = torch.allclose(z, ref, atol=1e-5)
        assert same == expect_same, f"shuffle_teacher={shuffle}: teacher patch order wrong"
        assert torch.equal(ids_restore[0], torch.arange(L))


if __name__ == '__main__':
    tests = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL {t.__name__}: {e}")
    print(f"{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
