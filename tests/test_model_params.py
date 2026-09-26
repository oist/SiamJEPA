# Checks on the parameter set of SiamJEPA:
#  - every trainable parameter gets a gradient each step, which is what lets
#    main_pretrain_siamjepa.py use DDP(find_unused_parameters=False);
#  - checkpoints written before the unused modules were removed (and before
#    the EMA copy became encoder-only) still load, including --resume's
#    optimizer state.
#
#   python tests/test_model_params.py      (or: pytest tests/)
# CPU only.
import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import models_siamjepa  # noqa: E402
import timm.optim.optim_factory as optim_factory  # noqa: E402


def small_model(**kwargs):
    # embed_dim stays 768 (projection_MLP's output width is fixed at 768)
    return models_siamjepa.SiamJEPA(
        embed_dim=768, depth=1, num_heads=12,
        decoder_embed_dim=768, decoder_depth=1, decoder_num_heads=12,
        kl_scale=0.01, beta=0.99, mask_ratio=0.75, **kwargs)


def test_all_trainable_params_get_grads():
    for kwargs in ({}, {'fix_view2_restore': False}, {'shuffle_teacher': True}):
        torch.manual_seed(0)
        model = small_model(**kwargs).train()
        loss, *_ = model(torch.randn(4, 3, 224, 224))
        loss.backward()
        missing = [n for n, p in model.named_parameters() if p.requires_grad and p.grad is None]
        assert not missing, f"{kwargs}: parameters without gradient: {missing}"


class _LegacySiamJEPA(models_siamjepa.SiamJEPA):
    """The model as it was before REMOVED_MODULES were deleted and the EMA copy
    was trimmed: same modules plus the never-used ones, registered in their
    original places, and a full EMA copy."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        d, dd = 768, 768
        children = dict(self._modules)
        self._modules.clear()
        extra = {'decoder_embed': nn.Linear(d, dd),
                 'decoder_pred_latent': nn.Linear(dd, d),
                 'decoder_pred': nn.Linear(dd, 16 * 16 * 3),
                 'decoder_embed_mae': nn.Linear(d, dd)}
        for name, mod in children.items():
            if name == 'decoder_blocks':
                self._modules['decoder_embed'] = extra['decoder_embed']
            if name == 'decoder_embed_deter':
                self._modules['decoder_embed_mae'] = extra['decoder_embed_mae']
            self._modules[name] = mod
            if name == 'decoder_norm':
                self._modules['decoder_pred_latent'] = extra['decoder_pred_latent']
                self._modules['decoder_pred'] = extra['decoder_pred']
        # full (untrimmed) EMA copy, as old checkpoints have it
        ema = self._modules.pop('ema_model')
        full = _copy_without_ema(self)
        full.load_state_dict({k: v for k, v in ema.state_dict().items()}, strict=False)
        for p in full.parameters():
            p.requires_grad = False
        self._modules['ema_model'] = full


def _copy_without_ema(model):
    import copy
    ema = model._modules.pop('ema_model', None)
    full = copy.deepcopy(model)
    if ema is not None:
        model._modules['ema_model'] = ema
    return full


def test_legacy_checkpoint_resume():
    torch.manual_seed(0)
    old = _LegacySiamJEPA(embed_dim=768, depth=1, num_heads=12, decoder_embed_dim=768,
                          decoder_depth=1, decoder_num_heads=12, kl_scale=0.01,
                          beta=0.99, mask_ratio=0.75)
    old_opt = torch.optim.AdamW(optim_factory.add_weight_decay(old, 0.1), lr=1e-3)
    loss, *_ = old(torch.randn(4, 3, 224, 224))
    loss.backward()
    old_opt.step()
    ckpt_model, ckpt_opt = old.state_dict(), old_opt.state_dict()
    assert any(k.startswith('decoder_pred.') for k in ckpt_model)
    assert any(k.startswith('ema_model.projector.') for k in ckpt_model)

    new = small_model()
    new.load_state_dict(ckpt_model, strict=True)
    new_opt = torch.optim.AdamW(optim_factory.add_weight_decay(new, 0.1), lr=1e-3)
    new_opt.load_state_dict(new.convert_legacy_optimizer_state(ckpt_opt, ckpt_model))

    # every kept parameter got its own Adam moments back
    old_params = dict(old.named_parameters())
    for name, p in new.named_parameters():
        if not p.requires_grad:
            continue
        s_new = new_opt.state[p]
        s_old = old_opt.state[old_params[name]]
        assert torch.equal(s_new['exp_avg'], s_old['exp_avg']), name
        assert torch.equal(s_new['exp_avg_sq'], s_old['exp_avg_sq']), name
        assert torch.equal(p, old_params[name]), name
    # and the model trains on from there
    loss, *_ = new(torch.randn(4, 3, 224, 224))
    loss.backward()
    new_opt.step()


def test_current_checkpoint_optimizer_untouched():
    torch.manual_seed(0)
    model = small_model()
    opt = torch.optim.AdamW(optim_factory.add_weight_decay(model, 0.1), lr=1e-3)
    sd = opt.state_dict()
    assert model.convert_legacy_optimizer_state(sd, model.state_dict()) is sd


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
