# Pretraining throughput / memory benchmark on synthetic data.
#
# Mirrors the real training step (engine_pretrain.py): DDP, no_sync gradient
# accumulation, GradScaler +
# clip_grad=3.0, AdamW, EMA update after every optimizer step. Data loading
# is left out on purpose (real runs are not data-bound: data ~0.0004 s/it).
#
#   torchrun --nproc_per_node=4 bench/bench_pretrain_speed.py --precision bf16
#
# --repo points at the checkout whose models_siamjepa.py / timm / util are
# imported, so the same script can time an older commit:
#   torchrun ... bench/bench_pretrain_speed.py --repo ../SiamJEPA-dev --precision fp32
import argparse
import os
import sys
import time


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--repo', default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    p.add_argument('--model', default='siamjepa_vit_base_patch16')
    p.add_argument('--batch_size', default=512, type=int, help='per-GPU micro-batch')
    p.add_argument('--accum_iter', default=4, type=int)
    p.add_argument('--precision', default='fp32', choices=['fp32', 'tf32', 'bf16'])
    p.add_argument('--bf16_predictor', action='store_true')
    p.add_argument('--warmup', default=8, type=int, help='micro-steps not timed')
    p.add_argument('--iters', default=24, type=int, help='micro-steps timed')
    p.add_argument('--tag', default='')
    p.add_argument('--find_unused_parameters', action='store_true',
                   help='DDP setting of older checkouts (they had never-used modules); '
                        'turned on automatically when the model still has them')
    return p.parse_args()


def main():
    args = get_args()
    sys.path.insert(0, os.path.abspath(args.repo))
    import torch
    import torch.distributed as dist
    import models_siamjepa
    from util.misc import NativeScalerWithGradNormCount as NativeScaler
    import timm.optim.optim_factory as optim_factory

    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    distributed = 'WORLD_SIZE' in os.environ
    if distributed:
        dist.init_process_group('nccl')
    torch.cuda.set_device(local_rank)
    device = torch.device('cuda', local_rank)
    torch.manual_seed(0)
    torch.backends.cudnn.benchmark = True
    if args.precision in ('tf32', 'bf16'):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    use_bf16 = args.precision == 'bf16'

    model = models_siamjepa.__dict__[args.model](kl_scale=0.01, beta=0.99, mask_ratio=0.75)
    if args.bf16_predictor:
        for m in model.modules():
            if isinstance(m, models_siamjepa.CSABlock):
                m.force_fp32 = False
    model.to(device)
    n_student = sum(p.numel() for n, p in model.named_parameters() if not n.startswith('ema_model.'))
    n_ema = sum(p.numel() for p in model.ema_model.parameters())

    find_unused = args.find_unused_parameters or any(
        hasattr(model, n) for n in ('decoder_pred', 'decoder_embed_mae'))

    model_without_ddp = model
    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], find_unused_parameters=find_unused)
        model_without_ddp = model.module
    param_groups = optim_factory.add_weight_decay(model_without_ddp, 0.1)
    optimizer = torch.optim.AdamW(param_groups, lr=1e-4, betas=(0.9, 0.95))
    loss_scaler = NativeScaler()

    samples = torch.randn(args.batch_size, 3, 224, 224, device=device)
    model.train(True)
    torch.cuda.reset_peak_memory_stats(device)

    def step(i):
        is_update = (i + 1) % args.accum_iter == 0
        ctx = model.no_sync() if (distributed and not is_update) else torch.enable_grad()
        with ctx:
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=use_bf16):
                loss, _, loss_sim1, loss_sim2 = model(samples)
            loss_value = loss.item()
            loss_scaler(loss / args.accum_iter, optimizer, parameters=model.parameters(),
                        clip_grad=3.0, update_grad=is_update)
        if is_update:
            optimizer.zero_grad(set_to_none=True)
            model_without_ddp.update_ema_model()
        return loss_value

    for i in range(args.warmup):
        step(i)
    torch.cuda.synchronize()
    t0 = time.time()
    losses = []
    for i in range(args.warmup, args.warmup + args.iters):
        losses.append(step(i))
    torch.cuda.synchronize()
    dt = (time.time() - t0) / args.iters
    mem = torch.cuda.max_memory_allocated(device) / 2**30

    if local_rank == 0:
        world = dist.get_world_size() if distributed else 1
        # 1,281,167 ImageNet train images, drop_last per rank
        its_per_epoch = 1281167 // (args.batch_size * world)
        print(f"RESULT tag={args.tag or args.precision} repo={os.path.basename(os.path.abspath(args.repo))} "
              f"model={args.model} bs={args.batch_size}x{args.accum_iter}x{world} "
              f"precision={args.precision} bf16_predictor={args.bf16_predictor} "
              f"s/it={dt:.4f} epoch_min={dt * its_per_epoch / 60:.1f} max_mem_GiB={mem:.1f} "
              f"params_student={n_student / 1e6:.1f}M params_ema={n_ema / 1e6:.1f}M "
              f"last_loss={losses[-1]:.4f}", flush=True)
    if distributed:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
