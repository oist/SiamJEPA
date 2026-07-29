# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import math
import sys
from typing import Iterable

import torch

import util.misc as misc
import util.lr_sched as lr_sched


def train_one_epoch_siamjepa(model: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    log_writer=None,
                    args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('loss_sim1', misc.SmoothedValue(window_size=20, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_sim2', misc.SmoothedValue(window_size=20, fmt='{value:.4f}'))
    metric_logger.add_meter('ema_beta', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter
    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    model_without_ddp = model.module if hasattr(model, "module") else model

    num_steps = len(data_loader)
    updates_per_epoch = (num_steps + accum_iter - 1) // accum_iter  # ceil

    total_updates = updates_per_epoch * args.epochs

    update_count = epoch * updates_per_epoch

    for data_iter_step, (samples, _) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(
                optimizer,
                data_iter_step / len(data_loader) + epoch,
                args
            )

        samples = samples.to(device, non_blocking=True)

        is_update_step = ((data_iter_step + 1) % accum_iter == 0) or ((data_iter_step + 1) == num_steps)

        maybe_no_sync = (model.no_sync if (hasattr(model, "no_sync") and not is_update_step) else None)

        if maybe_no_sync is None:
            with torch.cuda.amp.autocast(enabled=False):
                loss, _, loss_sim1, loss_sim2 = model(samples)

            loss_value = loss.item()
            if not math.isfinite(loss_value):
                print(f"Loss is {loss_value}, stopping training")
                sys.exit(1)

            loss = loss / accum_iter
            loss_scaler(
                loss, optimizer, parameters=model.parameters(),
                clip_grad=3.0, update_grad=True
            )
            optimizer.zero_grad(set_to_none=True)

            threshold1 = 0.5 * total_updates
            threshold2 = 0.75 * total_updates

            if update_count < threshold1:
                cur_beta = args.ema[0]
                cur_mask_ratio = args.mask_ratio[0]

            elif update_count < threshold2:
                cur_beta = args.ema[1]
                cur_mask_ratio = args.mask_ratio[1]

            else:
                cur_beta = args.ema[2]
                cur_mask_ratio = args.mask_ratio[2]

            model_without_ddp.beta = cur_beta
            model_without_ddp.mask_ratio = cur_mask_ratio
            model_without_ddp.update_ema_model()
            update_count += 1

        else:
            with maybe_no_sync():
                with torch.cuda.amp.autocast(enabled=False):
                    loss, _, loss_sim1, loss_sim2 = model(samples)

                loss_value = loss.item()
                if not math.isfinite(loss_value):
                    print(f"Loss is {loss_value}, stopping training")
                    sys.exit(1)

                loss = loss / accum_iter
                loss_scaler(
                    loss, optimizer, parameters=model.parameters(),
                    clip_grad=3.0, update_grad=False
                )

            cur_beta = model_without_ddp.beta  # ログ用

        metric_logger.update(loss=loss_value)
        metric_logger.update(loss_sim1=loss_sim1.item())
        metric_logger.update(loss_sim2=loss_sim2.item())
        metric_logger.update(ema_beta=cur_beta)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        loss_sim1_reduce = misc.all_reduce_mean(loss_sim1.item())
        loss_sim2_reduce = misc.all_reduce_mean(loss_sim2.item())
        beta_reduce = misc.all_reduce_mean(cur_beta)

        if log_writer is not None and is_update_step:
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('loss_sim1', loss_sim1_reduce, epoch_1000x)
            log_writer.add_scalar('loss_sim2', loss_sim2_reduce, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)
            log_writer.add_scalar('ema_beta', beta_reduce, epoch_1000x)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

