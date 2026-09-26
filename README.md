# SiamJEPA

SiamJEPA is a self-supervised visual representation learning method that combines a **Siamese student/teacher encoder** with a **JEPA-style (Joint-Embedding Predictive Architecture) predictive objective**, regularized with a **KL term** between student and teacher latent distributions. The teacher is updated as an exponential moving average (EMA) of the student, following a momentum schedule.

The codebase is built on top of Meta's [MAE](https://github.com/facebookresearch/mae) implementation and reuses conventions from [DeiT](https://github.com/facebookresearch/deit), [BEiT](https://github.com/microsoft/unilm/tree/master/beit), and [MoCo v3](https://github.com/facebookresearch/moco-v3).

## Paper

Makoto Yamada, "SiamJEPA: On the Role of Siamese Student Encoders in JEPA," arXiv:2607.04044.
[https://arxiv.org/abs/2607.04044](https://arxiv.org/abs/2607.04044)

```bibtex
@article{yamada2026siamjepa,
  title   = {SiamJEPA: On the Role of Siamese Student Encoders in JEPA},
  author  = {Yamada, Makoto},
  journal = {arXiv preprint arXiv:2607.04044},
  year    = {2026}
}
```

## Repository layout

**Core model and pretraining**

| Path | Description |
|---|---|
| `models_siamjepa.py` | SiamJEPA model definitions (Siamese ViT encoders, cross-attention decoder/predictor, projection MLP, EMA teacher update, KL regularization, Random Shuffle Teacher). |
| `models_vit.py` | Plain ViT backbones (`vit_small_patch16`, `vit_base_patch16`, `vit_large_patch16`, `vit_huge_patch14`, ...) used for downstream evaluation. |
| `main_pretrain_siamjepa.py` | Entry point for self-supervised pretraining. |
| `engine_pretrain.py` | Training loop for pretraining (`train_one_epoch_siamjepa`). |
| `run_pretrain.sh` | Example PJM batch script for multi-GPU pretraining via `torchrun`. |
| `run_pretrain_jepalike.sh` | Historical record of the `--kl_scale 1e-5` "JEPA-like" run discussed in the paper; use the parameterized `run_pretrain.sh` for new runs instead. |

**Main evaluation** — the headline linear-probe numbers reported in the paper (Table 1).

| Path | Description |
|---|---|
| `main_linprobe_siamjepa.py` | Standard linear probe: freezes the encoder, trains a BatchNorm + linear head with LARS. Takes `--finetune <checkpoint-path>`. |
| `engine_finetune.py` | Shared training/evaluation loop (`train_one_epoch`, `evaluate`) used by every linear-probe script, including the ones under `probes/`. |
| `run_linprobe_siamjepa.sh` | PJM batch script for `main_linprobe_siamjepa.py`. Takes `FINETUNE=<checkpoint>` and optional `USE_EMA=1` (probe the EMA/teacher instead of the student) via `pjsub -x`. |
| `check_and_submit_linprobe.sh` | Watches a pretraining run's `output_dir` and auto-submits `run_linprobe_siamjepa.sh` whenever a new milestone checkpoint appears. |
| `collect_results.py` | Aggregates linear-probe log files into a single `results.csv`. |

**`probes/`** — supplementary probes and ablations used in the paper's analysis sections, not needed to reproduce the headline Table 1 numbers. Each `main_*.py` here takes `--finetune <checkpoint-path> [--use_ema]`, same convention as `main_linprobe_siamjepa.py`, except `main_shuffle_robustness.py` (see below).

| Path | Description |
|---|---|
| `probes/main_linprobe_ijepa.py` | Linear probe under I-JEPA's own evaluation protocol (avg-pooled/4-layer-concat features, best of a probe-head grid) — for the head-to-head comparison against I-JEPA/DSeq-JEPA. |
| `probes/main_position_probe_siamjepa.py` | Position-decodability probe: predicts a frozen patch token's original grid position (out of 196), used to quantify the RST semantic/spatial trade-off. |
| `probes/main_knn_siamjepa.py` | Weighted k-NN evaluation (DINO-style), a cheaper alternative to linear probing. |
| `probes/main_shuffle_robustness.py` | Test-time robustness sweep (no retraining): patch shuffle, random token drop, and mean-colour occlusion, evaluated on an *already-trained* linear-probe head (`--probe_dir <output_dir_linprobe/run...>`, not `--finetune`). |
| `probes/run_linprobe_ijepa.sh` | PJM batch script for `probes/main_linprobe_ijepa.py`. |
| `probes/run_posprobe_siamjepa_cpu.sh` | PJM batch script for `probes/main_position_probe_siamjepa.py`. |
| `probes/run_knn_siamjepa.sh` / `probes/run_knn_siamjepa_cpu.sh` | PJM batch scripts for `probes/main_knn_siamjepa.py` (GPU / CPU-queue variants). |
| `probes/run_shuffle_robustness_cpu.sh` | PJM batch script for `probes/main_shuffle_robustness.py`. |

All `probes/run_*.sh` scripts are submitted from the repo root (e.g. `pjsub -x "..." probes/run_knn_siamjepa.sh`), same as every other `run_*.sh` — not from inside `probes/`.

**Shared**

| Path | Description |
|---|---|
| `utils.py` | Shared utility functions. |
| `util/` | Distributed training helpers (`misc.py`), LR schedule/decay, LARS optimizer, position-embedding interpolation, data augmentation (`crop.py`), dataset helpers. |
| `timm/` | Vendored/modified copy of `pytorch-image-models` (timm). |

## Requirements

- Python 3.8+
- PyTorch with CUDA support
- torchvision
- The vendored `timm` package under `timm/` (no separate install needed; imported directly from the repo root)

## Data

Pretraining and linear probing expect an ImageNet-style folder layout:

```
<data_path>/train/<class>/*.jpeg
<data_path>/val/<class>/*.jpeg
```

## Pretraining

Currently only `siamjepa_vit_base_patch16` is supported (the large/huge variants in `models_siamjepa.py` are not yet implemented).

```bash
torchrun --nproc_per_node=4 --master_port=29561 \
  main_pretrain_siamjepa.py \
  --data_path /path/to/imagenet/ \
  --output_dir ./output_dir_siamjepa \
  --model siamjepa_vit_base_patch16 \
  --batch_size 512 --accum_iter 4 \
  --blr 1.5e-4 --weight_decay 0.1 \
  --mask_ratio 0.75 0.75 0.75 \
  --ema 0.99 0.999 0.9999 \
  --kl_scale 0.01
```

Key arguments:

- `--mask_ratio` — three mask ratios applied across the Siamese views.
- `--ema` — start/mid/end momentum values for the EMA teacher schedule.
- `--kl_scale` — weight of the KL regularization term.
- `--shuffle_teacher` — enable the Random Shuffle Teacher (RST); see below.
- `--init_checkpoint` — warm-start the model weights from an existing checkpoint (fresh optimizer/epoch count); used for the curriculum below.

Pretraining was run on a single node with 4x NVIDIA H100 GPUs. See `run_pretrain.sh` for a full PJM job example.

### Random Shuffle Teacher (RST) and the semantic-to-spatial curriculum

Passing `--shuffle_teacher` randomly permutes the teacher/EMA branch's patch
tokens before the Sim-2 prediction target is computed. Because the
positional embedding still reflects each token's true grid position, this
breaks the correspondence between position and content, which discourages
the encoder from relying on spatial shortcuts and instead encourages more
semantic, position-invariant patch representations (at the cost of spatial
precision — see the paper for the position-decodability and test-time
robustness trade-off this induces).

The curriculum used in the paper is two ordinary pretraining runs, not a
special mode: train with RST first, then continue without it to restore
spatial correspondence.

```bash
# Stage 1: RST phase
torchrun --nproc_per_node=4 --master_port=29561 \
  main_pretrain_siamjepa.py \
  --data_path /path/to/imagenet/ --output_dir ./output_dir_siamjepa \
  --model siamjepa_vit_base_patch16 --batch_size 512 --accum_iter 4 \
  --blr 1.5e-4 --weight_decay 0.1 --mask_ratio 0.75 0.75 0.75 \
  --ema 0.99 0.999 0.9999 --kl_scale 0.01 --epochs 200 \
  --shuffle_teacher

# Stage 2: continue from the RST checkpoint, shuffling turned off
torchrun --nproc_per_node=4 --master_port=29561 \
  main_pretrain_siamjepa.py \
  --data_path /path/to/imagenet/ --output_dir ./output_dir_siamjepa \
  --model siamjepa_vit_base_patch16 --batch_size 512 --accum_iter 4 \
  --blr 1.5e-4 --weight_decay 0.1 --mask_ratio 0.75 0.75 0.75 \
  --ema 0.99 0.999 0.9999 --kl_scale 0.01 --epochs 250 \
  --init_checkpoint ./output_dir_siamjepa/<stage-1-run>/checkpoint-199.pth
```

`run_pretrain.sh` supports the same curriculum via `pjsub -x` — see the `SHUFFLE_TEACHER` example in its header comment.

## Linear probing

```bash
torchrun --nproc_per_node=8 --master_port=29532 \
  main_linprobe_siamjepa.py \
  --data_path /path/to/imagenet/ \
  --batch_size 256 \
  --finetune /path/to/output_dir_siamjepa/checkpoint-100.pth \
  --global_pool
```

This loads the pretrained SiamJEPA encoder into a `vit_base_patch16` backbone, freezes all weights except a BatchNorm + linear head, and trains the head with LARS. Add `--use_ema` to probe the EMA/teacher encoder instead of the student.

See `run_linprobe_siamjepa.sh` for a full PJM job example.

### Other evaluations (`probes/`)

`probes/main_linprobe_ijepa.py`, `probes/main_position_probe_siamjepa.py` and `probes/main_knn_siamjepa.py` all take the same `--finetune <checkpoint> [--use_ema]` as above. `probes/main_shuffle_robustness.py` is different: it takes `--probe_dir <finished output_dir_linprobe/run...>` and re-evaluates that *already-trained* linear-probe head under test-time perturbations, rather than starting from a pretrain checkpoint. See the repository-layout table for what each script measures, and its matching `probes/run_*.sh` script for a full job example (submitted from the repo root, e.g. `pjsub -x "FINETUNE=..." probes/run_knn_siamjepa.sh`).

## Reproducibility note

Results were obtained on a single node with 4x NVIDIA H100 GPUs. Rerunning on a different number/type of GPUs, driver/CUDA version, or library versions can change the effective batch size, numerics, and data loading order, which may lead to different results even with the same hyperparameters and seed.

## License

This repository is licensed on a per-file basis — check each file's own header for the license that applies to it.

- **Apache License 2.0** — files whose header explicitly states Apache License 2.0 (e.g. `utils.py`)
- **CC BY-NC 4.0** (see [`LICENSE`](./LICENSE)) — all other files, whose headers point to "the LICENSE file in the root directory of this source tree" (e.g. `models_siamjepa.py`, `main_pretrain_siamjepa.py`, `main_linprobe_siamjepa.py`, `models_vit.py`, `engine_pretrain.py`, `engine_finetune.py`, and most of `util/`)

## Acknowledgements

This project builds on and modifies code from:

- [MAE](https://github.com/facebookresearch/mae)
- [DeiT](https://github.com/facebookresearch/deit)
- [BEiT](https://github.com/microsoft/unilm/tree/master/beit)
- [MoCo v3](https://github.com/facebookresearch/moco-v3)
- [timm](https://github.com/rwightman/pytorch-image-models)
