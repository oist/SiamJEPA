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

| Path | Description |
|---|---|
| `models_siamjepa.py` | SiamJEPA model definitions (Siamese ViT encoders, cross-attention decoder/predictor, projection MLP, EMA teacher update, KL regularization). |
| `models_vit.py` | Plain ViT backbones (`vit_small_patch16`, `vit_base_patch16`, `vit_large_patch16`, `vit_huge_patch14`, ...) used for downstream evaluation. |
| `main_pretrain_siamjepa.py` | Entry point for self-supervised pretraining. |
| `main_linprobe_siamjepa.py` | Entry point for linear probing on top of a frozen pretrained encoder. |
| `engine_pretrain.py` | Training loop for pretraining (`train_one_epoch_siamjepa`). |
| `engine_finetune.py` | Training/evaluation loops for linear probing / finetuning (`train_one_epoch`, `evaluate`). |
| `utils.py` | Shared utility functions. |
| `util/` | Distributed training helpers (`misc.py`), LR schedule/decay, LARS optimizer, position-embedding interpolation, data augmentation (`crop.py`), dataset helpers. |
| `timm/` | Vendored/modified copy of `pytorch-image-models` (timm). |
| `run_pretrain.sh` | Example Slurm/PJM batch script for multi-GPU pretraining via `torchrun`. |
| `run_linprobe_siamjepa.sh` | Example Slurm/PJM batch script for linear probing. |

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

Pretraining was run on a single node with 4x NVIDIA H100 GPUs. See `run_pretrain.sh` for a full Slurm/PJM job example.

## Linear probing

```bash
torchrun --nproc_per_node=8 --master_port=29532 \
  main_linprobe_siamjepa.py \
  --data_path /path/to/imagenet/ \
  --batch_size 256 \
  --finetune /path/to/output_dir_siamjepa/checkpoint-100.pth \
  --global_pool
```

This loads the pretrained SiamJEPA encoder into a `vit_base_patch16` backbone, freezes all weights except a BatchNorm + linear head, and trains the head with LARS.

See `run_linprobe_siamjepa.sh` for a full Slurm/PJM job example.

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
