#!/bin/sh
#PJM -L rscgrp=a-batch
#PJM -L node=1
#PJM -L elapse=2:00:00
#PJM -j
#PJM -o ./logs/%n.%j.out

export PYTHONUNBUFFERED=1

# ------------------------------------------------------------------------------
# PCA visualization of frozen patch features (probes/visualize_pca_features.py)
# on CPU (a-batch). Checkpoints come from a spec file, one per line:
#   JEPA-like\n(λ=1e-5)=../SiamJEPA-dev/output_dir_siamjepa/run.../checkpoint-399.pth:ema
#
#   pjsub -x "CKPT_FILE=probes/pca_ckpts.txt,OUT=./output_dir_pca/fig.pdf,ARGS=--pca_mode per_image --fg_mask --img_size 448" probes/run_pca_vis_cpu.sh
#
# SCRIPT=probes/visualize_cls_attention.py runs the [CLS]-attention maps
# instead (same checkpoint file / image arguments).
#
# Submit from the repo root.
# ------------------------------------------------------------------------------
: "${CKPT_FILE:=}"   # one Label=/path/checkpoint.pth[:ema] per line (omit for scripts that take --run)
: "${OUT:=./output_dir_pca/pca_features.pdf}"
: "${ARGS:=}"
: "${SCRIPT:=probes/visualize_pca_features.py}"

mkdir -p "$(dirname "$OUT")"
echo "=== PCA visualization (CPU) ==="
echo "SCRIPT=$SCRIPT  CKPT_FILE=$CKPT_FILE  OUT=$OUT"
echo "ARGS=$ARGS"
echo "nproc: $(nproc)"
echo "==============================="

CKPT_FLAG=""
[ -n "$CKPT_FILE" ] && CKPT_FLAG="--ckpt_file $CKPT_FILE"
python3 "$SCRIPT" $CKPT_FLAG --out "$OUT" $ARGS
