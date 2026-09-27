#!/bin/sh
#PJM -L rscgrp=b-batch
#PJM -L node=1
#PJM -L elapse=5:59:59
#PJM -j
#PJM -o ./logs/%n.%j.out

export OMP_NUM_THREADS=4
export NCCL_ASYNC_ERROR_HANDLING=1
export PYTHONUNBUFFERED=1
export TORCHELASTIC_ERROR_FILE=$HOME/elastic_err_${PJM_JOBID:-$$}.json

# ------------------------------------------------------------------------------
# Linear probe following the I-JEPA protocol (see main_linprobe_ijepa.py):
# EMA encoder, avg-pooled last layer / last-4-layer concat, LARS bs16384,
# 50 epochs with /10 step every 15, lr {0.01,0.05,0.001} x wd {5e-4,0} -- all
# 12 heads trained in this one job.
#
#   pjsub -x "FINETUNE=./output_dir_siamjepa/.../checkpoint-250.pth" probes/run_linprobe_ijepa.sh
#
#   # resume a timed-out run (heads + optimizer are saved every epoch)
#   pjsub -x "FINETUNE=...,RESUME_DIR=./output_dir_linprobe_ijepa/run<jobid>_..." probes/run_linprobe_ijepa.sh
#
# Submit from the repo root (not from inside probes/) so relative paths like
# OUTPUT_DIR resolve the same way as every other run_*.sh script.
# ------------------------------------------------------------------------------
: "${FINETUNE:?Set FINETUNE to a checkpoint path}"
: "${USE_EMA:=1}"
: "${BATCH_SIZE:=512}"
: "${ACCUM_ITER:=8}"
: "${MASTER_PORT:=29541}"
: "${NPROC_PER_NODE:=4}"
: "${OUTPUT_DIR:=./output_dir_linprobe_ijepa}"
: "${RESUME_DIR:=}"

EXTRA=""
[ -n "$USE_EMA" ] && EXTRA="$EXTRA --use_ema"
[ -n "$RESUME_DIR" ] && EXTRA="$EXTRA --resume_dir $RESUME_DIR"

echo "=== Run config ==="
echo "FINETUNE=$FINETUNE  USE_EMA=$USE_EMA"
echo "BATCH_SIZE=$BATCH_SIZE x ACCUM_ITER=$ACCUM_ITER x NPROC=$NPROC_PER_NODE"
echo "OUTPUT_DIR=$OUTPUT_DIR  RESUME_DIR=$RESUME_DIR"
echo "==================="

torchrun \
  --nproc_per_node=$NPROC_PER_NODE \
  --master_port=$MASTER_PORT \
  probes/main_linprobe_ijepa.py \
  --output_dir $OUTPUT_DIR \
  --batch_size $BATCH_SIZE \
  --accum_iter $ACCUM_ITER \
  --finetune "$FINETUNE" \
  $EXTRA
