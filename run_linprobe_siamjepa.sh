#!/bin/sh
#PJM -L rscgrp=c-batch
#PJM -L node=1
#PJM -L elapse=5:59:59
#PJM -j

export OMP_NUM_THREADS=4
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=INFO

export PYTHONUNBUFFERED=1
export TORCHELASTIC_ERROR_FILE=$HOME/elastic_err_${PJM_JOBID:-$$}.json

# ------------------------------------------------------------------------------
# Hyperparameters. Override any of these at submission time with `pjsub -x`
# instead of editing this file. FINETUNE is required -- point it at a
# checkpoint produced under output_dir_siamjepa/<run.../checkpoint-N.pth>.
#
#   pjsub -x "FINETUNE=./output_dir_siamjepa/run6723042_.../checkpoint-100.pth" \
#     run_linprobe_siamjepa.sh
#
# Every run lands in its own self-describing subdirectory under OUTPUT_DIR
# (named after the checkpoint being evaluated; see util/experiment_tracking.py),
# so evaluating many checkpoints never overwrites earlier results.
# ------------------------------------------------------------------------------
: "${FINETUNE:?Set FINETUNE to a checkpoint path, e.g. pjsub -x \"FINETUNE=./output_dir_siamjepa/.../checkpoint-100.pth\" run_linprobe_siamjepa.sh}"
: "${WEIGHT_DECAY:=0}"
: "${BLR:=0.1}"
: "${BATCH_SIZE:=256}"
: "${MASTER_PORT:=29532}"
: "${OUTPUT_DIR:=./output_dir_linprobe}"

echo "=== Run config ==="
echo "FINETUNE=$FINETUNE"
echo "WEIGHT_DECAY=$WEIGHT_DECAY  BLR=$BLR  BATCH_SIZE=$BATCH_SIZE"
echo "OUTPUT_DIR=$OUTPUT_DIR  MASTER_PORT=$MASTER_PORT"
echo "==================="

torchrun \
  --nproc_per_node=8 \
  --master_port=$MASTER_PORT \
  main_linprobe_siamjepa.py \
  --output_dir $OUTPUT_DIR \
  --batch_size $BATCH_SIZE \
  --weight_decay $WEIGHT_DECAY \
  --blr $BLR \
  --finetune "$FINETUNE" \
  --global_pool
