#!/bin/sh
#PJM -L rscgrp=b-batch
#PJM -L node=1
#PJM -L elapse=1:00:00
#PJM -j
#PJM -o ./logs/%n.%j.out

export OMP_NUM_THREADS=4
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=INFO

export PYTHONUNBUFFERED=1
export TORCHELASTIC_ERROR_FILE=$HOME/elastic_err_${PJM_JOBID:-$$}.json

# ------------------------------------------------------------------------------
# Weighted k-NN evaluation (DINO-style) of a SiamJEPA pretrain checkpoint's
# frozen features -- no training. A single forward pass over train+val builds
# a feature bank, then classifies each val image by a similarity-weighted vote
# among its k nearest train neighbors. Much cheaper than linear probing (no
# backward pass, no 90-epoch loop), so this should finish in well under an
# hour even on a congested queue.
#
#   pjsub -x "FINETUNE=./output_dir_siamjepa/.../checkpoint-200.pth" run_knn_siamjepa.sh
#
#   # probe the EMA/teacher encoder instead of the student
#   pjsub -x "FINETUNE=...,USE_EMA=1" run_knn_siamjepa.sh
#
# Every run lands in its own self-describing subdirectory under OUTPUT_DIR
# (named after the checkpoint being evaluated; see util/experiment_tracking.py).
# ------------------------------------------------------------------------------
: "${FINETUNE:?Set FINETUNE to a checkpoint path, e.g. pjsub -x \"FINETUNE=./output_dir_siamjepa/.../checkpoint-200.pth\" run_knn_siamjepa.sh}"
: "${BATCH_SIZE:=512}"
: "${MASTER_PORT:=29572}"
: "${NPROC_PER_NODE:=4}"
: "${OUTPUT_DIR:=./output_dir_knn}"
: "${USE_EMA:=}"
: "${NB_KNN:=10 20 100 200}"
: "${TEMPERATURE:=0.07}"

if [ -n "$USE_EMA" ]; then
  USE_EMA_FLAG="--use_ema"
else
  USE_EMA_FLAG=""
fi

echo "=== Run config ==="
echo "FINETUNE=$FINETUNE"
echo "BATCH_SIZE=$BATCH_SIZE  NB_KNN=$NB_KNN  TEMPERATURE=$TEMPERATURE"
echo "OUTPUT_DIR=$OUTPUT_DIR  MASTER_PORT=$MASTER_PORT  USE_EMA=$USE_EMA"
echo "==================="

torchrun \
  --nproc_per_node=$NPROC_PER_NODE \
  --master_port=$MASTER_PORT \
  main_knn_siamjepa.py \
  --output_dir $OUTPUT_DIR \
  --batch_size $BATCH_SIZE \
  --finetune "$FINETUNE" \
  --global_pool \
  --nb_knn $NB_KNN \
  --temperature $TEMPERATURE \
  $USE_EMA_FLAG
