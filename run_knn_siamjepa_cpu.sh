#!/bin/sh
#PJM -L rscgrp=a-batch
#PJM -L node=1
#PJM -L elapse=24:00:00
#PJM -j
#PJM -o ./logs/%n.%j.out

export PYTHONUNBUFFERED=1

# ------------------------------------------------------------------------------
# CPU-only variant of run_knn_siamjepa.sh, for when the b-batch GPU queue
# (35 nodes cluster-wide) is congested. a-batch has 1020 CPU-only nodes and
# tends to start almost immediately even when b-batch has a multi-day queue.
#
# NOTE: a synthetic (no-data-loading) benchmark suggested ~2.8h for the full
# train+val feature extraction on a 120-core a-batch node, but a real run
# with actual ImageNet JPEG decode/resize did not even finish extracting
# train features within a 6h elapse limit (first attempt: jobs 6755648/56/57
# were killed by walltime). JPEG decode + filesystem I/O for 1.28M small
# files is apparently the real bottleneck, not the matmul. Elapse bumped to
# 24h as a generous safety margin; NUM_WORKERS raised in case I/O parallelism
# was the limiting factor. Single process, no torchrun
# (misc.init_distributed_mode falls back to non-distributed automatically
# when RANK/WORLD_SIZE aren't set).
#
#   pjsub -x "FINETUNE=./output_dir_siamjepa/.../checkpoint-200.pth" run_knn_siamjepa_cpu.sh
# ------------------------------------------------------------------------------
: "${FINETUNE:?Set FINETUNE to a checkpoint path, e.g. pjsub -x \"FINETUNE=./output_dir_siamjepa/.../checkpoint-200.pth\" run_knn_siamjepa_cpu.sh}"
: "${BATCH_SIZE:=64}"
: "${NUM_WORKERS:=32}"
: "${OUTPUT_DIR:=./output_dir_knn}"
: "${USE_EMA:=}"
: "${NB_KNN:=10 20 100 200}"
: "${TEMPERATURE:=0.07}"

if [ -n "$USE_EMA" ]; then
  USE_EMA_FLAG="--use_ema"
else
  USE_EMA_FLAG=""
fi

echo "=== Run config (CPU) ==="
echo "FINETUNE=$FINETUNE"
echo "BATCH_SIZE=$BATCH_SIZE  NUM_WORKERS=$NUM_WORKERS  NB_KNN=$NB_KNN  TEMPERATURE=$TEMPERATURE"
echo "OUTPUT_DIR=$OUTPUT_DIR  USE_EMA=$USE_EMA"
echo "nproc: $(nproc)"
echo "========================="

python3 main_knn_siamjepa.py \
  --device cpu \
  --output_dir $OUTPUT_DIR \
  --batch_size $BATCH_SIZE \
  --num_workers $NUM_WORKERS \
  --no_pin_mem \
  --finetune "$FINETUNE" \
  --global_pool \
  --nb_knn $NB_KNN \
  --temperature $TEMPERATURE \
  $USE_EMA_FLAG
