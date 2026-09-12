#!/bin/sh
#PJM -L rscgrp=a-batch
#PJM -L node=1
#PJM -L elapse=2:00:00
#PJM -j
#PJM -o ./logs/%n.%j.out

export PYTHONUNBUFFERED=1

# ------------------------------------------------------------------------------
# Position-decodability probe (CPU, a-batch -- see run_knn_siamjepa_cpu.sh for
# why a-batch is usually the faster route to a result despite being slower
# per-image than a GPU). Only needs a couple thousand images (not the full
# ImageNet train set like kNN), so this should be much quicker than the kNN
# CPU jobs even accounting for a-batch's queue behavior.
#
#   pjsub -x "FINETUNE=./output_dir_siamjepa/.../checkpoint-200.pth" run_posprobe_siamjepa_cpu.sh
#
#   # probe the EMA/teacher encoder instead of the student
#   pjsub -x "FINETUNE=...,USE_EMA=1" run_posprobe_siamjepa_cpu.sh
# ------------------------------------------------------------------------------
: "${FINETUNE:?Set FINETUNE to a checkpoint path, e.g. pjsub -x \"FINETUNE=./output_dir_siamjepa/.../checkpoint-200.pth\" run_posprobe_siamjepa_cpu.sh}"
: "${BATCH_SIZE:=64}"
: "${NUM_WORKERS:=32}"
: "${NUM_TRAIN_IMAGES:=2000}"
: "${NUM_VAL_IMAGES:=500}"
: "${PROBE_EPOCHS:=30}"
: "${OUTPUT_DIR:=./output_dir_posprobe}"
: "${USE_EMA:=}"

if [ -n "$USE_EMA" ]; then
  USE_EMA_FLAG="--use_ema"
else
  USE_EMA_FLAG=""
fi

echo "=== Run config (CPU) ==="
echo "FINETUNE=$FINETUNE"
echo "NUM_TRAIN_IMAGES=$NUM_TRAIN_IMAGES  NUM_VAL_IMAGES=$NUM_VAL_IMAGES  PROBE_EPOCHS=$PROBE_EPOCHS"
echo "OUTPUT_DIR=$OUTPUT_DIR  USE_EMA=$USE_EMA"
echo "nproc: $(nproc)"
echo "========================="

python3 main_position_probe_siamjepa.py \
  --device cpu \
  --output_dir $OUTPUT_DIR \
  --batch_size $BATCH_SIZE \
  --num_workers $NUM_WORKERS \
  --no_pin_mem \
  --finetune "$FINETUNE" \
  --global_pool \
  --num_train_images $NUM_TRAIN_IMAGES \
  --num_val_images $NUM_VAL_IMAGES \
  --probe_epochs $PROBE_EPOCHS \
  $USE_EMA_FLAG
