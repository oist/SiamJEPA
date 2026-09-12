#!/bin/sh
#PJM -L rscgrp=b-batch
#PJM -L node=1
#PJM -L elapse=168:00:00
#PJM -j
#PJM -o ./logs/%n.%j.out

export OMP_NUM_THREADS=4
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=INFO

export PYTHONUNBUFFERED=1
export TORCHELASTIC_ERROR_FILE=$HOME/elastic_err_${PJM_JOBID:-$$}.json

# ------------------------------------------------------------------------------
# Hyperparameters. Override any of these at submission time with `pjsub -x`
# instead of editing this file, e.g.:
#
#   # paper Table 3 cell: KL=0.01, weight_decay=0.1 (the current main run)
#   pjsub -x "KL_SCALE=0.01,WEIGHT_DECAY=0.1" run_pretrain.sh
#
#   # JEPA-like baseline: KL=0.00001, weight_decay=0.05
#   pjsub -x "KL_SCALE=0.00001,WEIGHT_DECAY=0.05" run_pretrain.sh
#
#   # a learning-rate sweep point, keeping everything else default
#   pjsub -x "BLR=2.0e-4" run_pretrain.sh
#
#   # warm-start from a checkpoint with the same architecture (e.g. a PhiNetv2
#   # checkpoint -- weights only, fresh optimizer/epoch; see --init_checkpoint)
#   pjsub -x "INIT_CHECKPOINT=/path/to/checkpoint-200.pth" run_pretrain.sh
#
#   # a longer single continuous schedule (e.g. 600 epochs instead of 400) --
#   # note b-batch caps a single job's elapse at 168h (604800s), so a 600-epoch
#   # run at our observed ~3.3 epoch/hr pace (~179h total) will likely need one
#   # resume-chained follow-up job; use RESUME to continue from a checkpoint
#   # with the same EPOCHS target and args.start_epoch inferred from it:
#   pjsub -x "EPOCHS=600" run_pretrain.sh
#   pjsub -x "EPOCHS=600,RESUME=./output_dir_siamjepa/run.../checkpoint-NNN.pth" run_pretrain.sh
#
# Every run still lands in its own self-describing subdirectory under
# OUTPUT_DIR (see util/experiment_tracking.py), so runs never overwrite
# each other's checkpoints/logs regardless of which variables were set.
# ------------------------------------------------------------------------------
: "${KL_SCALE:=0.01}"
: "${WEIGHT_DECAY:=0.1}"
: "${BLR:=1.5e-4}"
: "${BATCH_SIZE:=512}"
: "${ACCUM_ITER:=4}"
: "${MASK_RATIO:=0.75 0.75 0.75}"
: "${EMA:=0.99 0.999 0.9999}"
: "${MASTER_PORT:=29561}"
: "${OUTPUT_DIR:=./output_dir_siamjepa}"
: "${DATA_PATH:=/home/pj26000049/ku60000347/Python/Dataset/ImageNet/}"
: "${INIT_CHECKPOINT:=}"
: "${EPOCHS:=400}"
: "${RESUME:=}"

echo "=== Run config ==="
echo "KL_SCALE=$KL_SCALE  WEIGHT_DECAY=$WEIGHT_DECAY  BLR=$BLR"
echo "BATCH_SIZE=$BATCH_SIZE  ACCUM_ITER=$ACCUM_ITER  MASK_RATIO=$MASK_RATIO  EMA=$EMA"
echo "OUTPUT_DIR=$OUTPUT_DIR  DATA_PATH=$DATA_PATH  MASTER_PORT=$MASTER_PORT"
echo "INIT_CHECKPOINT=$INIT_CHECKPOINT  EPOCHS=$EPOCHS  RESUME=$RESUME"
echo "==================="

torchrun \
  --nproc_per_node=4 \
  --master_port=$MASTER_PORT \
  main_pretrain_siamjepa.py \
  --data_path $DATA_PATH \
  --output_dir $OUTPUT_DIR \
  --kl_scale=$KL_SCALE \
  --accum_iter $ACCUM_ITER \
  --blr $BLR \
  --batch_size $BATCH_SIZE \
  --mask_ratio $MASK_RATIO \
  --ema $EMA \
  --weight_decay $WEIGHT_DECAY \
  --epochs $EPOCHS \
  --resume "$RESUME" \
  --init_checkpoint "$INIT_CHECKPOINT"
