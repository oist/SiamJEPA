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
#   # Random Shuffle Teacher (RST) curriculum -- stage 1 (RST on), then stage 2
#   # warm-started from stage 1's checkpoint with RST off:
#   pjsub -x "EPOCHS=200,SHUFFLE_TEACHER=1" run_pretrain.sh
#
#   # faster numerics: PRECISION=tf32 or bf16 (default fp32 = original runs);
#   # BF16_PREDICTOR=1 also runs the predictor in bf16 (only with bf16)
#   pjsub -x "PRECISION=bf16" run_pretrain.sh
#
#   # ViT-L (does not fit 512/GPU: keep the effective batch with 256 x 8, or
#   # GRAD_CKPT=1 to recompute encoder activations)
#   pjsub -x "MODEL=siamjepa_vit_large_patch16,PRECISION=bf16,BATCH_SIZE=256,ACCUM_ITER=8" run_pretrain.sh
#
#   # view2 predictor placement: fixed by default (run dirs get "_v2fix");
#   # LEGACY_VIEW2_RESTORE=1 reproduces runs made before the fix
#   pjsub -x "LEGACY_VIEW2_RESTORE=1" run_pretrain.sh
#   pjsub -x "EPOCHS=250,INIT_CHECKPOINT=./output_dir_siamjepa/<stage-1-run>/checkpoint-199.pth" run_pretrain.sh
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
: "${SHUFFLE_TEACHER:=}"
: "${PRECISION:=fp32}"
: "${MODEL:=siamjepa_vit_base_patch16}"
: "${GRAD_CKPT:=}"
: "${NUM_WORKERS:=10}"
: "${CLIP_GRAD:=3.0}"
: "${WARMUP_EPOCHS:=40}"
: "${BF16_PREDICTOR:=}"
: "${LEGACY_VIEW2_RESTORE:=}"

SHUFFLE_FLAG=""
if [ -n "$SHUFFLE_TEACHER" ]; then
  SHUFFLE_FLAG="--shuffle_teacher"
fi
BF16_PREDICTOR_FLAG=""
GRAD_CKPT_FLAG=""
if [ -n "$GRAD_CKPT" ]; then
  GRAD_CKPT_FLAG="--grad_checkpointing"
fi
LEGACY_VIEW2_FLAG=""
if [ -n "$LEGACY_VIEW2_RESTORE" ]; then
  LEGACY_VIEW2_FLAG="--legacy_view2_restore"
fi
if [ -n "$BF16_PREDICTOR" ]; then
  BF16_PREDICTOR_FLAG="--bf16_predictor"
fi

echo "=== Run config ==="
echo "MODEL=$MODEL  GRAD_CKPT=$GRAD_CKPT  NUM_WORKERS=$NUM_WORKERS  CLIP_GRAD=$CLIP_GRAD  WARMUP_EPOCHS=$WARMUP_EPOCHS"
echo "KL_SCALE=$KL_SCALE  WEIGHT_DECAY=$WEIGHT_DECAY  BLR=$BLR"
echo "BATCH_SIZE=$BATCH_SIZE  ACCUM_ITER=$ACCUM_ITER  MASK_RATIO=$MASK_RATIO  EMA=$EMA"
echo "OUTPUT_DIR=$OUTPUT_DIR  DATA_PATH=$DATA_PATH  MASTER_PORT=$MASTER_PORT"
echo "INIT_CHECKPOINT=$INIT_CHECKPOINT  EPOCHS=$EPOCHS  RESUME=$RESUME"
echo "SHUFFLE_TEACHER=$SHUFFLE_TEACHER  PRECISION=$PRECISION  BF16_PREDICTOR=$BF16_PREDICTOR  LEGACY_VIEW2_RESTORE=$LEGACY_VIEW2_RESTORE"
echo "==================="

torchrun \
  --nproc_per_node=4 \
  --master_port=$MASTER_PORT \
  main_pretrain_siamjepa.py \
  --model $MODEL \
  --num_workers $NUM_WORKERS \
  --clip_grad $CLIP_GRAD \
  --warmup_epochs $WARMUP_EPOCHS \
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
  --init_checkpoint "$INIT_CHECKPOINT" \
  --precision $PRECISION \
  $SHUFFLE_FLAG $BF16_PREDICTOR_FLAG $LEGACY_VIEW2_FLAG $GRAD_CKPT_FLAG
