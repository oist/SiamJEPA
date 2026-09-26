#!/bin/sh
#PJM -L rscgrp=b-batch
#PJM -L node=1
#PJM -L elapse=1:00:00
#PJM -j
#PJM -o ./logs/%n.%j.out
# Speed/memory benchmark + equivalence check for the pretraining step.
#   pjsub bench/run_bench.sh
# Each CONFIGS entry is repo:precision[:per-GPU batch[:ckpt]], e.g.
#   pjsub -x "MODEL=siamjepa_vit_large_patch16,CONFIGS=new:bf16:256 new:bf16:512:ckpt" bench/run_bench.sh
# (repo old = OLD_REPO; precision fp32|tf32|bf16|bf16-pred; batch defaults to
# BATCH_SIZE; ckpt = gradient checkpointing). The accumulation steps are
# chosen so the effective batch over 4 GPUs stays at EFF_BATCH.
export OMP_NUM_THREADS=4
export PYTHONUNBUFFERED=1
: "${MODEL:=siamjepa_vit_base_patch16}"
: "${BATCH_SIZE:=512}"
: "${EFF_BATCH:=8192}"
: "${OLD_REPO:=../SiamJEPA-dev}"
: "${CHECKPOINT:=}"
: "${CONFIGS:=old:fp32 new:fp32 new:tf32 new:bf16 new:bf16-pred}"
: "${MASTER_PORT:=29581}"

nvidia-smi --query-gpu=name,memory.total --format=csv

if [ -n "$CHECKPOINT" ]; then
  python3 bench/check_speedup_equivalence.py --old_repo "$OLD_REPO" --checkpoint "$CHECKPOINT"
fi

for cfg in $CONFIGS; do
  IFS=: read which prec bs ckpt <<CFG
$cfg
CFG
  repo=.; unused=""; extra=""
  [ "$which" = old ] && repo=$OLD_REPO && unused=--find_unused_parameters
  [ "$prec" = bf16-pred ] && prec=bf16 && extra=--bf16_predictor
  [ -z "$bs" ] && bs=$BATCH_SIZE
  [ "$ckpt" = ckpt ] && extra="$extra --grad_checkpointing"
  accum=$(( EFF_BATCH / (bs * 4) )); [ $accum -lt 1 ] && accum=1
  torchrun --nproc_per_node=4 --master_port=$MASTER_PORT bench/bench_pretrain_speed.py \
    --repo "$repo" --model $MODEL --batch_size $bs --accum_iter $accum \
    --precision $prec $extra $unused --tag "$cfg" 2>&1 | grep -E "RESULT|Error|error|OutOfMemory" | head -5
done
