#!/bin/sh
#PJM -L rscgrp=b-batch
#PJM -L node=1
#PJM -L elapse=1:00:00
#PJM -j
#PJM -o ./logs/%n.%j.out
# Speed/memory benchmark + equivalence check for the pretraining step.
#   pjsub bench/run_bench.sh
#   pjsub -x "MODEL=siamjepa_vit_base_patch16,BATCH_SIZE=512" bench/run_bench.sh
export OMP_NUM_THREADS=4
export PYTHONUNBUFFERED=1
: "${MODEL:=siamjepa_vit_base_patch16}"
: "${BATCH_SIZE:=512}"
: "${ACCUM_ITER:=4}"
: "${OLD_REPO:=../SiamJEPA-dev}"
: "${CHECKPOINT:=}"
: "${CONFIGS:=old:fp32 new:fp32 new:tf32 new:bf16 new:bf16-pred}"
: "${MASTER_PORT:=29581}"

nvidia-smi --query-gpu=name,memory.total --format=csv

if [ -n "$CHECKPOINT" ]; then
  python3 bench/check_speedup_equivalence.py --old_repo "$OLD_REPO" --checkpoint "$CHECKPOINT"
fi

for cfg in $CONFIGS; do
  repo=.; unused=""
  [ "${cfg%%:*}" = old ] && repo=$OLD_REPO && unused=--find_unused_parameters
  prec=${cfg#*:}; extra=""
  [ "$prec" = bf16-pred ] && prec=bf16 && extra=--bf16_predictor
  torchrun --nproc_per_node=4 --master_port=$MASTER_PORT bench/bench_pretrain_speed.py \
    --repo "$repo" --model $MODEL --batch_size $BATCH_SIZE --accum_iter $ACCUM_ITER \
    --precision $prec $extra $unused --tag "$cfg" 2>&1 | grep -E "RESULT|Error|error|OutOfMemory" | head -5
done
