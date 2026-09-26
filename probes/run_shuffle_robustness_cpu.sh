#!/bin/sh
#PJM -L rscgrp=a-batch
#PJM -L node=1
#PJM -L elapse=2:00:00
#PJM -j
#PJM -o ./logs/%n.%j.out

export PYTHONUNBUFFERED=1

# ------------------------------------------------------------------------------
# Test-time robustness (patch shuffle / no position embedding / token drop)
# of a finished linear probe (CPU, a-batch). No
# training: reuses the probe's saved encoder + head (see
# main_shuffle_robustness.py). PROBE_DIR is an output_dir_linprobe/<run> dir.
#
#   pjsub -x "PROBE_DIR=./output_dir_linprobe/run6761375_..." probes/run_shuffle_robustness_cpu.sh
#
#   # only the clean baseline + nopos + token drop (skip the shuffle grids)
#   pjsub -x "PROBE_DIR=...,GRIDS=1" probes/run_shuffle_robustness_cpu.sh
#
#   # occlusion only (clean baseline + blockdrop/occ/occc at 25/50/75%)
#   pjsub -x "PROBE_DIR=...,OCCLUSION_ONLY=1" probes/run_shuffle_robustness_cpu.sh
#
#   # mechanism tests: blockdrop / occ / occvis at 25-75% and in-context pooling
#   pjsub -x "PROBE_DIR=...,MECHANISM_ONLY=1" probes/run_shuffle_robustness_cpu.sh
#
# KEEP_RATIOS / BLOCK_AREAS contain commas, which pjsub -x cannot pass -- edit
# the defaults. Submit from the repo root (not from inside probes/) so
# relative paths like OUTPUT_DIR resolve the same way as every other
# run_*.sh script.
# ------------------------------------------------------------------------------
: "${PROBE_DIR:?Set PROBE_DIR to a finished output_dir_linprobe/<run> directory}"
: "${GRIDS:=1,2,4,7,14}"
: "${KEEP_RATIOS:=0.75,0.5,0.25,0.1}"
: "${BLOCK_AREAS:=0.25,0.5,0.75}"
: "${OCCLUSION_ONLY:=}"
: "${MECHANISM_ONLY:=}"
: "${CTX_POOL_KS:=1,4,16,49}"
: "${NUM_VAL_IMAGES:=10000}"
: "${BATCH_SIZE:=64}"
: "${NUM_WORKERS:=32}"
: "${OUTPUT_DIR:=./output_dir_shuffle}"

EXTRA=""
if [ -n "$OCCLUSION_ONLY" ]; then
  GRIDS=1
  KEEP_RATIOS=""
  EXTRA="--skip_no_pos --block_areas $BLOCK_AREAS"
fi
if [ -n "$MECHANISM_ONLY" ]; then
  GRIDS=1
  KEEP_RATIOS=""
  EXTRA="--skip_no_pos --block_areas $BLOCK_AREAS --skip_center --occ_vispool --ctx_pool_ks $CTX_POOL_KS"
fi

echo "=== Run config (CPU) ==="
echo "PROBE_DIR=$PROBE_DIR"
echo "GRIDS=$GRIDS  KEEP_RATIOS=$KEEP_RATIOS  NUM_VAL_IMAGES=$NUM_VAL_IMAGES  EXTRA=$EXTRA"
echo "nproc: $(nproc)"
echo "========================="

python3 probes/main_shuffle_robustness.py \
  --probe_dir "$PROBE_DIR" \
  --grids "$GRIDS" \
  --keep_ratios "$KEEP_RATIOS" \
  --num_val_images $NUM_VAL_IMAGES \
  --batch_size $BATCH_SIZE \
  --num_workers $NUM_WORKERS \
  --output_dir $OUTPUT_DIR \
  $EXTRA
