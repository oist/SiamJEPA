#!/bin/sh
# Scan output_dir_siamjepa/run*/ for milestone checkpoints and submit a
# linprobe job (run_linprobe_siamjepa.sh) for each one not already
# submitted. Idempotent: each submission is recorded as a marker file
# next to the checkpoint, so re-running this script (e.g. periodically)
# only ever submits a checkpoint once.
set -e
cd "$(dirname "$0")"

MILESTONES="50 100 150 200 250 300 350 399"

for run_dir in output_dir_siamjepa/run*/; do
  [ -d "$run_dir" ] || continue
  run_dir=${run_dir%/}
  mkdir -p "$run_dir/.linprobe_submitted"
  for epoch in $MILESTONES; do
    ckpt="$run_dir/checkpoint-$epoch.pth"
    marker="$run_dir/.linprobe_submitted/checkpoint-$epoch"
    if [ -f "$ckpt" ] && [ ! -f "$marker" ]; then
      echo "Submitting linprobe for $ckpt"
      pjsub -x "FINETUNE=./$ckpt" run_linprobe_siamjepa.sh
      touch "$marker"
    fi
  done
done
