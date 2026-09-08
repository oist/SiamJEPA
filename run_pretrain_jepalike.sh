#!/bin/sh
# NOTE: kept as a historical record of job 6723138 (KL=0.00001, weight_decay=0.05).
# For new runs, use the parameterized run_pretrain.sh instead, e.g.:
#   pjsub -x "KL_SCALE=0.00001,WEIGHT_DECAY=0.05" run_pretrain.sh
#PJM -L rscgrp=b-batch
#PJM -L node=1
#PJM -L elapse=168:00:00
#PJM -j
#PJM -o ./logs/%n.%j.out

export OMP_NUM_THREADS=4
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=INFO

export PYTHONUNBUFFERED=1
export TORCHELASTIC_ERROR_FILE=$HOME/elastic_err_${SLURM_JOB_ID}_${SLURM_PROCID}.json

torchrun \
  --nproc_per_node=4 \
  --master_port=29562 \
  main_pretrain_siamjepa.py --data_path /home/pj26000049/ku60000347/Python/Dataset/ImageNet/ --output_dir ./output_dir_siamjepa --kl_scale=0.00001 --accum_iter 4 --blr 1.5e-4 --batch_size 512 --mask_ratio 0.75 0.75 0.75 --ema 0.99 0.999 0.9999 --weight_decay 0.05
