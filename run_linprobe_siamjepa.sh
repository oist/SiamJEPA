#!/bin/sh
#PJM -L rscgrp=c-batch
#PJM -L node=1
#PJM -L elapse=5:59:59
#PJM -j

export OMP_NUM_THREADS=4
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=INFO

export PYTHONUNBUFFERED=1
export TORCHELASTIC_ERROR_FILE=$HOME/elastic_err_${SLURM_JOB_ID}_${SLURM_PROCID}.json


torchrun --nproc_per_node=8 --master_port=29532 main_linprobe_siamjepa.py --batch_size 256 --finetune /home/pj26000049/ku60000347/Python/2026/SiamJEPA/output_dir_siamjepa/checkpoint-100.pth --global_pool 

