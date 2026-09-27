#!/bin/sh
#PJM -L rscgrp=a-batch
#PJM -L node=1
#PJM -L elapse=0:30:00
#PJM -j
#PJM -o ./logs/%n.%j.out
# Run the CPU regression tests as a job (not on the login node):
#   pjsub tests/run_tests_cpu.sh
export PYTHONUNBUFFERED=1
status=0
for t in tests/test_*.py; do
  echo "== $t"
  python3 "$t" || status=1
done
echo "TESTS_EXIT=$status"
exit $status
