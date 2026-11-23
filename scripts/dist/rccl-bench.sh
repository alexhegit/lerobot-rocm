#!/bin/bash
# RCCL all-reduce bandwidth on 2, 4 and 8 GPUs (torch.distributed, nccl backend = RCCL on ROCm).
set -euo pipefail
for n in 2 4 8; do
  torchrun --nproc-per-node=$n /outputs/rocm-train/jobs/rccl_allreduce_bench.py 2>&1 | grep ALLREDUCE
done
