"""All-reduce bus bandwidth with torch.distributed (RCCL) across the visible GPUs.

Launch: torchrun --nproc-per-node=N rccl_allreduce_bench.py
Bus bandwidth follows nccl-tests: algbw * 2 * (n - 1) / n.
"""

import os
import time

import torch
import torch.distributed as dist

dist.init_process_group("nccl")
rank, world = dist.get_rank(), dist.get_world_size()
torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
for size_mb in (1, 8, 64, 256):
    x = torch.ones(size_mb * 2**20 // 4, device="cuda")
    for _ in range(5):
        dist.all_reduce(x)
    torch.cuda.synchronize()
    iters = 20
    start = time.perf_counter()
    for _ in range(iters):
        dist.all_reduce(x)
    torch.cuda.synchronize()
    sec = (time.perf_counter() - start) / iters
    algbw = size_mb / 1024 / sec
    if rank == 0:
        print(f"ALLREDUCE world={world} size={size_mb}MB time={sec * 1e3:.3f}ms "
              f"busbw={algbw * 2 * (world - 1) / world:.1f}GB/s", flush=True)
dist.destroy_process_group()
