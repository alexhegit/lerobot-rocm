Another data point for this regression, from PyTorch distributed training (LeRobot) on 8× MI300X, in case it helps anyone who lands here from a build that still has the `true` default.

**Build:** PyTorch `2.12.0+rocm7.14.0a20260608` (TheRock nightly wheel in `rocm/primus:v26.4`), 160 CPUs.

**What we saw:** `torch.cuda.device_count()` narrows the process's CPU affinity from 160 CPUs to 8 (CPUs 0-7). `accelerate launch` calls it to count GPUs, so all ranks and every DataLoader worker they spawn inherit those 8 CPUs, regardless of which GPU a rank uses. `AMD_CPU_AFFINITY=0` restores the full mask; `HSA_OVERRIDE_CPU_AFFINITY_DEBUG` has no effect.

**Impact** (LeRobot training throughput, samples/s, 8 DataLoader workers per rank):

| Model (batch per GPU) | GPUs | default | `AMD_CPU_AFFINITY=0` |
|---|---:|---:|---:|
| ACT (32) | 1 / 2 / 4 / 8 | 584 / 1056 / 968 / 706 | 579 / 1066 / 2112 / 4122 |
| SmolVLA (8) | 4 / 8 | 173 / 137 | 269 / 506 |
| π0 (4) | 8 | 21 | 53 |

With the narrowed mask, `torch.profiler` showed unchanged GPU time but about 3× CPU time per step (every `hipLaunchKernel` and ATen dispatch slower), since ranks and workers shared 8 CPUs. RCCL bandwidth was fine (all-reduce busbw 278 GB/s on 8 GPUs).

Glad to see the default reverted. We are adding `ENV AMD_CPU_AFFINITY=0` to LeRobot's ROCm Dockerfile so images built on runtimes from this window are not affected.
