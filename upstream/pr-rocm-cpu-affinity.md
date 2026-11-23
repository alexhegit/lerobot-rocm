## Summary / Motivation

Multi-GPU training in the ROCm image slows down as GPUs are added. Some HIP runtimes narrow the CPU affinity of any process that initialises HIP to the CPUs nearest GPU 0 ([ROCm/rocm-systems#6134](https://github.com/ROCm/rocm-systems/pull/6134), reverted in [#6712](https://github.com/ROCm/rocm-systems/pull/6712)). `accelerate launch` calls `torch.cuda.device_count()`, so the launcher, every rank it spawns and every DataLoader worker share those CPUs.

The runtime in the default base image (`rocm/primus:v26.4`, PyTorch `2.12.0+rocm7.14.0a20260608`) has this behaviour: on an 8× MI300X node with 160 CPUs, `torch.cuda.device_count()` takes the process from 160 CPUs to 8.

## What changed

- `docker/Dockerfile.rocm` sets `ENV AMD_CPU_AFFINITY=0`, the runtime's upstream default again since #6712, so images built on the affected runtimes keep the full CPU set.

## How was this tested

On 8× AMD MI300X (ROCm 7.14), in an image built from `docker/Dockerfile.rocm`, with `AMD_CPU_AFFINITY=0` set at run time (equivalent to the new `ENV`):

- Affinity after `torch.cuda.device_count()`: 8 CPUs by default, 160 with `AMD_CPU_AFFINITY=0`. `HSA_OVERRIDE_CPU_AFFINITY_DEBUG` has no effect.
- `lerobot-train` throughput, samples/s, 60 steps, 8 DataLoader workers per rank, `accelerate launch --num_processes=N`:

  | Policy (batch per GPU) | GPUs | default | `AMD_CPU_AFFINITY=0` |
  |---|---:|---:|---:|
  | ACT (32) | 1 / 2 / 4 / 8 | 584 / 1056 / 968 / 706 | 579 / 1066 / 2112 / 4122 |
  | SmolVLA (8) | 4 / 8 | 173 / 137 | 269 / 506 |
  | π0 (4) | 8 | 21 | 53 |

- How it was narrowed down: RCCL all-reduce bandwidth was fine (278 GB/s bus bandwidth on 8 GPUs); eight independent single-GPU runs at once did not slow down; `torch.profiler` showed the same GPU time per step under DDP but about 3× the CPU time; DDP with `num_workers=0` was not slowed down; `taskset` showed every rank pinned to CPUs 0-7.

I did not rebuild the 81 GB image for this one-line change; the runs above set the variable in the container instead.

## Checklist (required before merge)

- [ ] Linting/formatting run (`pre-commit run -a`) (Dockerfile only)
- [ ] All tests pass locally (no Python change)
- [ ] Documentation updated (a ROCm known-issues section for `torch_accelerators.mdx` follows in a separate PR)
- [ ] CI is green
- [x] Community Review: I have reviewed another contributor's open PR and linked it here: #5213 (ROCm test report: https://github.com/huggingface/lerobot/pull/5213#issuecomment-7029511117)

## Reviewer notes

- Seen on a VM (MI300X VF). On bare metal the narrowed set would be the CPUs of GPU 0's NUMA node rather than 8 CPUs, which is still shared by all ranks.
