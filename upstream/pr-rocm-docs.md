## Summary / Motivation

The AMD GPUs (ROCm) section of `docs/source/torch_accelerators.mdx` covers installation. This adds what came up while training LeRobot policies on 8× AMD Instinct MI300X with ROCm 7.14, PyTorch 2.12 and `docker/Dockerfile.rocm`, and lists the policies that trained there.

## What changed

Two subsections under "AMD GPUs (ROCm)":

- **Known issues on ROCm**, each with the symptom and what fixed it:
  - `--group-add render` failing where the group has no name: pass the numeric GIDs.
  - Multi-GPU training getting slower as GPUs are added: HIP runtimes with ROCm/rocm-systems#6134 (reverted in #6712) pin every process that initialises HIP, including `accelerate launch` and so all ranks and DataLoader workers, to the CPUs nearest GPU 0; `AMD_CPU_AFFINITY=0` fixes it (ACT on 8 GPUs: 706 → 4,122 samples/s). #5472 sets it in `Dockerfile.rocm`.
  - A slow first step from MIOpen kernel compilation: a persistent `MIOPEN_USER_DB_PATH` / `MIOPEN_CUSTOM_CACHE_DIR` (464 s → 28 s).
  - No EGL rendering on Instinct GPUs: OSMesa for MuJoCo evaluation.
  - PEFT needing `torchao > 0.16`.
  - No FlashAttention; FLUX3's Video VAE needing NATTEN.
- **Policies trained on ROCm**: the policies and reward models that completed training steps and saved a checkpoint on MI300X.

Documentation only.

## How was this tested

- Each item was hit and resolved on 8× MI300X; the numbers come from those runs.
- `prettier --check --prose-wrap=preserve` and `typos` pass on the file.

I only have AMD GPUs.

## Checklist (required before merge)

- [x] Linting/formatting run (prettier, typos on the changed file)
- [ ] All tests pass locally (documentation only)
- [x] Documentation updated
- [ ] CI is green
- [x] Community Review: I have reviewed another contributor's open PR and linked it here: #5213 (ROCm test report: https://github.com/huggingface/lerobot/pull/5213#issuecomment-7029511117)

## Reviewer notes

- Related: #5472 (`ENV AMD_CPU_AFFINITY=0` in `docker/Dockerfile.rocm`).
