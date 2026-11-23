# PR draft: fix(train): build the sample weighter from the unwrapped policy

Local branch `fix/rabc-ddp-unwrap` (worktree `~/Repo/lerobot-rabc-ddp`), commit `6b51d9e2`, based on `main` `e5be0c16`. Not pushed.

## Summary / Motivation

RA-BC (`--sample_weighting.type=rabc`) fails on the first multi-GPU DDP run:

```
AttributeError: 'DistributedDataParallel' object has no attribute 'config'
```

`lerobot_train.train()` creates the sample weighter after `accelerator.prepare()` and passes the prepared `policy`. Under DDP that is the `DistributedDataParallel` wrapper, and `_make_rabc_weighter` reads `policy.config.chunk_size`. FSDP2 keeps the module's class, so sharded runs work; single-GPU runs are unaffected.

## What changed

- `make_sample_weighter(...)` now receives `accelerator.unwrap_model(policy)`, like the EMA and evaluation code in the same function.

## How was this tested

On 2× AMD MI300X (ROCm 7.14, torch 2.12), SmolVLA on `lerobot/svla_so100_pickplace` episode 0 with a Robometer progress file:

```bash
accelerate launch --num_processes=2 $(which lerobot-train) \
  --policy.path=lerobot/smolvla_base --dataset.repo_id=lerobot/svla_so100_pickplace --dataset.episodes='[0]' \
  --rename_map='{"observation.images.top": "observation.images.camera1", "observation.images.wrist": "observation.images.camera2"}' \
  --sample_weighting.type=rabc --sample_weighting.progress_path=robometer_progress.parquet --sample_weighting.head_mode=sparse \
  --batch_size=4 --steps=2
```

- Before: `AttributeError` above on both ranks.
- After: trains 2 steps (`dp_replicate=2`), `sample_weight_*` metrics logged.
- The same run with `--parallelism.dp_shard=2` (FSDP2) passed before and after.

I only have AMD GPUs; not run on NVIDIA. The change does not touch device code.

## Checklist

- [ ] Linting/formatting run
- [x] Relevant run passes (see above)
- [ ] Documentation updated (not needed)
- [ ] CI is green
- [x] Community Review: #5213 (ROCm test report)
