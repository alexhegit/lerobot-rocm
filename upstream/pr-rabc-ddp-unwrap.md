## Summary / Motivation

RA-BC (`--sample_weighting.type=rabc`) crashes at the start of any multi-GPU DDP run:

```
AttributeError: 'DistributedDataParallel' object has no attribute 'config'
```

`lerobot_train.train()` creates the sample weighter after `accelerator.prepare()`, so under DDP `make_sample_weighter` receives the `DistributedDataParallel` wrapper, and `_make_rabc_weighter` reads `policy.config.chunk_size`. FSDP2 keeps the module's class, so sharded runs work; single-GPU runs are unaffected.

## What changed

- `make_sample_weighter` unwraps a `DistributedDataParallel` policy before building the weighter, so every caller (the training script, custom loops) gets the fix.

## How was this tested

- New `tests/utils/test_sample_weighting.py::test_factory_rabc_accepts_ddp_wrapped_policy`: a real `DistributedDataParallel` on a single-process gloo group (as `tests/policies/praxis/test_praxis_training.py` does). It fails without the fix with the error above and passes with it.
- `pytest tests/utils/test_sample_weighting.py`: 23 passed, 1 skipped (`CUDA not available`).
- `ruff check`, `ruff format --check`, and `mypy` on `src/lerobot/utils/sample_weighting.py` pass.
- End to end on 2× AMD MI300X (ROCm 7.14, torch 2.12): SmolVLA on `lerobot/svla_so100_pickplace` with a Robometer progress file:

  ```bash
  accelerate launch --num_processes=2 $(which lerobot-train) \
    --policy.path=lerobot/smolvla_base --dataset.repo_id=lerobot/svla_so100_pickplace --dataset.episodes='[0]' \
    --rename_map='{"observation.images.top": "observation.images.camera1", "observation.images.wrist": "observation.images.camera2"}' \
    --sample_weighting.type=rabc --sample_weighting.progress_path=robometer_progress.parquet --sample_weighting.head_mode=sparse \
    --batch_size=4 --steps=2
  ```

  Before: the `AttributeError` above on both ranks. After: trains (`dp_replicate=2`) with `sample_weight_*` metrics logged. The same run with `--parallelism.dp_shard=2` (FSDP2) passed before and after.

**Hardware note:** I only have AMD GPUs; not run on NVIDIA. The change does not touch device code.

## Checklist (required before merge)

- [x] Linting/formatting run (`ruff check`, `ruff format --check`, `mypy` on the changed module)
- [x] All tests pass locally (`tests/utils/test_sample_weighting.py`)
- [ ] Documentation updated (not needed)
- [ ] CI is green
- [x] Community Review: I have reviewed another contributor's open PR and linked it here: #5213 (ROCm test report: https://github.com/huggingface/lerobot/pull/5213#issuecomment-7029511117)

## Reviewer notes

- Related: #5433 (fail early when RA-BC is used with a policy whose `forward` has no `reduction`).
