## Summary / Motivation

Weighted training (`--sample_weighting.type=rabc` or `uniform`) calls `policy.forward(batch, reduction="none")`. The RA-BC guide lists PI0, PI0.5, SmolVLA and FastWAM as supported. With another policy, for example ACT, `lerobot-train` loads the dataset, builds the model and the optimizer, and only then fails on the first step:

```
TypeError: ACTPolicy.forward() got an unexpected keyword argument 'reduction'
```

This PR fails right after the policy is created, with a message that says what is missing.

## What changed

- `lerobot.utils.sample_weighting.check_policy_supports_sample_weighting(policy)` raises `ValueError` when `policy.forward` has no `reduction` parameter.
- `lerobot_train.train()` calls it after `make_policy` when `cfg.sample_weighting` is set, before PEFT wrapping, so the check sees the policy's own `forward`.

New message:

```
ValueError: Sample weighting needs per-sample losses, but ACTPolicy.forward() has no `reduction` argument. Use a policy whose forward(batch, reduction='none') returns one loss per sample, or remove --sample_weighting.
```

No change for supported policies.

## How was this tested

- New tests in `tests/utils/test_sample_weighting.py`: one stub policy with `reduction` (accepted) and one without (rejected).
- `pytest tests/utils/test_sample_weighting.py`: 24 passed, 1 skipped (`CUDA not available`, the container had no GPU attached).
- The check against the real policy classes: `act` and `diffusion` are rejected; `pi0`, `pi05`, `smolvla`, `fastwam` are accepted, matching the RA-BC guide.
- Found while running RA-BC end to end: Robometer / TOPReward progress on `lerobot/svla_so100_pickplace` episode 0, then 2 training steps. ACT failed with the `TypeError` above; SmolVLA trained.

**Hardware note:** I only have AMD GPUs (MI300X, ROCm 7.14, torch 2.12). This was not run on an NVIDIA GPU. The check only inspects a function signature.

## Checklist (required before merge)

- [x] Linting/formatting run (`ruff check`, `ruff format --check` on the changed files)
- [x] All tests pass locally (`tests/utils/test_sample_weighting.py`)
- [ ] Documentation updated (not needed)
- [ ] CI is green
- [x] Community Review: I have reviewed another contributor's open PR and linked it here: #5213 (ROCm test report: https://github.com/huggingface/lerobot/pull/5213#issuecomment-7029511117)

## Reviewer notes

- Adding `reduction` support to ACT and Diffusion would be a separate change.
