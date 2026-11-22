## Summary / Motivation

Qwen3-VL rejects a video with fewer frames than its temporal patch size (2):

```
ValueError: t:1 must be larger than temporal_factor:2
```

TOPReward hits this in two places:

- `python -m lerobot.rewards.topreward.compute_rabc_weights` starts its prefix sweep at a one-frame prefix, so labelling any episode for RA-BC fails on the first anchor.
- `compute_reward` on a single image `(B, C, H, W)` also fails.

This reproduces with the pinned `transformers>=5.22.0,<5.23.0` (5.22.2 here), so it is not hardware-specific.

## What changed

- `_prepare_video_batch` takes `min_frames` (default 1, so the current behaviour is unchanged) and pads shorter videos by repeating their last frame.
- `TOPRewardEncoderProcessorStep` passes the processor's `video_processor.temporal_patch_size` (falls back to 1 if absent). The frame count in `video_metadata` follows the padded video.

No breaking changes.

## How was this tested

- New tests in `tests/rewards/test_topreward_processor.py`:
  - `test_prepare_video_batch_pads_short_videos_with_last_frame`
  - `test_prepare_video_batch_min_frames_keeps_long_videos`
  - `test_encoder_step_pads_single_frame_to_temporal_patch` (stubbed processor with `temporal_patch_size = 2`)

  All 3 fail without the fix and pass with it.
- `pytest tests/rewards/test_topreward_processor.py tests/rewards/test_topreward.py tests/rewards/test_modeling_topreward.py`: 34 passed, 1 skipped (`requires cuda`, the container had no GPU attached).
- End to end with `Qwen/Qwen3-VL-8B-Instruct`:

  ```bash
  python -m lerobot.rewards.topreward.compute_rabc_weights \
      --dataset-repo-id lerobot/svla_so100_pickplace --episodes 0 --num-samples 15 --device cuda \
      --output-path topreward_progress.parquet
  ```

  Before: fails on the first prefix. After: 454 rows, progress 0.0 to 0.9715, no NaN. The resulting parquet then trains SmolVLA with `--sample_weighting.type=rabc`.

**Hardware note:** I only have AMD GPUs (MI300X, ROCm 7.14, torch 2.12). This was not run on an NVIDIA GPU. The change is in CPU-side preprocessing, so it should not depend on the device.

## Checklist (required before merge)

- [x] Linting/formatting run (`ruff check`, `ruff format --check` on the changed files)
- [x] All tests pass locally (the TOPReward tests above)
- [ ] Documentation updated (not needed)
- [ ] CI is green
- [x] Community Review: I have reviewed another contributor's open PR and linked it here: #5213 (ROCm test report: https://github.com/huggingface/lerobot/pull/5213#issuecomment-7029511117)

## Reviewer notes

- Repeating the last frame makes a single image a static two-frame clip. The alternative, starting the prefix sweep at two frames, would leave `compute_reward` on single images broken.
