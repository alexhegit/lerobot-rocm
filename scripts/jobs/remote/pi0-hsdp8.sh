#!/bin/bash
set -euo pipefail
accelerate launch --num_processes=8 "$(which lerobot-train)" \
  --policy.type=pi0 \
  --policy.pretrained_path=lerobot/pi0_base \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --policy.dtype=bfloat16 \
  --policy.empty_cameras=1 \
  --policy.gradient_checkpointing=true \
  --policy.compile_model=false \
  --dataset.repo_id=lerobot/svla_so100_pickplace \
  --dataset.episodes='[0]' \
  --rename_map='{"observation.images.top":"observation.images.base_0_rgb","observation.images.wrist":"observation.images.left_wrist_0_rgb"}' \
  --batch_size=1 \
  --steps=2 \
  --env_eval_freq=0 \
  --save_freq=2 \
  --log_freq=1 \
  --num_workers=0 \
  --trackio.enable=false \
  --output_dir=/outputs/rocm-train/pi0-hsdp8 --parallelism.dp_replicate=2 --parallelism.dp_shard=4 --accelerator.fsdp.wrap_modules='["SiglipEncoderLayer"]' --policy.dtype=float32 --accelerator.mixed_precision=bf16
