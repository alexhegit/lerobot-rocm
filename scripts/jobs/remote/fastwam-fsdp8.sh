#!/bin/bash
set -euo pipefail
accelerate launch --num_processes=8 "$(which lerobot-train)" \
  --policy.path=lerobot/fastwam_base \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --policy.dtype=float32 \
  --policy.action_dim=7 \
  --policy.proprio_dim=8 \
  --dataset.repo_id=lerobot/libero \
  --dataset.episodes='[0]' \
  --batch_size=1 \
  --steps=2 \
  --env_eval_freq=0 \
  --save_freq=2 \
  --log_freq=1 \
  --num_workers=0 \
  --trackio.enable=false \
  --output_dir=/outputs/rocm-train/fastwam-fsdp8 --parallelism.dp_shard=8
