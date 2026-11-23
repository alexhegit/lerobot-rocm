#!/bin/bash
set -euo pipefail
export PYTHONPATH=/outputs/rocm-train/latest-main/src
accelerate launch --num_processes=1 "$(which lerobot-train)" \
  --policy.path=lerobot/wall-oss-0.5 \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --policy.use_language_recipe=false \
  --policy.camera_key_mapping='{"observation.images.top":"face_view","observation.images.wrist":"right_wrist_view"}' \
  --dataset.repo_id=lerobot/svla_so100_pickplace \
  --dataset.episodes='[0]' \
  --batch_size=1 \
  --steps=2 \
  --env_eval_freq=0 \
  --save_freq=2 \
  --log_freq=1 \
  --num_workers=0 \
  --trackio.enable=false \
  --output_dir=/outputs/rocm-train/latest-wall-1g
