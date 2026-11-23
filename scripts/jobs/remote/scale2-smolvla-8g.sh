#!/bin/bash
set -euo pipefail
export MIOPEN_USER_DB_PATH=/outputs/rocm-train/miopen/fast/db
export MIOPEN_CUSTOM_CACHE_DIR=/outputs/rocm-train/miopen/fast/cache
mkdir -p $MIOPEN_USER_DB_PATH $MIOPEN_CUSTOM_CACHE_DIR
accelerate launch --num_processes=8 "$(which lerobot-train)" \
  --policy.path=lerobot/smolvla_base \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --dataset.repo_id=lerobot/svla_so100_pickplace \
  --dataset.episodes='[0]' \
  --rename_map='{"observation.images.top": "observation.images.camera1", "observation.images.wrist": "observation.images.camera2"}' \
  --batch_size=1 \
  --steps=2 \
  --env_eval_freq=0 \
  --save_freq=2 \
  --log_freq=1 \
  --num_workers=0 \
  --trackio.enable=false \
  --output_dir=/outputs/rocm-train/scale2-smolvla-8g --batch_size=8 --steps=60 --log_freq=20 --save_freq=100000 --num_workers=8 --dataset.episodes=null
