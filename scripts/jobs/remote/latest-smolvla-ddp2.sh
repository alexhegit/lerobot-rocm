#!/bin/bash
set -euo pipefail
export PYTHONPATH=/outputs/rocm-train/latest-main/src
accelerate launch --num_processes=2 "$(which lerobot-train)" \
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
  --output_dir=/outputs/rocm-train/latest-smolvla-ddp2
