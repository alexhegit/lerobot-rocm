#!/bin/bash
set -euo pipefail
lerobot-train --reward_model.type=sarm --reward_model.annotation_mode=single_stage \
  --reward_model.image_key=observation.images.top --reward_model.device=cuda --reward_model.push_to_hub=false \
  --dataset.repo_id=lerobot/svla_so100_pickplace --dataset.episodes='[0]' --batch_size=1 --steps=2 \
  --env_eval_freq=0 --save_freq=2 --log_freq=1 --num_workers=0 --trackio.enable=false \
  --output_dir=/outputs/rocm-train/cover-sarm
