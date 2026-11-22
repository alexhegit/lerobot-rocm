#!/bin/bash
set -euo pipefail
lerobot-train --policy.path=lerobot/VLA-JEPA-Pretrain --policy.freeze_qwen=true \
  --policy.reinit_modules='["model.action_model.action_encoder","model.action_model.action_decoder","model.action_model.state_encoder"]' \
  --dataset.repo_id=lerobot/svla_so100_pickplace --policy.device=cuda --policy.push_to_hub=false --batch_size=1 --steps=2 --env_eval_freq=0 --save_freq=2 --log_freq=1 --num_workers=0 --trackio.enable=false --dataset.episodes='[0]' --output_dir=/outputs/rocm-train/cover-jepa
