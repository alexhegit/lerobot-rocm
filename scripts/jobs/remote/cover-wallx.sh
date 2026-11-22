#!/bin/bash
set -euo pipefail
lerobot-train --policy.type=wall_x --policy.pretrained_name_or_path=x-square-robot/wall-oss-flow \
  --policy.prediction_mode=diffusion --policy.attn_implementation=eager --policy.recipe=null \
  --dataset.repo_id=lerobot/svla_so100_pickplace --policy.device=cuda --policy.push_to_hub=false --batch_size=1 --steps=2 --env_eval_freq=0 --save_freq=2 --log_freq=1 --num_workers=0 --trackio.enable=false --dataset.episodes='[0]' --output_dir=/outputs/rocm-train/cover-wallx
