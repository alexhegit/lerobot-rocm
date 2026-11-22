#!/bin/bash
set -euo pipefail
lerobot-train --policy.path=lerobot/fineart_vla_base --policy.dtype=bfloat16 --policy.gradient_checkpointing=true \
  --dataset.repo_id=lerobot/aloha_sim_transfer_cube_human --policy.device=cuda --policy.push_to_hub=false --batch_size=1 --steps=2 --env_eval_freq=0 --save_freq=2 --log_freq=1 --num_workers=0 --trackio.enable=false --dataset.episodes='[0]' --output_dir=/outputs/rocm-train/cover-fineart-aloha
