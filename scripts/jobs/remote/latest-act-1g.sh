#!/bin/bash
set -euo pipefail
export PYTHONPATH=/outputs/rocm-train/latest-main/src
accelerate launch --num_processes=1 "$(which lerobot-train)" \
  --policy.type=act \
  --policy.dim_model=64 \
  --policy.n_action_steps=20 \
  --policy.chunk_size=20 \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --dataset.repo_id=lerobot/aloha_sim_transfer_cube_human \
  --dataset.episodes='[0]' \
  --batch_size=2 \
  --steps=2 \
  --env_eval_freq=0 \
  --save_freq=2 \
  --log_freq=1 \
  --num_workers=0 \
  --trackio.enable=false \
  --output_dir=/outputs/rocm-train/latest-act-1g
