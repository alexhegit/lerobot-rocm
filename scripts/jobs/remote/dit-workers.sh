#!/bin/bash
set -euo pipefail
lerobot-train \
  --policy.type=multi_task_dit \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --policy.horizon=32 \
  --policy.n_action_steps=24 \
  --policy.objective=diffusion \
  --policy.num_inference_steps=10 \
  --dataset.repo_id=lerobot/aloha_sim_transfer_cube_human \
  --dataset.episodes='[0]' \
  --batch_size=1 \
  --steps=20 \
  --env_eval_freq=0 \
  --save_freq=20 \
  --log_freq=5 \
  --num_workers=2 \
  --trackio.enable=false \
  --output_dir=/outputs/rocm-train/dit-workers
