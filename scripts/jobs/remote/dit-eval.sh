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
  --env.type=aloha \
  --env_eval_freq=2 \
  --eval.n_episodes=1 \
  --eval.batch_size=1 \
  --eval.max_episode_steps=2 \
  --batch_size=1 \
  --steps=2 \
  --save_freq=2 \
  --log_freq=1 \
  --num_workers=0 \
  --trackio.enable=false \
  --output_dir=/outputs/rocm-train/dit-eval
