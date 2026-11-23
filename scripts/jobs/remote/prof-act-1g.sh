#!/bin/bash
# torch.profiler over ACT steps 30-39 (rank 0), 1 GPU(s), batch 32 per GPU, full dataset.
set -euo pipefail
export MIOPEN_USER_DB_PATH=/outputs/rocm-train/miopen/fast/db
export MIOPEN_CUSTOM_CACHE_DIR=/outputs/rocm-train/miopen/fast/cache
OUT=/outputs/rocm-train/prof-act-1g-$(date +%s)
accelerate launch --num_processes=1 /outputs/rocm-train/jobs/profile_train.py \
  --policy.type=act --policy.dim_model=64 --policy.n_action_steps=20 --policy.chunk_size=20 \
  --policy.device=cuda --policy.push_to_hub=false \
  --dataset.repo_id=lerobot/aloha_sim_transfer_cube_human \
  --batch_size=32 --steps=42 --log_freq=42 --save_freq=100000 --env_eval_freq=0 --num_workers=8 \
  --trackio.enable=false --output_dir="$OUT"
