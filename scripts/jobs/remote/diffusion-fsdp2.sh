#!/bin/bash
set -euo pipefail
accelerate launch --num_processes=2 /outputs/rocm-train/jobs/fsdp_autowrap_train.py \
  --policy.type=diffusion \
  --policy.down_dims='[64,128,256]' \
  --policy.diffusion_step_embed_dim=32 \
  --policy.num_inference_steps=10 \
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
  --output_dir=/outputs/rocm-train/diffusion-fsdp2 --parallelism.dp_shard=2
