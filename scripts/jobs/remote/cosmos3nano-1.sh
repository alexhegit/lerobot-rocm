#!/bin/bash
set -euo pipefail
lerobot-train \
  --policy.type=cosmos3_nano \
  --policy.pretrained_path=lerobot/Cosmos3-Nano \
  --policy.load_action_modules=false \
  --policy.gradient_checkpointing=true \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --accelerator.mixed_precision=bf16 \
  --dataset.repo_id=lerobot/svla_so100_pickplace \
  --dataset.episodes='[0]' \
  --batch_size=1 \
  --steps=2 \
  --env_eval_freq=0 \
  --save_freq=2 \
  --log_freq=1 \
  --num_workers=0 \
  --trackio.enable=false \
  --output_dir=/outputs/rocm-train/cosmos3nano-1
