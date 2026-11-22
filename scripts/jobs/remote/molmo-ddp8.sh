#!/bin/bash
set -euo pipefail
bash /outputs/rocm-train/jobs/molmo-prep.sh
accelerate launch --num_processes=8 "$(which lerobot-train)" \
  --policy.path=/outputs/rocm-train/molmo-compat \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --policy.dtype=bfloat16 \
  --policy.train_mode_vlm=fft \
  --policy.gradient_checkpointing=true \
  --rename_map='{"observation.images.top":"observation.images.cam0","observation.images.wrist":"observation.images.cam1"}' \
  --dataset.repo_id=lerobot/svla_so100_pickplace \
  --dataset.episodes='[0]' \
  --batch_size=1 \
  --steps=2 \
  --env_eval_freq=0 \
  --save_freq=2 \
  --log_freq=1 \
  --num_workers=0 \
  --trackio.enable=false \
  --output_dir=/outputs/rocm-train/molmo-ddp8
