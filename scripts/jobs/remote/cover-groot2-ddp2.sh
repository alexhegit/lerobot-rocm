#!/bin/bash
set -euo pipefail
accelerate launch --num_processes=2 "$(which lerobot-train)" \
  --policy.type=groot \
  --policy.base_model_path=nvidia/GR00T-N1.7-3B \
  --policy.embodiment_tag=new_embodiment \
  --policy.use_bf16=true \
  --policy.use_relative_actions=false \
  --policy.chunk_size=16 \
  --policy.n_action_steps=16 \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --dataset.repo_id=lerobot/svla_so100_pickplace \
  --dataset.episodes='[0]' \
  --batch_size=1 --steps=2 --env_eval_freq=0 --save_freq=2 --log_freq=1 \
  --num_workers=0 --trackio.enable=false \
  --output_dir=/outputs/rocm-train/cover-groot2-ddp2
