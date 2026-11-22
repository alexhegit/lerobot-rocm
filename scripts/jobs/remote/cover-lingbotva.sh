#!/bin/bash
set -euo pipefail
lerobot-train --policy.path=lerobot/lingbot_va_libero_long --policy.attn_mode=flex \
  --peft.method_type=LORA --peft.r=8 --peft.lora_alpha=8 \
  --peft.target_modules='transformer\.blocks\.\d+\.attn[12]\.(to_q|to_v)' \
  --dataset.repo_id=lerobot/libero --policy.device=cuda --policy.push_to_hub=false --batch_size=1 --steps=2 --env_eval_freq=0 --save_freq=2 --log_freq=1 --num_workers=0 --trackio.enable=false --dataset.episodes='[0]' --output_dir=/outputs/rocm-train/cover-lingbotva
