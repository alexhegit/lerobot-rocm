#!/bin/bash
set -euo pipefail
export FSDP_WRAP_STRATEGY=params
accelerate launch --num_processes=2 /outputs/rocm-train/jobs/fsdp_autowrap_train.py \
  --policy.path=lerobot/fineart_vla_base \
  --policy.dtype=bfloat16 \
  --policy.gradient_checkpointing=true \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --dataset.repo_id=local/fineart_aloha3 \
  --dataset.root=/outputs/rocm-train/fineart-aloha3 \
  --dataset.episodes='[0]' \
  --batch_size=1 --steps=2 --env_eval_freq=0 --save_freq=2 --log_freq=1 \
  --num_workers=0 --trackio.enable=false \
  --output_dir=/outputs/rocm-train/cover-fineart-fsdp2-pw --parallelism.dp_shard=2
