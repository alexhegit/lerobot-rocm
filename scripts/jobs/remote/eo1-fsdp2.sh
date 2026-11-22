#!/bin/bash
set -euo pipefail
bash /outputs/rocm-train/jobs/eo1-prep.sh
accelerate launch --num_processes=2 /outputs/rocm-train/jobs/fsdp_autowrap_train.py \
  --policy.path=/outputs/rocm-train/eo1-base-compat \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --policy.dtype=bfloat16 \
  --policy.attn_implementation=sdpa \
  --dataset.repo_id=lerobot/svla_so100_pickplace \
  --dataset.episodes='[0]' \
  --batch_size=1 \
  --steps=2 \
  --env_eval_freq=0 \
  --save_freq=2 \
  --log_freq=1 \
  --num_workers=0 \
  --trackio.enable=false \
  --output_dir=/outputs/rocm-train/eo1-fsdp2 --parallelism.dp_shard=2  --policy.dtype=float32 --accelerator.mixed_precision=bf16
