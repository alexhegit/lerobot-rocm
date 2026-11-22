#!/bin/bash
set -euo pipefail
accelerate launch --num_processes=2 /outputs/rocm-train/jobs/lawam_fsdp_cast.py \
  --policy.path=jialei02/lawam-pretrain-lerobot \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --policy.dtype=float32 \
  --dataset.image_transforms.enable=false \
  --rename_map='{"observation.images.top":"observation.images.image","observation.images.wrist":"observation.images.image2"}' \
  --parallelism.dp_shard=2 \
  --dataset.repo_id=lerobot/svla_so100_pickplace \
  --dataset.episodes='[0]' \
  --batch_size=1 \
  --steps=2 \
  --env_eval_freq=0 \
  --save_freq=2 \
  --log_freq=1 \
  --num_workers=0 \
  --trackio.enable=false \
  --output_dir=/outputs/rocm-train/lawam-fsdp2
