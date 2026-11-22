#!/bin/bash
set -euo pipefail
lerobot-train --policy.path=Efficient-Large-Model/Long-WAM-LIBERO-IDM-LeRobot --policy.dtype=bfloat16 \
  --accelerator.mixed_precision=bf16 --dataset.repo_id=lerobot/libero --policy.device=cuda --policy.push_to_hub=false \
  --batch_size=1 --steps=2 --env_eval_freq=0 --save_freq=2 --log_freq=1 --num_workers=0 --trackio.enable=false \
  --dataset.episodes='[0]' --output_dir=/outputs/rocm-train/cover-longwam2
