#!/bin/bash
set -euo pipefail
export FSDP_WRAP_STRATEGY=params
accelerate launch --num_processes=2 /outputs/rocm-train/jobs/fsdp_autowrap_train.py --policy.path=Efficient-Large-Model/Long-WAM-LIBERO-IDM-LeRobot --policy.dtype=bfloat16 \
  --accelerator.mixed_precision=bf16 --dataset.repo_id=lerobot/libero --policy.device=cuda --policy.push_to_hub=false \
  --batch_size=1 --steps=2 --env_eval_freq=0 --save_freq=2 --log_freq=1 --num_workers=0 --trackio.enable=false \
  --dataset.episodes='[0]' --output_dir=/outputs/rocm-train/cover-longwam-fsdp2-pw --parallelism.dp_shard=2
