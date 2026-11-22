#!/bin/bash
set -euo pipefail
accelerate launch --num_processes=2 /outputs/rocm-train/jobs/fsdp_autowrap_train.py --policy.type=tdmpc --dataset.repo_id=lerobot/pusht --policy.device=cuda --policy.push_to_hub=false --batch_size=1 --steps=2 --env_eval_freq=0 --save_freq=2 --log_freq=1 --num_workers=0 --trackio.enable=false --dataset.episodes='[0]' --output_dir=/outputs/rocm-train/cover-tdmpc-fsdp2 --parallelism.dp_shard=2
