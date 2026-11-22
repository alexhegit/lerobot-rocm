#!/bin/bash
set -euo pipefail
export FSDP_CAST_DTYPE=bfloat16
accelerate launch --num_processes=2 /outputs/rocm-train/jobs/fsdp_autowrap_train.py --policy.type=wall_x --policy.pretrained_name_or_path=x-square-robot/wall-oss-flow \
  --policy.prediction_mode=diffusion --policy.attn_implementation=eager --policy.recipe=null \
  --dataset.repo_id=lerobot/svla_so100_pickplace --policy.device=cuda --policy.push_to_hub=false --batch_size=1 --steps=2 --env_eval_freq=0 --save_freq=2 --log_freq=1 --num_workers=0 --trackio.enable=false --dataset.episodes='[0]' --output_dir=/outputs/rocm-train/cover-wallx-fsdp2 --parallelism.dp_shard=2
