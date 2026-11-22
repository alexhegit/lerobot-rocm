#!/bin/bash
set -euo pipefail
accelerate launch --num_processes=2 /outputs/rocm-train/jobs/fsdp_autowrap_train.py --policy.type=evo1 --policy.training_stage=stage1 --policy.vlm_model_name=OpenGVLab/InternVL3-1B-hf \
  --policy.chunk_size=16 --policy.n_action_steps=16 --policy.max_state_dim=24 --policy.max_action_dim=24 \
  --dataset.repo_id=lerobot/svla_so100_pickplace --policy.device=cuda --policy.push_to_hub=false --batch_size=1 --steps=2 --env_eval_freq=0 --save_freq=2 --log_freq=1 --num_workers=0 --trackio.enable=false --dataset.episodes='[0]' --output_dir=/outputs/rocm-train/cover-evo1-fsdp2 --parallelism.dp_shard=2
