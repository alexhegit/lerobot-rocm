#!/bin/bash
set -euo pipefail
accelerate launch --num_processes=2 "$(which lerobot-train)" \
  --policy.path=lerobot/g05_so101 \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --policy.dtype=bfloat16 \
  --policy.use_language_recipe=false \
  --policy.predict_cot=false \
  --policy.runtime_system=system1 \
  --policy.camera_order='["observation.images.top", "observation.images.wrist_left", "observation.images.wrist"]' \
  --policy.n_obs_steps=1 \
  --dataset.repo_id=lerobot/svla_so100_pickplace \
  --dataset.episodes='[0]' \
  --batch_size=1 \
  --steps=2 \
  --env_eval_freq=0 \
  --save_freq=2 \
  --log_freq=1 \
  --num_workers=0 \
  --trackio.enable=false \
  --output_dir=/outputs/rocm-train/g05-ddp2
