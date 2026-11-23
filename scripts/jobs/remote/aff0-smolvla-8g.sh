#!/bin/bash
set -euo pipefail
export AMD_CPU_AFFINITY=0
python3 -c "import os, torch; torch.cuda.device_count(); print('AFFINITY_AFTER_HIP_INIT', len(os.sched_getaffinity(0)))"
export MIOPEN_USER_DB_PATH=/outputs/rocm-train/miopen/fast/db
export MIOPEN_CUSTOM_CACHE_DIR=/outputs/rocm-train/miopen/fast/cache
accelerate launch --num_processes=8 "$(which lerobot-train)" \
  --policy.path=lerobot/smolvla_base \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --dataset.repo_id=lerobot/svla_so100_pickplace \
  --dataset.episodes='[0]' \
  --rename_map='{"observation.images.top": "observation.images.camera1", "observation.images.wrist": "observation.images.camera2"}' \
  --batch_size=1 \
  --steps=2 \
  --env_eval_freq=0 \
  --save_freq=2 \
  --log_freq=1 \
  --num_workers=0 \
  --trackio.enable=false \
  --output_dir=/outputs/rocm-train/aff0-smolvla-8g --batch_size=8 --steps=60 --log_freq=20 --save_freq=100000 --num_workers=8 --dataset.episodes=null
