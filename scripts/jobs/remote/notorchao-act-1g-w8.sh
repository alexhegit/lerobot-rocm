#!/bin/bash
set -euo pipefail
pip uninstall -y -q torchao
python3 -c "import torch; from lerobot.datasets import LeRobotDataset; print('NO_TORCHAO cuda_initialized_after_dataset_import', torch.cuda.is_initialized())"
export MIOPEN_USER_DB_PATH=/outputs/rocm-train/miopen/fast/db
export MIOPEN_CUSTOM_CACHE_DIR=/outputs/rocm-train/miopen/fast/cache
accelerate launch --num_processes=1 "$(which lerobot-train)" \
  --policy.type=act \
  --policy.dim_model=64 \
  --policy.n_action_steps=20 \
  --policy.chunk_size=20 \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --dataset.repo_id=lerobot/aloha_sim_transfer_cube_human \
  --dataset.episodes='[0]' \
  --batch_size=2 \
  --steps=2 \
  --env_eval_freq=0 \
  --save_freq=2 \
  --log_freq=1 \
  --num_workers=0 \
  --trackio.enable=false \
  --output_dir=/outputs/rocm-train/notorchao-act-1g-w8 --batch_size=32 --steps=60 --log_freq=20 --save_freq=100000 --dataset.episodes=null --num_workers=8
