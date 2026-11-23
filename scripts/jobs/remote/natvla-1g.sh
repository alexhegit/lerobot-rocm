#!/bin/bash
# NAT VLA (added upstream in #5416, not in the image): run from the latest-main source tree.
set -euo pipefail
export PYTHONPATH=/outputs/rocm-train/latest-main/src
python3 -c "import lerobot; print('lerobot from', lerobot.__file__)"
accelerate launch --num_processes=1 "$(which lerobot-train)" \
  --policy.type=nat_vla \
  --policy.setup="single franka robotic arm in libero" \
  --policy.control_mode="delta end-effector pose" \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --dataset.repo_id=lerobot/libero \
  --dataset.episodes='[0]' \
  --batch_size=1 \
  --steps=2 \
  --env_eval_freq=0 \
  --save_freq=2 \
  --log_freq=1 \
  --num_workers=0 \
  --trackio.enable=false \
  --output_dir=/outputs/rocm-train/natvla-1g
