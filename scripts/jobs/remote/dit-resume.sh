#!/bin/bash
set -euo pipefail
lerobot-train \
  --resume=true \
  --config_path=/outputs/rocm-train/dit-resume-seed/checkpoints/000002/pretrained_model/train_config.json \
  --steps=4
