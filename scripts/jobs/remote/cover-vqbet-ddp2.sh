#!/bin/bash
set -euo pipefail
accelerate launch --num_processes=2 "$(which lerobot-train)" --policy.type=vqbet --dataset.repo_id=lerobot/aloha_sim_transfer_cube_human --policy.device=cuda --policy.push_to_hub=false --batch_size=1 --steps=2 --env_eval_freq=0 --save_freq=2 --log_freq=1 --num_workers=0 --trackio.enable=false --dataset.episodes='[0]' --output_dir=/outputs/rocm-train/cover-vqbet-ddp2
