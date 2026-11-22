#!/bin/bash
set -euo pipefail
accelerate launch --num_processes=2 "$(which lerobot-train)" --policy.path=lerobot/dm05_base --policy.add_state=false --policy.chunk_size=10 --policy.n_action_steps=10 \
  --rename_map='{"observation.images.image":"observation.images.front","observation.images.image2":"observation.images.wrist"}' \
  --dataset.repo_id=lerobot/libero --policy.device=cuda --policy.push_to_hub=false --batch_size=1 --steps=2 --env_eval_freq=0 --save_freq=2 --log_freq=1 --num_workers=0 --trackio.enable=false --dataset.episodes='[0]' --output_dir=/outputs/rocm-train/cover-dm05-ddp2
