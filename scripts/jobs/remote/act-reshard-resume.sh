#!/bin/bash
set -euo pipefail
OUT=/outputs/rocm-train/act-reshard-resume
rm -rf "$OUT"
accelerate launch --num_processes=2 "$(which lerobot-train)" --policy.type=act --policy.dim_model=64 --policy.n_action_steps=20 --policy.chunk_size=20   --policy.device=cuda --policy.push_to_hub=false   --dataset.repo_id=lerobot/aloha_sim_transfer_cube_human --dataset.episodes='[0]' --parallelism.dp_shard=2 --batch_size=2 --steps=2 --env_eval_freq=0 --save_freq=2 --log_freq=1 --num_workers=0 --trackio.enable=false  --output_dir=$OUT
accelerate launch --num_processes=4 "$(which lerobot-train)" --resume=true --config_path=$OUT/checkpoints/000002/pretrained_model/train_config.json --steps=4 --parallelism.dp_shard=4
test -d "$OUT/checkpoints/000004"; find "$OUT/checkpoints/000004" -maxdepth 2 | sort; echo TOPOLOGY_OK
