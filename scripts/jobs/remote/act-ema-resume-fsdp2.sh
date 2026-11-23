#!/bin/bash
set -euo pipefail

rm -rf /outputs/rocm-train/act-ema-resume-fsdp2
echo "== seed: 2 steps"
accelerate launch --num_processes=2 "$(which lerobot-train)" --policy.type=act --policy.dim_model=64 --policy.n_action_steps=20 --policy.chunk_size=20   --policy.device=cuda --policy.push_to_hub=false   --dataset.repo_id=lerobot/aloha_sim_transfer_cube_human --dataset.episodes='[0]' --parallelism.dp_shard=2 --ema.type=constant --batch_size=1 --steps=2 --env_eval_freq=0 --save_freq=2 --log_freq=1 --num_workers=0 --trackio.enable=false --output_dir=/outputs/rocm-train/act-ema-resume-fsdp2
test -d /outputs/rocm-train/act-ema-resume-fsdp2/checkpoints/000002
echo "== resume: steps 3-4"
accelerate launch --num_processes=2 "$(which lerobot-train)" --resume=true \
  --config_path=/outputs/rocm-train/act-ema-resume-fsdp2/checkpoints/000002/pretrained_model/train_config.json --steps=4
test -d /outputs/rocm-train/act-ema-resume-fsdp2/checkpoints/000004
find /outputs/rocm-train/act-ema-resume-fsdp2/checkpoints/000004 -maxdepth 2 | sort
cat /outputs/rocm-train/act-ema-resume-fsdp2/checkpoints/000004/training_state/training_step.json 2>/dev/null || true
echo "RESUME_OK"
