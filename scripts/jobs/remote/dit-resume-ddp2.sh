#!/bin/bash
set -euo pipefail

rm -rf /outputs/rocm-train/dit-resume-ddp2
echo "== seed: 2 steps"
accelerate launch --num_processes=2 "$(which lerobot-train)" --policy.type=multi_task_dit --policy.device=cuda --policy.push_to_hub=false   --policy.horizon=32 --policy.n_action_steps=24 --policy.objective=diffusion --policy.num_inference_steps=10   --dataset.repo_id=lerobot/aloha_sim_transfer_cube_human --dataset.episodes='[0]'  --batch_size=1 --steps=2 --env_eval_freq=0 --save_freq=2 --log_freq=1 --num_workers=0 --trackio.enable=false --output_dir=/outputs/rocm-train/dit-resume-ddp2
test -d /outputs/rocm-train/dit-resume-ddp2/checkpoints/000002
echo "== resume: steps 3-4"
accelerate launch --num_processes=2 "$(which lerobot-train)" --resume=true \
  --config_path=/outputs/rocm-train/dit-resume-ddp2/checkpoints/000002/pretrained_model/train_config.json --steps=4
test -d /outputs/rocm-train/dit-resume-ddp2/checkpoints/000004
find /outputs/rocm-train/dit-resume-ddp2/checkpoints/000004 -maxdepth 2 | sort
cat /outputs/rocm-train/dit-resume-ddp2/checkpoints/000004/training_state/training_step.json 2>/dev/null || true
echo "RESUME_OK"
