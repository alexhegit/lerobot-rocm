#!/bin/bash
set -euo pipefail
export FSDP_CAST_DTYPE=float32
rm -rf /outputs/rocm-train/smolvla-resume-fsdp2
echo "== seed: 2 steps"
accelerate launch --num_processes=2 /outputs/rocm-train/jobs/fsdp_autowrap_train.py --policy.path=lerobot/smolvla_base --policy.device=cuda --policy.push_to_hub=false   --dataset.repo_id=lerobot/svla_so100_pickplace --dataset.episodes='[0]'   --rename_map='{"observation.images.top": "observation.images.camera1", "observation.images.wrist": "observation.images.camera2"}' --parallelism.dp_shard=2 --accelerator.mixed_precision=bf16 --accelerator.fsdp.wrap_modules='["SmolVLMEncoderLayer"]' --batch_size=1 --steps=2 --env_eval_freq=0 --save_freq=2 --log_freq=1 --num_workers=0 --trackio.enable=false --output_dir=/outputs/rocm-train/smolvla-resume-fsdp2
test -d /outputs/rocm-train/smolvla-resume-fsdp2/checkpoints/000002
echo "== resume: steps 3-4"
accelerate launch --num_processes=2 /outputs/rocm-train/jobs/fsdp_autowrap_train.py --resume=true \
  --config_path=/outputs/rocm-train/smolvla-resume-fsdp2/checkpoints/000002/pretrained_model/train_config.json --steps=4
test -d /outputs/rocm-train/smolvla-resume-fsdp2/checkpoints/000004
find /outputs/rocm-train/smolvla-resume-fsdp2/checkpoints/000004 -maxdepth 2 | sort
cat /outputs/rocm-train/smolvla-resume-fsdp2/checkpoints/000004/training_state/training_step.json 2>/dev/null || true
echo "RESUME_OK"
