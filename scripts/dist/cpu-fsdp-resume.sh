#!/bin/bash
# CPU-only reproductions (2 processes, torchrun, gloo; no GPU devices in the container) of two FSDP issues
# seen on MI300X, to show they are device-independent:
#   1. --accelerator.fsdp.min_num_params: fully_shard rejects the ModuleList it selects.
#   2. FSDP resume of Multi-task DiT: "Missing key in checkpoint state_dict: optimizer.state...".
set -uo pipefail
STAMP=$(date +%s)
COMMON="--policy.type=multi_task_dit --policy.device=cpu --policy.push_to_hub=false \
  --policy.horizon=32 --policy.n_action_steps=24 --policy.objective=diffusion --policy.num_inference_steps=10 \
  --dataset.repo_id=lerobot/aloha_sim_transfer_cube_human --dataset.episodes=[0] \
  --batch_size=1 --env_eval_freq=0 --save_freq=2 --log_freq=1 --num_workers=0 --trackio.enable=false \
  --parallelism.dp_shard=2 --accelerator.init_process_group.backend=gloo"
TRAIN="torchrun --nproc-per-node=2 $(which lerobot-train)"
python3 -c "import torch; print('cuda available', torch.cuda.is_available())"

echo "== 1. size-based wrap (min_num_params)"
$TRAIN $COMMON --accelerator.fsdp.min_num_params=100000 --steps=2 \
  --output_dir=/outputs/rocm-train/cpu-fsdp-minparams-$STAMP 2>&1 | grep -aE "fully_shard does not support|step:[0-9]" | head -n 3
echo "MINPARAMS_EXIT=${PIPESTATUS[0]}"

echo "== 2. resume, seed: 2 steps"
OUT=/outputs/rocm-train/cpu-fsdp-resume-$STAMP
$TRAIN $COMMON --accelerator.fsdp.wrap_modules=[\"CLIPEncoderLayer\",\"TransformerBlock\"] --steps=2 \
  --output_dir="$OUT" || exit 1
ls "$OUT/checkpoints/000002/training_state"
echo "== 2. resume: steps 3-4"
$TRAIN --resume=true --config_path="$OUT/checkpoints/000002/pretrained_model/train_config.json" --steps=4
echo "RESUME_EXIT=$?"
