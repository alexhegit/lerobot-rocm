#!/bin/bash
# Train ACT from scratch on all 50 episodes of aloha_sim_transfer_cube_human with DDP on 2 GPUs
# (batch 32 per GPU, 64 in total; 2 GPUs scaled best for ACT on this node), then evaluate 50
# episodes of AlohaTransferCube-v0 on 1 GPU. STEPS defaults to 12500 = 100k steps x batch 8 / 64.
set -euo pipefail
STEPS=${STEPS:-12500}
# Persistent MIOpen kernel cache: the first conv step otherwise spends minutes compiling kernels.
export MIOPEN_USER_DB_PATH=/outputs/rocm-train/miopen/fast/db
export MIOPEN_CUSTOM_CACHE_DIR=/outputs/rocm-train/miopen/fast/cache
mkdir -p "$MIOPEN_USER_DB_PATH" "$MIOPEN_CUSTOM_CACHE_DIR"
OUT=/outputs/rocm-train/act-aloha-train
rm -rf "$OUT"
accelerate launch --num_processes=2 "$(which lerobot-train)" \
  --policy.type=act \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --dataset.repo_id=lerobot/aloha_sim_transfer_cube_human \
  --batch_size=32 \
  --steps="$STEPS" \
  --save_freq="$STEPS" \
  --log_freq=250 \
  --env_eval_freq=0 \
  --num_workers=8 \
  --trackio.enable=false \
  --output_dir="$OUT"
mkdir -p /outputs/rocm-train/act-aloha-trained
cp -r "$OUT/checkpoints/last/pretrained_model" /outputs/rocm-train/act-aloha-trained/
apt-get update -qq && apt-get install -y -qq libosmesa6 libglu1-mesa >/dev/null
export MUJOCO_GL=osmesa
HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES%%,*} lerobot-eval \
  --policy.path=/outputs/rocm-train/act-aloha-trained/pretrained_model \
  --policy.device=cuda \
  --env.type=aloha \
  --env.task=AlohaTransferCube-v0 \
  --eval.n_episodes=50 \
  --eval.batch_size=10 \
  --output_dir="$OUT-eval"
find "$OUT-eval" -name "eval_info.json" -exec python3 -c "import json,sys; d=json.load(open(sys.argv[1])); print('EVAL', json.dumps(d.get('overall', d.get('aggregated', {}))))" {} \;
