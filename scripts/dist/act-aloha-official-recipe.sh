#!/bin/bash
# ACT with the recipe of lerobot/act_aloha_sim_transfer_cube_human (1 GPU, batch 8, 100k steps), then
# 50 sim episodes, for a like-for-like comparison with the official checkpoint evaluated on ROCm.
set -euo pipefail
STEPS=${STEPS:-100000}
export MIOPEN_USER_DB_PATH=/outputs/rocm-train/miopen/fast/db
export MIOPEN_CUSTOM_CACHE_DIR=/outputs/rocm-train/miopen/fast/cache
mkdir -p "$MIOPEN_USER_DB_PATH" "$MIOPEN_CUSTOM_CACHE_DIR"
OUT=/outputs/rocm-train/act-aloha-recipe
rm -rf "$OUT" "$OUT-eval"
lerobot-train \
  --policy.type=act \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --dataset.repo_id=lerobot/aloha_sim_transfer_cube_human \
  --batch_size=8 \
  --steps="$STEPS" \
  --save_freq="$STEPS" \
  --log_freq=1000 \
  --env_eval_freq=0 \
  --num_workers=8 \
  --trackio.enable=false \
  --output_dir="$OUT"
mkdir -p /outputs/rocm-train/act-aloha-recipe-trained
cp -r "$OUT/checkpoints/last/pretrained_model" /outputs/rocm-train/act-aloha-recipe-trained/
apt-get update -qq && apt-get install -y -qq libosmesa6 libglu1-mesa >/dev/null
export MUJOCO_GL=osmesa
lerobot-eval \
  --policy.path=/outputs/rocm-train/act-aloha-recipe-trained/pretrained_model \
  --policy.device=cuda \
  --env.type=aloha \
  --env.task=AlohaTransferCube-v0 \
  --eval.n_episodes=50 \
  --eval.batch_size=10 \
  --output_dir="$OUT-eval"
python3 -c "import json; d=json.load(open('$OUT-eval/eval_info.json'))['overall']; print('EVAL', d['pc_success'], d['n_success'], d['n_episodes'], d['pc_success_ci95'])"
