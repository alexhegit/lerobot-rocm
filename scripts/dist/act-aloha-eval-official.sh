#!/bin/bash
# Evaluate the official lerobot/act_aloha_sim_transfer_cube_human checkpoint on ROCm (1 GPU), 50 episodes
# of AlohaTransferCube-v0, rendered with OSMesa (MI300X has no graphics context for EGL).
set -euo pipefail
apt-get update -qq && apt-get install -y -qq libosmesa6 libglu1-mesa >/dev/null
export MUJOCO_GL=osmesa
OUT=/outputs/rocm-train/act-aloha-eval-official
MIGRATED=/outputs/rocm-train/act-official-migrated
rm -rf "$OUT" "$MIGRATED"
# The Hub checkpoint predates the processor format; migrate it locally (no --push-to-hub).
python3 -m lerobot.processor.migrate_policy_normalization \
  --pretrained-path lerobot/act_aloha_sim_transfer_cube_human --output-dir "$MIGRATED"
lerobot-eval \
  --policy.path="$MIGRATED" \
  --policy.device=cuda \
  --env.type=aloha \
  --env.task=AlohaTransferCube-v0 \
  --eval.n_episodes=50 \
  --eval.batch_size=10 \
  --output_dir="$OUT"
find "$OUT" -name "eval_info.json" -exec python3 -c "import json,sys; d=json.load(open(sys.argv[1])); print('EVAL', json.dumps(d.get('overall', d.get('aggregated', {}))))" {} \;
