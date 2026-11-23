#!/bin/bash
# GR00T N1.7 LIBERO-Spatial checkpoint on ROCm, as documented in docs/source/groot.mdx (reference 95%):
# libero_spatial, 10 tasks x 5 episodes, OSMesa rendering.
set -uo pipefail
apt-get update -qq && apt-get install -y -qq libosmesa6 libglu1-mesa >/dev/null
export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa
OUT=/outputs/rocm-train/eval-groot-libero
rm -rf "$OUT"
yes N | lerobot-eval \
  --policy.path=nvidia/gr00t17-lerobot-libero_spatial-640 \
  --policy.embodiment_tag=libero_sim \
  --policy.device=cuda \
  --env.type=libero \
  --env.task=libero_spatial \
  --rename_map='{"observation.images.image2": "observation.images.wrist_image"}' \
  --eval.n_episodes=5 \
  --eval.batch_size=5 \
  --output_dir="$OUT"
python3 -c "import json; d=json.load(open('$OUT/eval_info.json'))['overall']; print('EVAL', d['pc_success'], d['n_success'], d['n_episodes'], d['pc_success_ci95'])"
