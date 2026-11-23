#!/bin/bash
# Official lerobot/smolvla_libero (revision pinned as in docs/source/simulation.mdx) on ROCm:
# LIBERO-Spatial, all 10 tasks x 5 episodes, OSMesa rendering.
set -euo pipefail
apt-get update -qq && apt-get install -y -qq libosmesa6 libglu1-mesa >/dev/null
export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa
OUT=/outputs/rocm-train/eval-smolvla-libero
rm -rf "$OUT"
# LIBERO asks once whether to set a custom dataset folder; answer no.
yes N | lerobot-eval \
  --policy.path=lerobot/smolvla_libero \
  --policy.pretrained_revision=a24f1f380461b665158d52acd140152c79e8b36d \
  --policy.device=cuda \
  --env.type=libero --env.task=libero_spatial \
  --rename_map='{"observation.images.image": "observation.images.camera1", "observation.images.image2": "observation.images.camera2"}' \
  --eval.n_episodes=5 --eval.batch_size=5 \
  --output_dir="$OUT"
python3 -c "import json; d=json.load(open('$OUT/eval_info.json'))['overall']; print('EVAL', d['pc_success'], d['n_success'], d['n_episodes'], d['pc_success_ci95'])"
