#!/bin/bash
# Official lerobot/diffusion_pusht on ROCm: 50 PushT episodes (pymunk rendering, no GL needed).
set -euo pipefail
# The image's migrate_policy_normalization fails to save this config ("Couldn't encode 84"); use latest main.
export PYTHONPATH=/outputs/rocm-train/latest-main/src
export MIOPEN_USER_DB_PATH=/outputs/rocm-train/miopen/fast/db
export MIOPEN_CUSTOM_CACHE_DIR=/outputs/rocm-train/miopen/fast/cache
OUT=/outputs/rocm-train/eval-diffusion-pusht
MIGRATED=/outputs/rocm-train/diffusion-pusht-migrated
rm -rf "$OUT" "$MIGRATED"
POLICY=lerobot/diffusion_pusht
if ! python3 -c "from huggingface_hub import hf_hub_download; hf_hub_download('$POLICY', 'policy_preprocessor.json')" 2>/dev/null; then
  python3 -m lerobot.processor.migrate_policy_normalization --pretrained-path "$POLICY" --output-dir "$MIGRATED"
  POLICY=$MIGRATED
fi
lerobot-eval --policy.path="$POLICY" --policy.device=cuda --env.type=pusht \
  --eval.n_episodes=50 --eval.batch_size=10 --output_dir="$OUT"
python3 -c "import json; d=json.load(open('$OUT/eval_info.json'))['overall']; print('EVAL', d['pc_success'], d['n_success'], d['n_episodes'], d['pc_success_ci95'], d['avg_max_reward'])"
