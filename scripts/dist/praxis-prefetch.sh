#!/bin/bash
# Prefetch Praxis-1 (no GPU): the converted DROID checkpoint (54 GB), the Wan2.2 VAE, the Qwen3.5-2B
# text encoder, and a short DROID episode list from the PR #11 revision with the Cartesian columns.
set -euo pipefail
export PYTHONPATH=/outputs/rocm-train/latest-main/src
python3 - <<'PY'
from huggingface_hub import hf_hub_download, snapshot_download
print(snapshot_download("runwayml/praxis-1-droid-lerobot"))
print(hf_hub_download("Wan-AI/Wan2.2-TI2V-5B", "Wan2.2_VAE.pth"))
print(snapshot_download("Qwen/Qwen3.5-2B"))
PY
python3 /outputs/rocm-train/latest-main/examples/praxis/select_droid_episodes.py \
  --repo-id lerobot/droid_1.0.1 --revision refs/pr/11 --out /outputs/rocm-train/praxis/droid_episodes.json
python3 -c "import json; e=json.load(open('/outputs/rocm-train/praxis/droid_episodes.json')); print('episodes', len(e), e[:4])"
echo PREFETCH_OK
