#!/bin/bash
set -euo pipefail
python - <<'PY'
import json
from pathlib import Path
from huggingface_hub import snapshot_download

dest = Path("/outputs/rocm-train/eo1-base-compat")
dest.mkdir(parents=True, exist_ok=True)
snapshot_download("lerobot/eo1-base", local_dir=str(dest))
cfg_path = dest / "config.json"
cfg = json.loads(cfg_path.read_text())
cfg.pop("use_language_recipe", None)
cfg_path.write_text(json.dumps(cfg))
print("eo1 config patched")
PY
