#!/bin/bash
set -euo pipefail
python - <<'PY'
import dataclasses
import json
from pathlib import Path

from huggingface_hub import snapshot_download
from lerobot.policies.molmoact2.configuration_molmoact2 import MolmoAct2Config

ling = Path("/outputs/rocm-train/lingbot-compat")
ling.mkdir(parents=True, exist_ok=True)
snapshot_download("lerobot/lingbot_vla_v2_base", local_dir=str(ling))
ling_cfg_path = ling / "config.json"
ling_cfg = json.loads(ling_cfg_path.read_text())
ling_cfg["align_params"] = {}
ling_cfg_path.write_text(json.dumps(ling_cfg))
print("lingbot align_params cleared")

molmo = Path("/outputs/rocm-train/molmo-compat")
molmo.mkdir(parents=True, exist_ok=True)
snapshot_download("lerobot/MolmoAct2-SO100_101-LeRobot", local_dir=str(molmo))
molmo_cfg_path = molmo / "config.json"
molmo_cfg = json.loads(molmo_cfg_path.read_text())
allowed = {field.name for field in dataclasses.fields(MolmoAct2Config)} | {"type"}
extra = sorted(set(molmo_cfg) - allowed)
for key in extra:
    molmo_cfg.pop(key)
molmo_cfg_path.write_text(json.dumps(molmo_cfg))
print("molmo stripped", extra)
print("molmo image_keys", molmo_cfg.get("image_keys"))
PY
