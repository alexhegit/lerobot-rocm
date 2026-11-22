#!/bin/bash
# Rebuild /outputs/rocm-train/molmo-compat: the MolmoAct2 SO100 checkpoint with config fields the
# image's MolmoAct2Config does not know removed, and train_mode_vlm=fft.
set -euo pipefail
python3 - <<'PY'
import dataclasses
import json
import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download

from lerobot.policies.molmoact2.configuration_molmoact2 import MolmoAct2Config

molmo = Path("/outputs/rocm-train/molmo-compat")
if (molmo / "config.json").is_file() and (molmo / "model.safetensors").exists():
    print("molmo-compat exists")
    raise SystemExit
if molmo.exists():
    shutil.rmtree(molmo)
molmo.mkdir(parents=True)
src = Path(snapshot_download("lerobot/MolmoAct2-SO100_101-LeRobot"))
for item in src.iterdir():
    dest = molmo / item.name
    if item.name == "model.safetensors":
        dest.symlink_to(item.resolve())
    elif item.is_dir():
        shutil.copytree(item, dest, symlinks=False)
    else:
        shutil.copy2(item.resolve(), dest)
if not (molmo / "model.safetensors").exists():
    (molmo / "model.safetensors").symlink_to(
        Path(hf_hub_download("lerobot/MolmoAct2-SO100_101-LeRobot", "model.safetensors")).resolve()
    )
cfg = json.loads((molmo / "config.json").read_text())
allowed = {f.name for f in dataclasses.fields(MolmoAct2Config)} | {"type"}
extra = sorted(set(cfg) - allowed)
for key in extra:
    cfg.pop(key)
cfg["train_mode_vlm"] = "fft"
(molmo / "config.json").write_text(json.dumps(cfg))
print("molmo-compat rebuilt, stripped", extra)
PY
