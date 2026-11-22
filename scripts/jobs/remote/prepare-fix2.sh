#!/bin/bash
set -euo pipefail
python - <<'PY'
import dataclasses
import json
import shutil
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download, snapshot_download
from safetensors.torch import load_file, save_file

from lerobot.policies.molmoact2.configuration_molmoact2 import MolmoAct2Config

root = Path("/outputs/rocm-train")

# --- lingbot ---
ling = root / "lingbot-compat"
if ling.exists():
    shutil.rmtree(ling)
ling.mkdir(parents=True)
snapshot_download("lerobot/lingbot_vla_v2_base", local_dir=str(ling))
weight = ling / "model.safetensors"
if not weight.is_file():
    hf_hub_download("lerobot/lingbot_vla_v2_base", "model.safetensors", local_dir=str(ling))
cfg = json.loads((ling / "config.json").read_text())
cfg["align_params"] = {}
(ling / "config.json").write_text(json.dumps(cfg))
print("lingbot bytes", weight.stat().st_size, "align", cfg["align_params"])

# --- molmo ---
molmo = root / "molmo-compat"
if molmo.exists():
    shutil.rmtree(molmo)
molmo.mkdir(parents=True)
snapshot_download("lerobot/MolmoAct2-SO100_101-LeRobot", local_dir=str(molmo))
mweight = molmo / "model.safetensors"
if not mweight.is_file():
    hf_hub_download("lerobot/MolmoAct2-SO100_101-LeRobot", "model.safetensors", local_dir=str(molmo))
mcfg = json.loads((molmo / "config.json").read_text())
allowed = {field.name for field in dataclasses.fields(MolmoAct2Config)} | {"type"}
extra = sorted(set(mcfg) - allowed)
for key in extra:
    mcfg.pop(key)
mcfg["train_mode_vlm"] = "fft"
(molmo / "config.json").write_text(json.dumps(mcfg))
print("molmo bytes", mweight.stat().st_size, "stripped", extra, "train_mode", mcfg["train_mode_vlm"])

# --- lawam fp32 cast ---
lawam = root / "lawam-fp32-compat"
if lawam.exists():
    shutil.rmtree(lawam)
lawam.mkdir(parents=True)
src = Path(snapshot_download("jialei02/lawam-pretrain-lerobot"))
for item in src.iterdir():
    dest = lawam / item.name
    if item.is_dir():
        shutil.copytree(item, dest, dirs_exist_ok=True)
    elif item.name != "model.safetensors":
        shutil.copy2(item, dest)
src_weight = src / "model.safetensors"
if not src_weight.is_file():
    src_weight = Path(hf_hub_download("jialei02/lawam-pretrain-lerobot", "model.safetensors"))
state = load_file(str(src_weight))
dtypes = {}
out = {}
for key, value in state.items():
    if value.is_floating_point() and value.dtype != torch.float32:
        dtypes[str(value.dtype)] = dtypes.get(str(value.dtype), 0) + 1
        out[key] = value.float().contiguous()
    else:
        out[key] = value
save_file(out, str(lawam / "model.safetensors"))
print("lawam cast", dtypes, "out_bytes", (lawam / "model.safetensors").stat().st_size)
PY
