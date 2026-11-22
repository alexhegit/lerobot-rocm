#!/bin/bash
set -euo pipefail
python3 - <<'PY'
import dataclasses, json
from pathlib import Path
from lerobot.policies.molmoact2.configuration_molmoact2 import MolmoAct2Config
p = Path("/outputs/rocm-train/molmo-lora-compat/config.json")
cfg = json.loads(p.read_text())
allowed = {f.name for f in dataclasses.fields(MolmoAct2Config)} | {"type"}
for key in sorted(set(cfg) - allowed):
    cfg.pop(key)
cfg["train_mode_vlm"] = "lora"
p.write_text(json.dumps(cfg))
print("molmo stripped for nonstrict")
PY
uv pip install 'torchao>0.16'
python3 - <<'PY'
import torch, torchvision, torchao
assert torch.version.hip
assert "+rocm" in torchvision.__version__
print("torchao", torchao.__version__, "hip", torch.version.hip)
PY
python3 /outputs/rocm-train/jobs/molmo_lora_nonstrict.py \
  --policy.path=/outputs/rocm-train/molmo-lora-compat \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --policy.dtype=bfloat16 \
  --policy.train_mode_vlm=lora \
  --policy.gradient_checkpointing=true \
  --rename_map='{"observation.images.top":"observation.images.cam0","observation.images.wrist":"observation.images.cam1"}' \
  --dataset.repo_id=lerobot/svla_so100_pickplace \
  --dataset.episodes='[0]' \
  --batch_size=1 \
  --steps=2 \
  --env_eval_freq=0 \
  --save_freq=2 \
  --log_freq=1 \
  --num_workers=0 \
  --trackio.enable=false \
  --output_dir=/outputs/rocm-train/molmo-lora-nonstrict
