#!/bin/bash
set -euo pipefail
python3 - <<'PY'
import dataclasses, json
from pathlib import Path
from lerobot.policies.molmoact2.configuration_molmoact2 import MolmoAct2Config
from lerobot.policies.molmoact2.processor_molmoact2 import make_molmoact2_pre_post_processors
root = Path("/outputs/rocm-train/molmo-lora-compat")
cfg_json = json.loads((root / "config.json").read_text())
allowed = {field.name for field in dataclasses.fields(MolmoAct2Config)} | {"type"}
for key in sorted(set(cfg_json) - allowed):
    cfg_json.pop(key)
cfg_json["train_mode_vlm"] = "lora"
(root / "config.json").write_text(json.dumps(cfg_json))
cfg = MolmoAct2Config.from_pretrained(root)
cfg.device = "cpu"
_, post = make_molmoact2_pre_post_processors(cfg)
post.save_pretrained(root)
print("postprocessor", (root / "policy_postprocessor.json").is_file(), flush=True)
PY
uv pip install 'torchao>0.16'
python3 - <<'PY'
import torch, torchvision
assert torch.version.hip, "ROCm torch was replaced"
assert "+rocm" in torchvision.__version__, torchvision.__version__
print("hip", torch.version.hip, flush=True)
PY
exec "$@"
