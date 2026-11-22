#!/bin/bash
# flux3 on ROCm with upstream PR #5213 (plain PyTorch neighborhood attention, no NATTEN).
# The PR source is mounted at /pr and put first on PYTHONPATH.
set -euo pipefail
uv pip install 'torchao>0.16'
export PYTHONPATH=/pr/src
python3 - <<'PY'
import torch, torchvision
assert torch.version.hip, "ROCm torch was replaced"
assert "+rocm" in torchvision.__version__, torchvision.__version__
import lerobot
from lerobot.utils.import_utils import _natten_available
print("lerobot from", lerobot.__file__, "natten available", _natten_available)
assert lerobot.__file__.startswith("/pr/src"), lerobot.__file__
assert not _natten_available
PY
accelerate launch --num_processes=2 "$(which lerobot-train)" \
  --policy.path=black-forest-labs/flux-3-action-so101 \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --policy.attn_mode=torch \
  --policy.gradient_checkpointing=true \
  --peft.method_type=LORA \
  --peft.r=8 \
  --peft.lora_alpha=8 \
  --accelerator.mixed_precision=bf16 \
  --rename_map='{"observation.images.top":"observation.images.scene"}' \
  --dataset.repo_id=lerobot/svla_so100_pickplace \
  --dataset.episodes='[0]' \
  --batch_size=1 \
  --steps=2 \
  --env_eval_freq=0 \
  --save_freq=2 \
  --log_freq=1 \
  --num_workers=0 \
  --trackio.enable=false \
  --output_dir=/outputs/rocm-train/flux3-pr5213-ddp2
