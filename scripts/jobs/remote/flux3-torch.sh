#!/bin/bash
set -euo pipefail
python3 - <<'PY'
import torch, torchvision
assert torch.version.hip
assert "+rocm" in torchvision.__version__
print("before", torch.__version__, torch.version.hip, torchvision.__version__)
try:
    from flash_attn_interface import flash_attn_func
    print("IMPORT_OK", flash_attn_func)
except Exception as exc:
    print("IMPORT_FAIL", type(exc).__name__, exc)
print("attn_mode torch")
PY
uv pip install 'natten==0.21.7' 'torchao>0.16'
python3 - <<'PY'
import torch, torchvision
assert torch.version.hip, "ROCm torch was replaced"
assert "+rocm" in torchvision.__version__, torchvision.__version__
import natten, torchao
print(
    "after",
    torch.__version__,
    torch.version.hip,
    torchvision.__version__,
    "natten",
    natten.__version__,
    "torchao",
    torchao.__version__,
)
PY
lerobot-train \
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
  --output_dir=/outputs/rocm-train/flux3-torch
