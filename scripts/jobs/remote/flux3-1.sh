#!/bin/bash
set -euo pipefail
python3 - <<'PY'
import torch, torchvision
assert torch.version.hip
assert "+rocm" in torchvision.__version__
try:
    from flash_attn_interface import flash_attn_func
    print("IMPORT_OK", flash_attn_func)
except Exception as exc:
    print("IMPORT_FAIL", type(exc).__name__, exc)
PY
if ! grep -q IMPORT_OK <(python3 -c 'from flash_attn_interface import flash_attn_func; print("IMPORT_OK")' 2>/dev/null || true); then
  set +e
  uv pip install 'flash-attn' 
  install_ec=$?
  set -e
  python3 - <<'PY'
import torch, torchvision
assert torch.version.hip, "ROCm torch was replaced"
assert "+rocm" in torchvision.__version__, torchvision.__version__
try:
    from flash_attn_interface import flash_attn_func
    print("IMPORT_OK", torch.__version__)
except Exception as exc:
    print("IMPORT_STILL_FAIL", type(exc).__name__, exc)
    raise SystemExit(3)
PY
fi
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
  --output_dir=/outputs/rocm-train/flux3-1
