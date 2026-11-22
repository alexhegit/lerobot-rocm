#!/bin/bash
set -euo pipefail
uv pip install 'torchao>0.16'
python3 - <<'PY'
import torch, torchvision, torchao
assert torch.version.hip, "ROCm torch was replaced"
assert "+rocm" in torchvision.__version__, torchvision.__version__
print("torchao", torchao.__version__, torch.__version__)
PY
accelerate launch --num_processes=2 /outputs/rocm-train/jobs/fsdp_autowrap_train.py --policy.path=lerobot/lingbot_va_libero_long --policy.attn_mode=flex \
  --peft.method_type=LORA --peft.r=8 --peft.lora_alpha=8 \
  --peft.target_modules='transformer\.blocks\.\d+\.attn[12]\.(to_q|to_v)' \
  --dataset.repo_id=lerobot/libero --policy.device=cuda --policy.push_to_hub=false --batch_size=1 --steps=2 --env_eval_freq=0 --save_freq=2 --log_freq=1 --num_workers=0 --trackio.enable=false --dataset.episodes='[0]' --output_dir=/outputs/rocm-train/cover-lingbotva-fsdp2 --parallelism.dp_shard=2
