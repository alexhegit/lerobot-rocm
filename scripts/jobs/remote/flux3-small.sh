#!/bin/bash
set -euo pipefail
uv pip install 'natten==0.21.7' 'torchao>0.16'
python3 - <<'PY'
import torch, torchvision, natten, torchao
assert torch.version.hip, "ROCm torch was replaced"
assert "+rocm" in torchvision.__version__, torchvision.__version__
print("ok", torch.__version__, "natten", natten.__version__, "torchao", torchao.__version__)
PY
export F3_NATTEN_BACKEND=flex-fna
lerobot-train \
  --policy.path=black-forest-labs/flux-3-action-so101 \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --policy.attn_mode=torch \
  --policy.gradient_checkpointing=true \
  --policy.canvas_hw='[64,128]' \
  --policy.n_obs_steps=2 \
  --policy.history_snapshots=1 \
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
  --output_dir=/outputs/rocm-train/flux3-small
