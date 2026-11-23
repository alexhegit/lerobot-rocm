#!/bin/bash
# Praxis-1 (added upstream, not in the image) LoRA fine-tune smoke on DROID (PR #11 revision with the
# Cartesian columns), two episodes of 93+ frames. Runs from the latest-main source tree.
set -euo pipefail
export PYTHONPATH=/outputs/rocm-train/latest-main/src
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
uv pip install 'torchao>0.16'
python3 -c "import torch, torchvision; assert torch.version.hip and '+rocm' in torchvision.__version__; import lerobot; print('lerobot from', lerobot.__file__)"
EPISODES=$(python3 -c "import json; print(json.load(open('/outputs/rocm-train/praxis/droid_episodes.json'))[:2])")
# The docs say a renamed key replaces an existing one, but the code refuses the collision: move the
# joint-space state/action out of the way as well.
RENAME='{"observation.state":"observation.state.joint_unused","action":"action.joint_unused","observation.state.cartesian":"observation.state","action.cartesian":"action","observation.images.exterior_1_left":"observation.images.exterior","observation.images.wrist_left":"observation.images.wrist"}'
accelerate launch --num_processes=2 "$(which lerobot-train)" \
  --policy.path=runwayml/praxis-1-droid-lerobot \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --peft.r=16 \
  --peft.lora_alpha=32 \
  --dataset.repo_id=lerobot/droid_1.0.1 \
  --dataset.revision=refs/pr/11 \
  --dataset.episodes="$EPISODES" \
  --rename_map="$RENAME" \
  --batch_size=1 \
  --steps=2 \
  --env_eval_freq=0 \
  --save_freq=2 \
  --log_freq=1 \
  --num_workers=2 \
  --trackio.enable=false \
  --output_dir=/outputs/rocm-train/praxis-lora-ddp2
