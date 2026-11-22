#!/bin/bash
set -euo pipefail
python3 - <<'PY'
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.rewards import make_reward_model, make_reward_model_config, make_reward_pre_post_processors

ds = LeRobotDataset("lerobot/svla_so100_pickplace", episodes=[0])
item = ds[0]
cfg = make_reward_model_config(
    "topreward",
    device="cuda",
    image_key="observation.images.top",
    vlm_name="Qwen/Qwen3-VL-8B-Instruct",
)
assert torch.version.hip
model = make_reward_model(cfg)
print("trainable", model.is_trainable)
pre, _post = make_reward_pre_post_processors(cfg)
batch = pre(item)
reward = model.compute_reward(batch)
print("TOPREWARD_REWARD", reward.detach().float().cpu().tolist())
PY
