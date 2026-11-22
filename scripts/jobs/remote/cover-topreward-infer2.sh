#!/bin/bash
set -euo pipefail
python3 - <<'PY'
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.lerobot_types import TransitionKey
from lerobot.rewards import make_reward_model, make_reward_model_config, make_reward_pre_post_processors

ds = LeRobotDataset("lerobot/svla_so100_pickplace", episodes=[0])
frames = torch.stack([ds[i]["observation.images.top"] for i in range(4)])
task = ds[0].get("task", "pick")
cfg = make_reward_model_config(
    "topreward",
    device="cuda",
    image_key="observation.images.top",
    vlm_name="Qwen/Qwen3-VL-8B-Instruct",
)
assert torch.version.hip
model = make_reward_model(cfg)
pre, _post = make_reward_pre_post_processors(cfg)
step = next(s for s in pre.steps if type(s).__name__.startswith("TOPReward"))
transition = {
    TransitionKey.OBSERVATION: {"observation.images.top": frames.unsqueeze(0)},
    TransitionKey.COMPLEMENTARY_DATA: {"task": task},
}
encoded = step(transition)[TransitionKey.OBSERVATION]
reward = model.compute_reward(encoded)
print("trainable", model.is_trainable)
print("TOPREWARD_REWARD", reward.detach().float().cpu().tolist())
PY
