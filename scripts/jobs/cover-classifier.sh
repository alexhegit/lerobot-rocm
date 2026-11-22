#!/bin/bash
# reward_classifier smoke: pusht frames resized to 128x128 (the CNN head expects a 4x4 feature map).
set -euo pipefail
python3 - <<'PY'
import torch
import torch.nn.functional as F
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.datasets import LeRobotDataset
from lerobot.rewards import make_reward_model, make_reward_model_config, make_reward_pre_post_processors
from lerobot.rewards.classifier.modeling_classifier import Classifier
from lerobot.utils.constants import REWARD

assert torch.cuda.is_available() and torch.version.hip
img_key = "observation.image"
ds = LeRobotDataset("lerobot/pusht", episodes=[0])
n = len(ds)
idx = torch.linspace(0, n - 1, 8).long().tolist()
items = [ds[i] for i in idx]
images = torch.stack([it[img_key] for it in items])
images = F.interpolate(images, size=(128, 128), mode="bilinear", align_corners=False)
rewards = torch.tensor([float(it[REWARD]) for it in items])
labels = (rewards > rewards.median()).float()
print("frames", idx, "rewards", [round(r, 3) for r in rewards.tolist()], "labels", labels.tolist())

cfg = make_reward_model_config(
    "reward_classifier",
    device="cuda",
    model_type="cnn",
    num_cameras=1,
    num_classes=2,
    input_features={img_key: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 128, 128))},
)
stats = {img_key: {"mean": torch.full((3, 1, 1), 0.5), "std": torch.full((3, 1, 1), 0.25)}}
model = make_reward_model(cfg).to("cuda")
pre, _post = make_reward_pre_post_processors(cfg, dataset_stats=stats)
print("model_name", cfg.model_name, "trainable", model.is_trainable,
      "params", sum(p.numel() for p in model.parameters() if p.requires_grad))

opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                        lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
model.train()
for step in range(2):
    sl = slice(step * 4, step * 4 + 4)
    batch = pre({img_key: images[sl], REWARD: labels[sl]})
    loss, out = model.forward(batch)
    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
    opt.step()
    print(f"step {step + 1} loss {loss.item():.4f} {({k: v for k, v in out.items() if k != 'loss'})}")

model.eval()
with torch.no_grad():
    r = model.compute_reward(pre({img_key: images, REWARD: labels}))
print("CLASSIFIER_REWARD", r.cpu().tolist())

out_dir = "/outputs/rocm-train/cover-classifier"
model.save_pretrained(out_dir)
reloaded = Classifier.from_pretrained(out_dir).to("cuda").eval()
with torch.no_grad():
    r2 = reloaded.compute_reward(pre({img_key: images, REWARD: labels}))
assert torch.equal(r.cpu(), r2.cpu()), (r, r2)
print("CLASSIFIER_PASS reload ok")
PY
