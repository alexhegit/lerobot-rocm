#!/bin/bash
set -euo pipefail
python3 - <<'PY'
from lerobot.datasets import LeRobotDataset, write_stats
from lerobot.scripts.augment_dataset_quantile_stats import compute_quantile_stats_for_dataset

root = "/outputs/rocm-train/fineart-aloha3"
ds = LeRobotDataset(repo_id="local/fineart_aloha3", root=root, download_videos=False)
new_stats = compute_quantile_stats_for_dataset(ds, use_sampling=True, skip_images=True)
if ds.meta.stats:
    for key, feature_stats in ds.meta.stats.items():
        new_stats.setdefault(key, feature_stats)
write_stats(new_stats, ds.meta.root)
assert "q01" in new_stats["action"] and "q99" in new_stats["action"]
print("quantile_stats", new_stats["action"]["q01"].shape)
PY
lerobot-train \
  --policy.path=lerobot/fineart_vla_base \
  --policy.dtype=bfloat16 \
  --policy.gradient_checkpointing=true \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --dataset.repo_id=local/fineart_aloha3 \
  --dataset.root=/outputs/rocm-train/fineart-aloha3 \
  --dataset.episodes='[0]' \
  --batch_size=1 --steps=2 --env_eval_freq=0 --save_freq=2 --log_freq=1 \
  --num_workers=0 --trackio.enable=false \
  --output_dir=/outputs/rocm-train/cover-fineart3b
