#!/bin/bash
set -euo pipefail
python - <<'PY'
from lerobot.datasets import LeRobotDataset, write_stats
from lerobot.scripts.augment_dataset_quantile_stats import compute_quantile_stats_for_dataset

ds = LeRobotDataset("lerobot/svla_so100_pickplace", download_videos=False)
new_stats = compute_quantile_stats_for_dataset(ds, use_sampling=True, skip_images=True)
if ds.meta.stats:
    for key, feature_stats in ds.meta.stats.items():
        new_stats.setdefault(key, feature_stats)
write_stats(new_stats, ds.meta.root)
state = new_stats["observation.state"]
action = new_stats["action"]
print("quantile root", ds.meta.root)
print("state", sorted(state), "q01", len(state["q01"]), "q99", len(state["q99"]))
print("action", sorted(action), "q01", len(action["q01"]), "q99", len(action["q99"]))
PY
