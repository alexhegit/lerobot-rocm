#!/bin/bash
# RA-BC end to end on ROCm: label one episode with a zero-shot reward model (local parquet only),
# then train SmolVLA for 2 steps with --sample_weighting.type=rabc. Usage: cover-rabc.sh robometer|topreward
set -euo pipefail
model="$1"
out=/outputs/rocm-train/rabc
mkdir -p "$out"
progress="$out/${model}_progress.parquet"
python3 -c "import torch; assert torch.version.hip, 'ROCm torch missing'; print('hip', torch.version.hip)"

if [[ -f "$progress" ]]; then
  echo "reuse $progress"
elif [[ "$model" == robometer ]]; then
  python3 -m lerobot.rewards.robometer.compute_rabc_weights \
    --dataset-repo-id lerobot/svla_so100_pickplace \
    --reward-model-path lerobot/Robometer-4B \
    --image-key observation.images.top \
    --episodes 0 --batch-size 16 --device cuda \
    --output-path "$progress"
else
  # The prefix sweep starts at a 1-frame video; Qwen3-VL needs at least temporal_patch_size (2) frames.
  python3 - "$progress" <<'PY'
import sys

from lerobot.lerobot_types import TransitionKey
from lerobot.rewards.topreward import compute_rabc_weights as crw
from lerobot.rewards.topreward.processor_topreward import TOPRewardEncoderProcessorStep

_orig_call = TOPRewardEncoderProcessorStep.__call__


def _call(self, transition):
    obs = transition[TransitionKey.OBSERVATION]
    frames = obs[self.image_key]
    if frames.ndim == 5 and frames.shape[1] == 1:
        obs = dict(obs)
        obs[self.image_key] = frames.repeat(1, 2, 1, 1, 1)
        transition = {**transition, TransitionKey.OBSERVATION: obs}
    return _orig_call(self, transition)


TOPRewardEncoderProcessorStep.__call__ = _call
print("PATCH topreward 1-frame prefix -> 2 frames")
sys.argv = [
    "compute_rabc_weights",
    "--dataset-repo-id", "lerobot/svla_so100_pickplace",
    "--vlm-name", "Qwen/Qwen3-VL-8B-Instruct",
    "--episodes", "0",
    "--num-samples", "15",
    "--device", "cuda",
    "--output-path", sys.argv[1],
]
crw.main()
PY
fi

python3 - "$progress" <<'PY'
import sys
import pandas as pd
df = pd.read_parquet(sys.argv[1])
p = df["progress_sparse"]
print("PROGRESS rows", len(df), "columns", list(df.columns))
print("PROGRESS min %.4f max %.4f first %.4f last %.4f nan %d" % (p.min(), p.max(), p.iloc[0], p.iloc[-1], p.isna().sum()))
PY

lerobot-train \
  --policy.path=lerobot/smolvla_base \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --dataset.repo_id=lerobot/svla_so100_pickplace \
  --dataset.episodes='[0]' \
  --rename_map='{"observation.images.top": "observation.images.camera1", "observation.images.wrist": "observation.images.camera2"}' \
  --sample_weighting.type=rabc \
  --sample_weighting.progress_path="$progress" \
  --sample_weighting.head_mode=sparse \
  --batch_size=4 --steps=2 --env_eval_freq=0 --save_freq=2 --log_freq=1 \
  --num_workers=0 --trackio.enable=false \
  --output_dir="/outputs/rocm-train/rabc-smolvla-${model}"
