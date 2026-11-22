#!/bin/bash
set -u
ROOT=/data/lerobot/outputs/rocm-train
LOG="$ROOT/run-coverage.log"
RESULTS="$ROOT/results.txt"
exec >>"$LOG" 2>&1
echo "coverage-finish2 begin $(date -Is)"

run_docker() {
  local name="$1"
  local script="$2"
  local host_out="$ROOT/$name"
  echo "----- $(date -Is) BEGIN ${name} devices=0 -----"
  echo "disk_free_gb=$(df -BG / | awk 'NR==2 {print $4}')"
  rm -rf "$host_out"
  local started ec elapsed
  started=$(date +%s)
  timeout --signal=TERM 5400 docker run --rm --name "rocm-${name}" \
    --ipc=host --shm-size 16g \
    --device=/dev/kfd --device=/dev/dri \
    --group-add 44 --group-add 992 \
    -e HIP_VISIBLE_DEVICES=0 \
    -e MIOPEN_FIND_MODE=2 \
    -v /data/lerobot/outputs:/outputs \
    -v /root/.cache/huggingface:/root/.cache/huggingface \
    lerobot-rocm \
    bash "/outputs/rocm-train/jobs/${script}"
  ec=$?
  elapsed=$(( $(date +%s) - started ))
  if [[ "$ec" -eq 124 ]]; then
    echo "RESULT ${name} TIMEOUT exit=${ec} seconds=${elapsed}"
    docker rm -f "rocm-${name}" >/dev/null 2>&1 || true
  elif [[ "$ec" -eq 0 ]]; then
    echo "RESULT ${name} PASS exit=${ec} seconds=${elapsed}"
  else
    echo "RESULT ${name} FAIL exit=${ec} seconds=${elapsed}"
  fi
  echo "${name} $([[ $ec -eq 0 ]] && echo PASS || echo FAIL) exit=${ec} seconds=${elapsed}" >>"$RESULTS"
  echo "----- $(date -Is) END ${name} -----"
  find "$host_out" -type f \( -name '*.safetensors' -o -name '*.bin' -o -name '*.pt' \) -delete 2>/dev/null || true
}

cat >"$ROOT/jobs/cover-topreward-infer2.sh" <<'EOF'
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
EOF
chmod +x "$ROOT/jobs/cover-topreward-infer2.sh" "$ROOT/jobs/cover-fineart3.sh"
run_docker cover-fineart3 cover-fineart3.sh || true
run_docker cover-topreward-infer2 cover-topreward-infer2.sh || true
echo "coverage-finish2 end $(date -Is)"
