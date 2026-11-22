#!/bin/bash
# Lingbot retry on GPUs 2-3 while lawam FSDP uses 0-1. Real config file, weight symlink.
set -u
ROOT=/data/lerobot/outputs/rocm-train
LOG="$ROOT/run-lingbot-only.log"
RESULTS="$ROOT/results.txt"
SNAP=/root/.cache/huggingface/hub/models--lerobot--lingbot_vla_v2_base/snapshots/32d0a32a881c351d28b6c970ab4ecbc119d29e79
LINK="$ROOT/lingbot-link"
mkdir -p "$ROOT/jobs"
: >"$LOG"
exec >>"$LOG" 2>&1

rm -rf "$LINK"
mkdir -p "$LINK"
cp -L "$SNAP/config.json" "$SNAP/policy_preprocessor.json" "$SNAP/policy_postprocessor.json" "$LINK/"
cp -aL "$SNAP/tokenizer" "$SNAP/image_processor" "$SNAP/chat_template_tokenizer" "$LINK/"
ln -s "$SNAP/model.safetensors" "$LINK/model.safetensors"
python3 - <<PY
import json
from pathlib import Path
p = Path("$LINK/config.json")
cfg = json.loads(p.read_text())
cfg["align_params"] = {}
p.write_text(json.dumps(cfg))
w = Path("$LINK/model.safetensors")
print("config_bytes", p.stat().st_size, "weight_bytes", w.stat().st_size, "align", cfg["align_params"])
PY

run_job() {
  local name="$1"
  local devices="$2"
  local script="$3"
  local host_out="$ROOT/$name"
  rm -rf "$host_out"
  echo "----- $(date -Is) running ${name} devices=${devices} -----"
  echo "disk_free_gb=$(df -BG / | awk 'NR==2{gsub(/G/,"",$4); print $4}')"
  local started
  started=$(date +%s)
  timeout --signal=TERM 5400 docker run --rm --name "rocm-${name}" \
    --ipc=host --shm-size 16g \
    --device=/dev/kfd --device=/dev/dri \
    --group-add 44 --group-add 992 \
    -e HIP_VISIBLE_DEVICES="$devices" \
    -e MIOPEN_FIND_MODE=2 \
    -v /data/lerobot/outputs:/outputs \
    -v /root/.cache/huggingface:/root/.cache/huggingface \
    lerobot-rocm \
    bash "/outputs/rocm-train/jobs/${script}"
  local ec=$?
  local elapsed=$(( $(date +%s) - started ))
  local ckpt="$host_out/checkpoints/000002/pretrained_model/model.safetensors"
  local stepf="$host_out/checkpoints/000002/training_state/training_step.json"
  local has_ckpt=0
  if [[ -f "$ckpt" || -f "$stepf" ]]; then has_ckpt=1; fi
  if find "$host_out/checkpoints/000002" -name '*.distcp' 2>/dev/null | grep -q .; then has_ckpt=1; fi
  local verdict="FAIL"
  if [[ "$ec" -eq 0 && "$has_ckpt" -eq 1 ]]; then
    verdict="PASS"
  elif [[ "$ec" -eq 124 ]]; then
    verdict="TIMEOUT"
    docker rm -f "rocm-${name}" >/dev/null 2>&1 || true
  fi
  echo "RESULT ${name} ${verdict} exit=${ec} seconds=${elapsed}"
  if [[ -f "$stepf" ]]; then echo "STEP $(tr -d '\n' <"$stepf")"; fi
  echo "${name} ${verdict} exit=${ec} seconds=${elapsed}" >>"$RESULTS"
  find "$host_out" -type d -name training_state -prune -exec rm -rf {} + 2>/dev/null || true
  find "$host_out" -type f \( -name '*.safetensors' -o -name '*.distcp' \) -delete 2>/dev/null || true
  [[ "$verdict" == "PASS" ]]
}

COMMON='  --dataset.repo_id=lerobot/svla_so100_pickplace \
  --dataset.episodes='"'"'[0]'"'"' \
  --batch_size=1 \
  --steps=2 \
  --env_eval_freq=0 \
  --save_freq=2 \
  --log_freq=1 \
  --num_workers=0 \
  --trackio.enable=false'

write_one() {
  local name="$1"
  local launcher="$2"
  local extra="$3"
  cat >"$ROOT/jobs/${name}.sh" <<EOF
#!/bin/bash
set -euo pipefail
${launcher} \\
  --policy.path=/outputs/rocm-train/lingbot-link \\
  --policy.device=cuda \\
  --policy.push_to_hub=false \\
  --policy.dtype=float32 \\
  ${extra} \\
  --policy.state_slots='{"observation.state.arm.position": {"origin_keys": [{"observation.state": {"start": 0, "end": 5}}]}, "observation.state.effector.position": {"origin_keys": [{"observation.state": {"start": 5, "end": 6}}]}}' \\
  --policy.action_slots='{"action.arm.position": {"origin_keys": [{"action": {"start": 0, "end": 5}}]}, "action.effector.position": {"origin_keys": [{"action": {"start": 5, "end": 6}}]}}' \\
  --rename_map='{"observation.images.top": "observation.images.camera_top", "observation.images.wrist": "observation.images.camera_wrist_left"}' \\
${COMMON} \\
  --output_dir=/outputs/rocm-train/${name}
EOF
  chmod +x "$ROOT/jobs/${name}.sh"
}

write_one lingbot-1 "lerobot-train" "--accelerator.mixed_precision=bf16"
write_one lingbot-ddp2 'accelerate launch --num_processes=2 "$(which lerobot-train)"' "--accelerator.mixed_precision=bf16"
write_one lingbot-fsdp2 'accelerate launch --num_processes=2 "$(which lerobot-train)"' "--parallelism.dp_shard=2"

echo "lingbot-only begin $(date -Is)"
if run_job lingbot-1 2 lingbot-1.sh; then
  run_job lingbot-ddp2 2,3 lingbot-ddp2.sh || true
  run_job lingbot-fsdp2 2,3 lingbot-fsdp2.sh || true
fi
rm -rf "$LINK"
echo "lingbot-only end $(date -Is)"
