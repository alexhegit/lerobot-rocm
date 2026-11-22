#!/bin/bash
# Rerun lingbot from a symlink (no second 24GB copy) and lawam FSDP after an in-process fp32 cast.
set -u
ROOT=/data/lerobot/outputs/rocm-train
LOG="$ROOT/run-lingbot-lawam.log"
STATUS="$ROOT/STATUS.md"
RESULTS="$ROOT/results.txt"
SNAP=/root/.cache/huggingface/hub/models--lerobot--lingbot_vla_v2_base/snapshots/32d0a32a881c351d28b6c970ab4ecbc119d29e79
mkdir -p "$ROOT/jobs"
: >"$LOG"
exec >>"$LOG" 2>&1

write_status() {
  local state="$1"
  local note="$2"
  cat >"$STATUS" <<EOF
# rocm-train smoke

- updated: $(date -Is)
- batch: lingbot-lawam
- state: ${state}
- note: ${note}

Log: ${LOG}
EOF
  echo
  echo "----- $(date -Is) ${state}: ${note} -----"
}

free_weights() {
  local host_out="$1"
  find "$host_out" -type d -name training_state -prune -exec rm -rf {} + 2>/dev/null || true
  find "$host_out" -type f \( -name '*.safetensors' -o -name '*.distcp' -o -name '*.bin' -o -name '*.pt' \) -delete 2>/dev/null || true
}

run_job() {
  local name="$1"
  local devices="$2"
  local script="$3"
  local host_out="$ROOT/$name"
  rm -rf "$host_out"
  write_status "running" "$name"
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
  if [[ -f "$ckpt" || -f "$stepf" ]]; then
    has_ckpt=1
  elif find "$host_out/checkpoints/000002" -name '*.distcp' 2>/dev/null | grep -q .; then
    has_ckpt=1
  fi
  local verdict="FAIL"
  if [[ "$ec" -eq 0 && "$has_ckpt" -eq 1 ]]; then
    verdict="PASS"
  elif [[ "$ec" -eq 124 ]]; then
    verdict="TIMEOUT"
    docker rm -f "rocm-${name}" >/dev/null 2>&1 || true
  fi
  local step="missing"
  if [[ -f "$stepf" ]]; then
    step=$(tr -d '\n' <"$stepf")
  fi
  echo "RESULT ${name} ${verdict} exit=${ec} seconds=${elapsed}"
  echo "STEP ${step}"
  echo "${name} ${verdict} exit=${ec} seconds=${elapsed}" >>"$RESULTS"
  free_weights "$host_out"
  [[ "$verdict" == "PASS" ]]
}

write_status "starting" "lingbot symlink"
rm -rf "$ROOT/lingbot-compat" "$ROOT/lawam-fp32-compat" "$ROOT/molmo-compat"
mkdir -p "$ROOT/lingbot-compat"
cp -a "$SNAP/config.json" "$SNAP/policy_preprocessor.json" "$SNAP/policy_postprocessor.json" "$ROOT/lingbot-compat/"
cp -a "$SNAP/tokenizer" "$SNAP/image_processor" "$SNAP/chat_template_tokenizer" "$ROOT/lingbot-compat/"
ln -s "$SNAP/model.safetensors" "$ROOT/lingbot-compat/model.safetensors"
python3 - <<'PY'
import json
from pathlib import Path
p = Path("/data/lerobot/outputs/rocm-train/lingbot-compat/config.json")
cfg = json.loads(p.read_text())
cfg["align_params"] = {}
p.write_text(json.dumps(cfg))
weight = Path("/data/lerobot/outputs/rocm-train/lingbot-compat/model.safetensors")
print("lingbot link", weight.resolve(), "bytes", weight.stat().st_size, "align", cfg["align_params"])
PY

COMMON='  --dataset.repo_id=lerobot/svla_so100_pickplace \
  --dataset.episodes='"'"'[0]'"'"' \
  --batch_size=1 \
  --steps=2 \
  --env_eval_freq=0 \
  --save_freq=2 \
  --log_freq=1 \
  --num_workers=0 \
  --trackio.enable=false'

cat >"$ROOT/jobs/lingbot-1.sh" <<EOF
#!/bin/bash
set -euo pipefail
lerobot-train \\
  --policy.path=/outputs/rocm-train/lingbot-compat \\
  --policy.device=cuda \\
  --policy.push_to_hub=false \\
  --policy.dtype=float32 \\
  --accelerator.mixed_precision=bf16 \\
  --policy.state_slots='{"observation.state.arm.position": {"origin_keys": [{"observation.state": {"start": 0, "end": 5}}]}, "observation.state.effector.position": {"origin_keys": [{"observation.state": {"start": 5, "end": 6}}]}}' \\
  --policy.action_slots='{"action.arm.position": {"origin_keys": [{"action": {"start": 0, "end": 5}}]}, "action.effector.position": {"origin_keys": [{"action": {"start": 5, "end": 6}}]}}' \\
  --rename_map='{"observation.images.top": "observation.images.camera_top", "observation.images.wrist": "observation.images.camera_wrist_left"}' \\
${COMMON} \\
  --output_dir=/outputs/rocm-train/lingbot-1
EOF

cat >"$ROOT/jobs/lingbot-ddp2.sh" <<EOF
#!/bin/bash
set -euo pipefail
accelerate launch --num_processes=2 "\$(which lerobot-train)" \\
  --policy.path=/outputs/rocm-train/lingbot-compat \\
  --policy.device=cuda \\
  --policy.push_to_hub=false \\
  --policy.dtype=float32 \\
  --accelerator.mixed_precision=bf16 \\
  --policy.state_slots='{"observation.state.arm.position": {"origin_keys": [{"observation.state": {"start": 0, "end": 5}}]}, "observation.state.effector.position": {"origin_keys": [{"observation.state": {"start": 5, "end": 6}}]}}' \\
  --policy.action_slots='{"action.arm.position": {"origin_keys": [{"action": {"start": 0, "end": 5}}]}, "action.effector.position": {"origin_keys": [{"action": {"start": 5, "end": 6}}]}}' \\
  --rename_map='{"observation.images.top": "observation.images.camera_top", "observation.images.wrist": "observation.images.camera_wrist_left"}' \\
${COMMON} \\
  --output_dir=/outputs/rocm-train/lingbot-ddp2
EOF

cat >"$ROOT/jobs/lingbot-fsdp2.sh" <<EOF
#!/bin/bash
set -euo pipefail
accelerate launch --num_processes=2 "\$(which lerobot-train)" \\
  --policy.path=/outputs/rocm-train/lingbot-compat \\
  --policy.device=cuda \\
  --policy.push_to_hub=false \\
  --policy.dtype=float32 \\
  --parallelism.dp_shard=2 \\
  --policy.state_slots='{"observation.state.arm.position": {"origin_keys": [{"observation.state": {"start": 0, "end": 5}}]}, "observation.state.effector.position": {"origin_keys": [{"observation.state": {"start": 5, "end": 6}}]}}' \\
  --policy.action_slots='{"action.arm.position": {"origin_keys": [{"action": {"start": 0, "end": 5}}]}, "action.effector.position": {"origin_keys": [{"action": {"start": 5, "end": 6}}]}}' \\
  --rename_map='{"observation.images.top": "observation.images.camera_top", "observation.images.wrist": "observation.images.camera_wrist_left"}' \\
${COMMON} \\
  --output_dir=/outputs/rocm-train/lingbot-fsdp2
EOF

cat >"$ROOT/jobs/lawam-fsdp2.sh" <<EOF
#!/bin/bash
set -euo pipefail
accelerate launch --num_processes=2 /outputs/rocm-train/jobs/lawam_fsdp_cast.py \\
  --policy.path=jialei02/lawam-pretrain-lerobot \\
  --policy.device=cuda \\
  --policy.push_to_hub=false \\
  --policy.dtype=float32 \\
  --dataset.image_transforms.enable=false \\
  --rename_map='{"observation.images.top":"observation.images.image","observation.images.wrist":"observation.images.image2"}' \\
  --parallelism.dp_shard=2 \\
${COMMON} \\
  --output_dir=/outputs/rocm-train/lawam-fsdp2
EOF
chmod +x "$ROOT/jobs/"*.sh

echo "lingbot-lawam begin $(date -Is)"
if run_job lingbot-1 0 lingbot-1.sh; then
  run_job lingbot-ddp2 0,1 lingbot-ddp2.sh || true
  run_job lingbot-fsdp2 0,1 lingbot-fsdp2.sh || true
fi
run_job lawam-fsdp2 0,1 lawam-fsdp2.sh || true
rm -rf "$ROOT/lingbot-compat"
write_status "finished" "$(grep -E '^(lingbot|lawam-fsdp2)' "$RESULTS" | paste -sd '; ' - || echo none)"
echo "lingbot-lawam end $(date -Is)"
