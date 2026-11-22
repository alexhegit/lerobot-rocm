#!/bin/bash
# Remaining smokes: lingbot, lawam FSDP, MolmoAct2.
set -u
ROOT=/data/lerobot/outputs/rocm-train
LOG="$ROOT/run-remaining.log"
STATUS="$ROOT/STATUS.md"
RESULTS="$ROOT/results.txt"
mkdir -p "$ROOT/jobs"
: >"$LOG"
exec >>"$LOG" 2>&1

write_status() {
  local state="$1"
  local note="$2"
  cat >"$STATUS" <<EOF
# rocm-train smoke

- updated: $(date -Is)
- batch: remaining
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
  find "$host_out" -type f \( -name '*.safetensors' -o -name '*.distcp' -o -name '*.bin' -o -name '*.pt' -o -name '*.pth' \) -delete 2>/dev/null || true
}

ensure_disk() {
  local avail_g
  avail_g=$(df -BG / | awk 'NR==2{gsub(/G/,"",$4); print $4}')
  echo "disk_free_gb=${avail_g}"
  if [[ "${avail_g}" -lt 80 ]]; then
    find "$ROOT" -type d -name training_state -prune -exec rm -rf {} + 2>/dev/null || true
    find "$ROOT" -type f \( -name '*.safetensors' -o -name '*.distcp' \) -delete 2>/dev/null || true
  fi
}

run_job() {
  local name="$1"
  local devices="$2"
  local script="$3"
  local host_out="$ROOT/$name"
  ensure_disk
  rm -rf "$host_out"
  write_status "running" "$name"
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

write_status "starting" "patch configs"
cat >"$ROOT/jobs/patch-remaining.sh" <<'EOF'
#!/bin/bash
set -euo pipefail
python - <<'PY'
import dataclasses
import json
from pathlib import Path

from huggingface_hub import snapshot_download
from lerobot.policies.molmoact2.configuration_molmoact2 import MolmoAct2Config

ling = Path("/outputs/rocm-train/lingbot-compat")
ling.mkdir(parents=True, exist_ok=True)
snapshot_download("lerobot/lingbot_vla_v2_base", local_dir=str(ling))
ling_cfg_path = ling / "config.json"
ling_cfg = json.loads(ling_cfg_path.read_text())
ling_cfg["align_params"] = {}
ling_cfg_path.write_text(json.dumps(ling_cfg))
print("lingbot align_params cleared")

molmo = Path("/outputs/rocm-train/molmo-compat")
molmo.mkdir(parents=True, exist_ok=True)
snapshot_download("lerobot/MolmoAct2-SO100_101-LeRobot", local_dir=str(molmo))
molmo_cfg_path = molmo / "config.json"
molmo_cfg = json.loads(molmo_cfg_path.read_text())
allowed = {field.name for field in dataclasses.fields(MolmoAct2Config)} | {"type"}
extra = sorted(set(molmo_cfg) - allowed)
for key in extra:
    molmo_cfg.pop(key)
molmo_cfg_path.write_text(json.dumps(molmo_cfg))
print("molmo stripped", extra)
print("molmo image_keys", molmo_cfg.get("image_keys"))
PY
EOF
chmod +x "$ROOT/jobs/patch-remaining.sh"

if ! timeout --signal=TERM 3600 docker run --rm --name rocm-patch-remaining \
  --network host \
  -v /data/lerobot/outputs:/outputs \
  -v /root/.cache/huggingface:/root/.cache/huggingface \
  lerobot-rocm \
  bash /outputs/rocm-train/jobs/patch-remaining.sh
then
  echo "RESULT patch-remaining FAIL"
  echo "patch-remaining FAIL" >>"$RESULTS"
  write_status "finished" "patch failed"
  exit 1
fi
echo "RESULT patch-remaining PASS"

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
accelerate launch --num_processes=2 "\$(which lerobot-train)" \\
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

cat >"$ROOT/jobs/molmo-1.sh" <<EOF
#!/bin/bash
set -euo pipefail
lerobot-train \\
  --policy.path=/outputs/rocm-train/molmo-compat \\
  --policy.device=cuda \\
  --policy.push_to_hub=false \\
  --policy.dtype=bfloat16 \\
  --policy.gradient_checkpointing=true \\
  --rename_map='{"observation.images.top":"observation.images.cam0","observation.images.wrist":"observation.images.cam1"}' \\
${COMMON} \\
  --output_dir=/outputs/rocm-train/molmo-1
EOF

cat >"$ROOT/jobs/molmo-ddp2.sh" <<EOF
#!/bin/bash
set -euo pipefail
accelerate launch --num_processes=2 "\$(which lerobot-train)" \\
  --policy.path=/outputs/rocm-train/molmo-compat \\
  --policy.device=cuda \\
  --policy.push_to_hub=false \\
  --policy.dtype=bfloat16 \\
  --policy.gradient_checkpointing=true \\
  --rename_map='{"observation.images.top":"observation.images.cam0","observation.images.wrist":"observation.images.cam1"}' \\
${COMMON} \\
  --output_dir=/outputs/rocm-train/molmo-ddp2
EOF
chmod +x "$ROOT/jobs/"lingbot-*.sh "$ROOT/jobs/lawam-fsdp2.sh" "$ROOT/jobs/"molmo-*.sh

echo "remaining begin $(date -Is)"
if run_job lingbot-1 0 lingbot-1.sh; then
  run_job lingbot-ddp2 0,1 lingbot-ddp2.sh || true
  run_job lingbot-fsdp2 0,1 lingbot-fsdp2.sh || true
fi
run_job lawam-fsdp2 0,1 lawam-fsdp2.sh || true
if run_job molmo-1 0 molmo-1.sh; then
  run_job molmo-ddp2 0,1 molmo-ddp2.sh || true
fi
ensure_disk
write_status "finished" "$(grep -E '^(lingbot|lawam-fsdp2|molmo|patch-remaining)' "$RESULTS" | paste -sd '; ' - || echo none)"
echo "remaining end $(date -Is)"
