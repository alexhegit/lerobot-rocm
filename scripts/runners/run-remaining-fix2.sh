#!/bin/bash
# Fix remaining: lingbot (full local weights + empty align), lawam FSDP (fp32 cast), molmo (fft).
set -u
ROOT=/data/lerobot/outputs/rocm-train
LOG="$ROOT/run-remaining-fix2.log"
STATUS="$ROOT/STATUS.md"
RESULTS="$ROOT/results.txt"
mkdir -p "$ROOT/jobs"
: >"$LOG"
exec >>"$LOG" 2>&1

COMPAT_GLOBS=(lingbot-compat molmo-compat lawam-fp32-compat)

write_status() {
  local state="$1"
  local note="$2"
  cat >"$STATUS" <<EOF
# rocm-train smoke

- updated: $(date -Is)
- batch: remaining-fix2
- state: ${state}
- note: ${note}

Log: ${LOG}
EOF
  echo
  echo "----- $(date -Is) ${state}: ${note} -----"
}

is_compat_path() {
  local p="$1"
  for name in "${COMPAT_GLOBS[@]}"; do
    [[ "$p" == *"/outputs/rocm-train/${name}"* || "$p" == *"/rocm-train/${name}"* || "$p" == *"/${name}/"* || "$p" == *"/${name}" ]] && return 0
  done
  return 1
}

free_weights() {
  local host_out="$1"
  if is_compat_path "$host_out"; then
    return 0
  fi
  find "$host_out" -type d -name training_state -prune -exec rm -rf {} + 2>/dev/null || true
  find "$host_out" -type f \( -name '*.safetensors' -o -name '*.distcp' -o -name '*.bin' -o -name '*.pt' -o -name '*.pth' \) -delete 2>/dev/null || true
}

ensure_disk() {
  local avail_g
  avail_g=$(df -BG / | awk 'NR==2{gsub(/G/,"",$4); print $4}')
  echo "disk_free_gb=${avail_g}"
  if [[ "${avail_g}" -lt 80 ]]; then
    echo "low disk; scrubbing finished smoke weights (compat dirs protected)"
    find "$ROOT" -mindepth 1 -maxdepth 1 -type d ! -name 'lingbot-compat' ! -name 'molmo-compat' ! -name 'lawam-fp32-compat' ! -name 'jobs' | while read -r d; do
      find "$d" -type d -name training_state -prune -exec rm -rf {} + 2>/dev/null || true
      find "$d" -type f \( -name '*.safetensors' -o -name '*.distcp' \) -delete 2>/dev/null || true
    done
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

write_status "starting" "prepare compat checkpoints"
cat >"$ROOT/jobs/prepare-fix2.sh" <<'EOF'
#!/bin/bash
set -euo pipefail
python - <<'PY'
import dataclasses
import json
import shutil
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download, snapshot_download
from safetensors.torch import load_file, save_file

from lerobot.policies.molmoact2.configuration_molmoact2 import MolmoAct2Config

root = Path("/outputs/rocm-train")

# --- lingbot ---
ling = root / "lingbot-compat"
if ling.exists():
    shutil.rmtree(ling)
ling.mkdir(parents=True)
snapshot_download("lerobot/lingbot_vla_v2_base", local_dir=str(ling))
weight = ling / "model.safetensors"
if not weight.is_file():
    hf_hub_download("lerobot/lingbot_vla_v2_base", "model.safetensors", local_dir=str(ling))
cfg = json.loads((ling / "config.json").read_text())
cfg["align_params"] = {}
(ling / "config.json").write_text(json.dumps(cfg))
print("lingbot bytes", weight.stat().st_size, "align", cfg["align_params"])

# --- molmo ---
molmo = root / "molmo-compat"
if molmo.exists():
    shutil.rmtree(molmo)
molmo.mkdir(parents=True)
snapshot_download("lerobot/MolmoAct2-SO100_101-LeRobot", local_dir=str(molmo))
mweight = molmo / "model.safetensors"
if not mweight.is_file():
    hf_hub_download("lerobot/MolmoAct2-SO100_101-LeRobot", "model.safetensors", local_dir=str(molmo))
mcfg = json.loads((molmo / "config.json").read_text())
allowed = {field.name for field in dataclasses.fields(MolmoAct2Config)} | {"type"}
extra = sorted(set(mcfg) - allowed)
for key in extra:
    mcfg.pop(key)
mcfg["train_mode_vlm"] = "fft"
(molmo / "config.json").write_text(json.dumps(mcfg))
print("molmo bytes", mweight.stat().st_size, "stripped", extra, "train_mode", mcfg["train_mode_vlm"])

# --- lawam fp32 cast ---
lawam = root / "lawam-fp32-compat"
if lawam.exists():
    shutil.rmtree(lawam)
lawam.mkdir(parents=True)
src = Path(snapshot_download("jialei02/lawam-pretrain-lerobot"))
for item in src.iterdir():
    dest = lawam / item.name
    if item.is_dir():
        shutil.copytree(item, dest, dirs_exist_ok=True)
    elif item.name != "model.safetensors":
        shutil.copy2(item, dest)
src_weight = src / "model.safetensors"
if not src_weight.is_file():
    src_weight = Path(hf_hub_download("jialei02/lawam-pretrain-lerobot", "model.safetensors"))
state = load_file(str(src_weight))
dtypes = {}
out = {}
for key, value in state.items():
    if value.is_floating_point() and value.dtype != torch.float32:
        dtypes[str(value.dtype)] = dtypes.get(str(value.dtype), 0) + 1
        out[key] = value.float().contiguous()
    else:
        out[key] = value
save_file(out, str(lawam / "model.safetensors"))
print("lawam cast", dtypes, "out_bytes", (lawam / "model.safetensors").stat().st_size)
PY
EOF
chmod +x "$ROOT/jobs/prepare-fix2.sh"

if ! timeout --signal=TERM 7200 docker run --rm --name rocm-prepare-fix2 \
  --network host \
  -v /data/lerobot/outputs:/outputs \
  -v /root/.cache/huggingface:/root/.cache/huggingface \
  lerobot-rocm \
  bash /outputs/rocm-train/jobs/prepare-fix2.sh
then
  echo "RESULT prepare-fix2 FAIL"
  echo "prepare-fix2 FAIL" >>"$RESULTS"
  write_status "finished" "prepare failed"
  exit 1
fi
echo "RESULT prepare-fix2 PASS"
echo "prepare-fix2 PASS" >>"$RESULTS"

# verify weights exist on host
for f in lingbot-compat/model.safetensors molmo-compat/model.safetensors lawam-fp32-compat/model.safetensors; do
  if [[ ! -f "$ROOT/$f" ]]; then
    echo "MISSING $f"
    write_status "finished" "missing $f"
    exit 1
  fi
  echo "ok $f $(stat -c%s "$ROOT/$f")"
done

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
  --policy.path=/outputs/rocm-train/lawam-fp32-compat \\
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
  --policy.train_mode_vlm=fft \\
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
  --policy.train_mode_vlm=fft \\
  --policy.gradient_checkpointing=true \\
  --rename_map='{"observation.images.top":"observation.images.cam0","observation.images.wrist":"observation.images.cam1"}' \\
${COMMON} \\
  --output_dir=/outputs/rocm-train/molmo-ddp2
EOF
chmod +x "$ROOT/jobs/"lingbot-*.sh "$ROOT/jobs/lawam-fsdp2.sh" "$ROOT/jobs/"molmo-*.sh

echo "remaining-fix2 begin $(date -Is)"
if run_job lingbot-1 0 lingbot-1.sh; then
  run_job lingbot-ddp2 0,1 lingbot-ddp2.sh || true
  run_job lingbot-fsdp2 0,1 lingbot-fsdp2.sh || true
fi
run_job lawam-fsdp2 0,1 lawam-fsdp2.sh || true
if run_job molmo-1 0 molmo-1.sh; then
  run_job molmo-ddp2 0,1 molmo-ddp2.sh || true
fi

# final cleanup of compat dirs and leftover weights
rm -rf "$ROOT/lingbot-compat" "$ROOT/molmo-compat" "$ROOT/lawam-fp32-compat"
ensure_disk
write_status "finished" "$(grep -E '^(prepare-fix2|lingbot|lawam-fsdp2|molmo)' "$RESULTS" | paste -sd '; ' - || echo none)"
echo "remaining-fix2 end $(date -Is)"
