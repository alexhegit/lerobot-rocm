#!/bin/bash
# Retry remaining batch2/3 failures. Appends to results.txt; new log only.
set -u
ROOT=/data/lerobot/outputs/rocm-train
LOG="$ROOT/run-batch23-retry.log"
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
- batch: 2-3-retry
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
    echo "low disk; scrubbing finished smoke weights"
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
  # FSDP may save DCP shards instead of a single safetensors
  local dcp_dir="$host_out/checkpoints/000002/pretrained_model"
  local has_ckpt=0
  if [[ -f "$ckpt" ]]; then
    has_ckpt=1
  elif find "$dcp_dir" -name '*.distcp' 2>/dev/null | grep -q .; then
    has_ckpt=1
  elif [[ -f "$host_out/checkpoints/000002/training_state/training_step.json" ]]; then
    # accept training_step + exit 0 as evidence when weights already scrubbed mid-run
    has_ckpt=1
  fi
  local stepf="$host_out/checkpoints/000002/training_state/training_step.json"
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

COMMON_SO100="  --dataset.repo_id=lerobot/svla_so100_pickplace \\
  --dataset.episodes='[0]' \\
  --batch_size=1 \\
  --steps=2 \\
  --env_eval_freq=0 \\
  --save_freq=2 \\
  --log_freq=1 \\
  --num_workers=0 \\
  --trackio.enable=false"

COMMON_LIBERO="  --dataset.repo_id=lerobot/libero \\
  --dataset.episodes='[0]' \\
  --batch_size=1 \\
  --steps=2 \\
  --env_eval_freq=0 \\
  --save_freq=2 \\
  --log_freq=1 \\
  --num_workers=0 \\
  --trackio.enable=false"

write_job() {
  local file="$1"
  local launcher="$2"
  local body="$3"
  local common="$4"
  local out="$5"
  cat >"$ROOT/jobs/${file}" <<EOF
#!/bin/bash
set -euo pipefail
${launcher} \\
${body} \\
${common} \\
  --output_dir=/outputs/rocm-train/${out}
EOF
  chmod +x "$ROOT/jobs/${file}"
}

echo "batch23-retry begin $(date -Is)"
write_status "starting" "batch23-retry"

# --- g05: disable CoT/recipe so prompt does not need atomic_task ---
G05_BODY='  --policy.path=lerobot/g05_so101 \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --policy.dtype=bfloat16 \
  --policy.use_language_recipe=false \
  --policy.predict_cot=false \
  --policy.runtime_system=system1 \
  --policy.camera_order='"'"'["observation.images.top", "observation.images.wrist_left", "observation.images.wrist"]'"'"' \
  --policy.n_obs_steps=1'
write_job g05-1.sh "lerobot-train" "$G05_BODY" "$COMMON_SO100" "g05-1"
write_job g05-ddp2.sh 'accelerate launch --num_processes=2 "$(which lerobot-train)"' "$G05_BODY" "$COMMON_SO100" "g05-ddp2"

# --- fastwam: match checkpoint 7/8 on libero episode 0 ---
FAST_BODY='  --policy.path=lerobot/fastwam_base \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --policy.dtype=float32 \
  --policy.action_dim=7 \
  --policy.proprio_dim=8'
write_job fastwam-1.sh "lerobot-train" "$FAST_BODY" "$COMMON_LIBERO" "fastwam-1"
write_job fastwam-ddp2.sh 'accelerate launch --num_processes=2 "$(which lerobot-train)"' "$FAST_BODY" "$COMMON_LIBERO" "fastwam-ddp2"
write_job fastwam-fsdp2.sh 'accelerate launch --num_processes=2 "$(which lerobot-train)"' "${FAST_BODY} \\
  --parallelism.dp_shard=2" "$COMMON_LIBERO" "fastwam-fsdp2"

# --- cosmos3nano DDP one more try: bf16 policy weights ---
NANO_DDP_BODY='  --policy.type=cosmos3_nano \
  --policy.pretrained_path=lerobot/Cosmos3-Nano \
  --policy.load_action_modules=false \
  --policy.gradient_checkpointing=true \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --policy.dtype=bfloat16 \
  --accelerator.mixed_precision=bf16'
write_job cosmos3nano-ddp2.sh 'accelerate launch --num_processes=2 "$(which lerobot-train)"' "$NANO_DDP_BODY" "$COMMON_SO100" "cosmos3nano-ddp2"

# --- lingbot: disable distillation teachers via empty align_params ---
LING_BODY='  --policy.path=lerobot/lingbot_vla_v2_base \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --policy.dtype=float32 \
  --accelerator.mixed_precision=bf16 \
  --policy.align_params='"'"'{}'"'"' \
  --policy.state_slots='"'"'{"observation.state.arm.position": {"origin_keys": [{"observation.state": {"start": 0, "end": 5}}]}, "observation.state.effector.position": {"origin_keys": [{"observation.state": {"start": 5, "end": 6}}]}}'"'"' \
  --policy.action_slots='"'"'{"action.arm.position": {"origin_keys": [{"action": {"start": 0, "end": 5}}]}, "action.effector.position": {"origin_keys": [{"action": {"start": 5, "end": 6}}]}}'"'"' \
  --rename_map='"'"'{"observation.images.top": "observation.images.camera_top", "observation.images.wrist": "observation.images.camera_wrist_left"}'"'"''
write_job lingbot-1.sh "lerobot-train" "$LING_BODY" "$COMMON_SO100" "lingbot-1"
write_job lingbot-ddp2.sh 'accelerate launch --num_processes=2 "$(which lerobot-train)"' "$LING_BODY" "$COMMON_SO100" "lingbot-ddp2"
write_job lingbot-fsdp2.sh 'accelerate launch --num_processes=2 "$(which lerobot-train)"' "${LING_BODY} \\
  --parallelism.dp_shard=2" "$COMMON_SO100" "lingbot-fsdp2"

# --- wall FSDP: force float32 masters ---
WALL_FSDP='  --policy.path=lerobot/wall-oss-0.5 \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --policy.dtype=float32 \
  --policy.use_language_recipe=false \
  --policy.camera_key_mapping='"'"'{"observation.images.top":"face_view","observation.images.wrist":"right_wrist_view"}'"'"' \
  --parallelism.dp_shard=2 \
  --accelerator.mixed_precision=bf16'
write_job wall-fsdp2.sh 'accelerate launch --num_processes=2 "$(which lerobot-train)"' "$WALL_FSDP" "$COMMON_SO100" "wall-fsdp2"

# --- lawam FSDP: force float32 ---
LAWAM_FSDP='  --policy.path=jialei02/lawam-pretrain-lerobot \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --policy.dtype=float32 \
  --dataset.image_transforms.enable=false \
  --rename_map='"'"'{"observation.images.top":"observation.images.image","observation.images.wrist":"observation.images.image2"}'"'"' \
  --parallelism.dp_shard=2 \
  --accelerator.mixed_precision=bf16'
write_job lawam-fsdp2.sh 'accelerate launch --num_processes=2 "$(which lerobot-train)"' "$LAWAM_FSDP" "$COMMON_SO100" "lawam-fsdp2"

# --- optional molmoact2 ---
MOLMO_BODY='  --policy.path=lerobot/MolmoAct2-SO100_101-LeRobot \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --policy.dtype=bfloat16'
write_job molmo-1.sh "lerobot-train" "$MOLMO_BODY" "$COMMON_SO100" "molmo-1"
write_job molmo-ddp2.sh 'accelerate launch --num_processes=2 "$(which lerobot-train)"' "$MOLMO_BODY" "$COMMON_SO100" "molmo-ddp2"

run_chain() {
  local name="$1"
  shift
  local modes=("$@")
  if run_job "${name}-1" 0 "${name}-1.sh"; then
    for mode in "${modes[@]}"; do
      case "$mode" in
        ddp) run_job "${name}-ddp2" 0,1 "${name}-ddp2.sh" || true ;;
        fsdp) run_job "${name}-fsdp2" 0,1 "${name}-fsdp2.sh" || true ;;
      esac
    done
  fi
}

run_chain g05 ddp
run_chain fastwam ddp fsdp

write_status "running" "cosmos3nano-ddp2 retry"
run_job cosmos3nano-ddp2 0,1 cosmos3nano-ddp2.sh || true

run_chain lingbot ddp fsdp

write_status "running" "wall-fsdp2 retry"
run_job wall-fsdp2 0,1 wall-fsdp2.sh || true
write_status "running" "lawam-fsdp2 retry"
run_job lawam-fsdp2 0,1 lawam-fsdp2.sh || true

avail_g=$(df -BG / | awk 'NR==2{gsub(/G/,"",$4); print $4}')
if [[ "${avail_g}" -ge 100 ]]; then
  run_chain molmo ddp
else
  echo "SKIP molmo disk_free_gb=${avail_g}"
  echo "molmo SKIP disk_free_gb=${avail_g}" >>"$RESULTS"
fi

ensure_disk
write_status "finished" "$(grep -E '^(g05|fastwam|cosmos3nano-ddp2|lingbot|wall-fsdp2|lawam-fsdp2|molmo)' "$RESULTS" | paste -sd '; ' - || echo none)"
echo "batch23-retry end $(date -Is)"
