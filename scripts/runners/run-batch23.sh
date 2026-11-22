#!/bin/bash
# Finish phase 2 (g05, pi05) after writing q01/q99, then phase 3 FSDP models.
# Does not push the dataset. Deletes large checkpoint weights after each verdict.
set -u
ROOT=/data/lerobot/outputs/rocm-train
LOG="$ROOT/run-batch23.log"
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
- batch: 2-3
- state: ${state}
- note: ${note}

Log: ${LOG}
EOF
  echo
  echo "----- $(date -Is) ${state}: ${note} -----"
}

free_weights() {
  local host_out="$1"
  find "$host_out" -name '*.safetensors' -delete 2>/dev/null || true
}

run_job() {
  local name="$1"
  local devices="$2"
  local script="$3"
  local host_out="$ROOT/$name"
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
  local verdict="FAIL"
  if [[ "$ec" -eq 0 && -f "$ckpt" ]]; then
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

write_status "starting" "quantile stats"
cat >"$ROOT/jobs/quantile-prep.sh" <<'EOF'
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
EOF

COMMON='  --dataset.repo_id=lerobot/svla_so100_pickplace \
  --dataset.episodes='"'"'[0]'"'"' \
  --batch_size=1 \
  --steps=2 \
  --env_eval_freq=0 \
  --save_freq=2 \
  --log_freq=1 \
  --num_workers=0 \
  --trackio.enable=false'

write_pair() {
  local name="$1"
  local body="$2"
  local fsdp_extra="$3"
  cat >"$ROOT/jobs/${name}-1.sh" <<EOF
#!/bin/bash
set -euo pipefail
lerobot-train \\
${body} \\
${COMMON} \\
  --output_dir=/outputs/rocm-train/${name}-1
EOF
  cat >"$ROOT/jobs/${name}-ddp2.sh" <<EOF
#!/bin/bash
set -euo pipefail
accelerate launch --num_processes=2 "\$(which lerobot-train)" \\
${body} \\
${COMMON} \\
  --output_dir=/outputs/rocm-train/${name}-ddp2
EOF
  cat >"$ROOT/jobs/${name}-fsdp2.sh" <<EOF
#!/bin/bash
set -euo pipefail
accelerate launch --num_processes=2 "\$(which lerobot-train)" \\
${body} \\
${fsdp_extra} \\
${COMMON} \\
  --output_dir=/outputs/rocm-train/${name}-fsdp2
EOF
}

write_pair g05 '  --policy.path=lerobot/g05_so101 \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --policy.dtype=bfloat16 \
  --policy.camera_order='"'"'["observation.images.top", "observation.images.wrist_left", "observation.images.wrist"]'"'"' \
  --policy.n_obs_steps=1' ''

write_pair pi05 '  --policy.type=pi05 \
  --policy.pretrained_path=lerobot/pi05_base \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --policy.dtype=bfloat16 \
  --policy.empty_cameras=1 \
  --policy.gradient_checkpointing=true \
  --policy.compile_model=false \
  --rename_map='"'"'{"observation.images.top":"observation.images.base_0_rgb","observation.images.wrist":"observation.images.left_wrist_0_rgb"}'"'"'' ''

write_pair wall '  --policy.path=lerobot/wall-oss-0.5 \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --policy.use_language_recipe=false \
  --policy.camera_key_mapping='"'"'{"observation.images.top":"face_view","observation.images.wrist":"right_wrist_view"}'"'"'' '  --parallelism.dp_shard=2'

write_pair fastwam '  --policy.path=lerobot/fastwam_base \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --policy.dtype=float32 \
  --policy.action_dim=6 \
  --policy.proprio_dim=6' '  --parallelism.dp_shard=2'

write_pair lingbot '  --policy.path=lerobot/lingbot_vla_v2_base \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --policy.dtype=float32 \
  --accelerator.mixed_precision=bf16 \
  --policy.state_slots='"'"'{"observation.state.arm.position": {"origin_keys": [{"observation.state": {"start": 0, "end": 5}}]}, "observation.state.effector.position": {"origin_keys": [{"observation.state": {"start": 5, "end": 6}}]}}'"'"' \
  --policy.action_slots='"'"'{"action.arm.position": {"origin_keys": [{"action": {"start": 0, "end": 5}}]}, "action.effector.position": {"origin_keys": [{"action": {"start": 5, "end": 6}}]}}'"'"' \
  --rename_map='"'"'{"observation.images.top": "observation.images.camera_top", "observation.images.wrist": "observation.images.camera_wrist_left"}'"'"'' '  --parallelism.dp_shard=2'

write_pair cosmos3edge '  --policy.type=cosmos3_edge \
  --policy.pretrained_path=lerobot/Cosmos3-Edge \
  --policy.load_action_modules=false \
  --policy.gradient_checkpointing=true \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --accelerator.mixed_precision=bf16' '  --parallelism.dp_shard=2 \
  --accelerator.fsdp.mixed_precision_policy.param_dtype=bfloat16 \
  --accelerator.fsdp.mixed_precision_policy.reduce_dtype=bfloat16'

write_pair cosmos3nano '  --policy.type=cosmos3_nano \
  --policy.pretrained_path=lerobot/Cosmos3-Nano \
  --policy.load_action_modules=false \
  --policy.gradient_checkpointing=true \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --accelerator.mixed_precision=bf16' '  --parallelism.dp_shard=2 \
  --accelerator.fsdp.mixed_precision_policy.param_dtype=bfloat16 \
  --accelerator.fsdp.mixed_precision_policy.reduce_dtype=bfloat16'

write_pair lawam '  --policy.path=jialei02/lawam-pretrain-lerobot \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --dataset.image_transforms.enable=false \
  --rename_map='"'"'{"observation.images.top":"observation.images.image","observation.images.wrist":"observation.images.image2"}'"'"'' '  --parallelism.dp_shard=2'

chmod +x "$ROOT/jobs/"*.sh

echo "batch23 begin $(date -Is)"
if ! timeout --signal=TERM 1800 docker run --rm --name rocm-quantile \
  --network host \
  -v /data/lerobot/outputs:/outputs \
  -v /root/.cache/huggingface:/root/.cache/huggingface \
  lerobot-rocm \
  bash /outputs/rocm-train/jobs/quantile-prep.sh
then
  echo "RESULT quantile-prep FAIL"
  echo "quantile-prep FAIL" >>"$RESULTS"
  write_status "finished" "quantile-prep failed"
  exit 1
fi
echo "RESULT quantile-prep PASS"

run_chain() {
  local name="$1"
  local with_fsdp="$2"
  if run_job "${name}-1" 0 "${name}-1.sh"; then
    run_job "${name}-ddp2" 0,1 "${name}-ddp2.sh" || true
    if [[ "$with_fsdp" == "fsdp" ]]; then
      run_job "${name}-fsdp2" 0,1 "${name}-fsdp2.sh" || true
    fi
  fi
}

run_chain g05 no
run_chain pi05 no
run_chain wall fsdp
run_chain fastwam fsdp
run_chain lingbot fsdp
run_chain cosmos3edge fsdp
run_chain cosmos3nano fsdp
run_chain lawam fsdp

write_status "finished" "$(grep -E '^(g05|pi05|wall|fastwam|lingbot|cosmos3|lawam|quantile)' "$RESULTS" | paste -sd '; ' - || echo none)"
echo "batch23 end $(date -Is)"
