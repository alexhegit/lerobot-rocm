#!/bin/bash
# g05 retry on GPUs 2 and 3 while the pi/eo1 queue uses 0 and 1.
set -u
ROOT=/data/lerobot/outputs/rocm-train
LOG="$ROOT/run-g05-fix.log"
RESULTS="$ROOT/results.txt"
mkdir -p "$ROOT/jobs"
: >"$LOG"
exec >>"$LOG" 2>&1

cat >"$ROOT/jobs/g05-1.sh" <<'EOF'
#!/bin/bash
set -euo pipefail
lerobot-train \
  --policy.path=lerobot/g05_so101 \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --policy.dtype=bfloat16 \
  --policy.camera_order='["observation.images.top", "observation.images.wrist_left", "observation.images.wrist"]' \
  --policy.n_obs_steps=1 \
  --dataset.repo_id=lerobot/svla_so100_pickplace \
  --dataset.episodes='[0]' \
  --batch_size=1 \
  --steps=2 \
  --env_eval_freq=0 \
  --save_freq=2 \
  --log_freq=1 \
  --num_workers=0 \
  --trackio.enable=false \
  --output_dir=/outputs/rocm-train/g05-1
EOF

cat >"$ROOT/jobs/g05-ddp2.sh" <<'EOF'
#!/bin/bash
set -euo pipefail
accelerate launch --num_processes=2 "$(which lerobot-train)" \
  --policy.path=lerobot/g05_so101 \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --policy.dtype=bfloat16 \
  --policy.camera_order='["observation.images.top", "observation.images.wrist_left", "observation.images.wrist"]' \
  --policy.n_obs_steps=1 \
  --dataset.repo_id=lerobot/svla_so100_pickplace \
  --dataset.episodes='[0]' \
  --batch_size=1 \
  --steps=2 \
  --env_eval_freq=0 \
  --save_freq=2 \
  --log_freq=1 \
  --num_workers=0 \
  --trackio.enable=false \
  --output_dir=/outputs/rocm-train/g05-ddp2
EOF
chmod +x "$ROOT/jobs/g05-1.sh" "$ROOT/jobs/g05-ddp2.sh"

run_job() {
  local name="$1"
  local devices="$2"
  local script="$3"
  local host_out="$ROOT/$name"
  rm -rf "$host_out"
  echo "----- $(date -Is) running ${name} devices=${devices} -----"
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
  local verdict="FAIL"
  if [[ "$ec" -eq 0 && -f "$ckpt" ]]; then
    verdict="PASS"
  elif [[ "$ec" -eq 124 ]]; then
    verdict="TIMEOUT"
    docker rm -f "rocm-${name}" >/dev/null 2>&1 || true
  fi
  echo "RESULT ${name} ${verdict} exit=${ec} seconds=${elapsed}"
  echo "${name} ${verdict} exit=${ec} seconds=${elapsed}" >>"$RESULTS"
  [[ "$verdict" == "PASS" ]]
}

echo "g05-fix begin $(date -Is)"
if run_job g05-1 2 g05-1.sh; then
  run_job g05-ddp2 2,3 g05-ddp2.sh || true
fi
echo "g05-fix end $(date -Is)"
