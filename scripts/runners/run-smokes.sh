#!/bin/bash
# Overnight smoke queue for lerobot-rocm on DOUni. One job at a time.
set -u
ROOT=/data/lerobot/outputs/rocm-train
LOG="$ROOT/run.log"
STATUS="$ROOT/STATUS.md"
mkdir -p "$ROOT/jobs"
exec >>"$LOG" 2>&1

write_status() {
  local state="$1"
  local note="$2"
  cat >"$STATUS" <<EOF
# rocm-train smoke

- updated: $(date -Is)
- state: ${state}
- note: ${note}

Log: ${LOG}
EOF
  echo
  echo "----- $(date -Is) ${state}: ${note} -----"
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
  echo "${name} ${verdict} exit=${ec} seconds=${elapsed}" >>"$ROOT/results.txt"
  if [[ "$verdict" == "PASS" ]]; then
    return 0
  fi
  return 1
}

write_status "starting" "queue"
echo "queue begin $(date -Is)"

cat >"$ROOT/jobs/diffusion-1.sh" <<'EOF'
#!/bin/bash
set -euo pipefail
lerobot-train \
  --policy.type=diffusion \
  --policy.down_dims='[64,128,256]' \
  --policy.diffusion_step_embed_dim=32 \
  --policy.num_inference_steps=10 \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --dataset.repo_id=lerobot/aloha_sim_transfer_cube_human \
  --dataset.episodes='[0]' \
  --batch_size=2 \
  --steps=2 \
  --env_eval_freq=0 \
  --save_freq=2 \
  --log_freq=1 \
  --num_workers=0 \
  --trackio.enable=false \
  --output_dir=/outputs/rocm-train/diffusion-1
EOF

cat >"$ROOT/jobs/diffusion-ddp2.sh" <<'EOF'
#!/bin/bash
set -euo pipefail
accelerate launch --num_processes=2 "$(which lerobot-train)" \
  --policy.type=diffusion \
  --policy.down_dims='[64,128,256]' \
  --policy.diffusion_step_embed_dim=32 \
  --policy.num_inference_steps=10 \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --dataset.repo_id=lerobot/aloha_sim_transfer_cube_human \
  --dataset.episodes='[0]' \
  --batch_size=2 \
  --steps=2 \
  --env_eval_freq=0 \
  --save_freq=2 \
  --log_freq=1 \
  --num_workers=0 \
  --trackio.enable=false \
  --output_dir=/outputs/rocm-train/diffusion-ddp2
EOF

cat >"$ROOT/jobs/act-fsdp2.sh" <<'EOF'
#!/bin/bash
set -euo pipefail
accelerate launch --num_processes=2 "$(which lerobot-train)" \
  --policy.type=act \
  --policy.dim_model=64 \
  --policy.n_action_steps=20 \
  --policy.chunk_size=20 \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --parallelism.dp_shard=2 \
  --dataset.repo_id=lerobot/aloha_sim_transfer_cube_human \
  --dataset.episodes='[0]' \
  --batch_size=2 \
  --steps=2 \
  --env_eval_freq=0 \
  --save_freq=2 \
  --log_freq=1 \
  --num_workers=0 \
  --trackio.enable=false \
  --output_dir=/outputs/rocm-train/act-fsdp2
EOF

cat >"$ROOT/jobs/dit-1.sh" <<'EOF'
#!/bin/bash
set -euo pipefail
lerobot-train \
  --policy.type=multi_task_dit \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --policy.horizon=32 \
  --policy.n_action_steps=24 \
  --policy.objective=diffusion \
  --policy.num_inference_steps=10 \
  --dataset.repo_id=lerobot/aloha_sim_transfer_cube_human \
  --dataset.episodes='[0]' \
  --batch_size=1 \
  --steps=2 \
  --env_eval_freq=0 \
  --save_freq=2 \
  --log_freq=1 \
  --num_workers=0 \
  --trackio.enable=false \
  --output_dir=/outputs/rocm-train/dit-1
EOF

cat >"$ROOT/jobs/dit-ddp2.sh" <<'EOF'
#!/bin/bash
set -euo pipefail
accelerate launch --num_processes=2 "$(which lerobot-train)" \
  --policy.type=multi_task_dit \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --policy.horizon=32 \
  --policy.n_action_steps=24 \
  --policy.objective=diffusion \
  --policy.num_inference_steps=10 \
  --dataset.repo_id=lerobot/aloha_sim_transfer_cube_human \
  --dataset.episodes='[0]' \
  --batch_size=1 \
  --steps=2 \
  --env_eval_freq=0 \
  --save_freq=2 \
  --log_freq=1 \
  --num_workers=0 \
  --trackio.enable=false \
  --output_dir=/outputs/rocm-train/dit-ddp2
EOF

cat >"$ROOT/jobs/smolvla-1.sh" <<'EOF'
#!/bin/bash
set -euo pipefail
lerobot-train \
  --policy.path=lerobot/smolvla_base \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --dataset.repo_id=lerobot/svla_so100_pickplace \
  --dataset.episodes='[0]' \
  --rename_map='{"observation.images.top": "observation.images.camera1", "observation.images.wrist": "observation.images.camera2"}' \
  --batch_size=1 \
  --steps=2 \
  --env_eval_freq=0 \
  --save_freq=2 \
  --log_freq=1 \
  --num_workers=0 \
  --trackio.enable=false \
  --output_dir=/outputs/rocm-train/smolvla-1
EOF

cat >"$ROOT/jobs/smolvla-ddp2.sh" <<'EOF'
#!/bin/bash
set -euo pipefail
accelerate launch --num_processes=2 "$(which lerobot-train)" \
  --policy.path=lerobot/smolvla_base \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --dataset.repo_id=lerobot/svla_so100_pickplace \
  --dataset.episodes='[0]' \
  --rename_map='{"observation.images.top": "observation.images.camera1", "observation.images.wrist": "observation.images.camera2"}' \
  --batch_size=1 \
  --steps=2 \
  --env_eval_freq=0 \
  --save_freq=2 \
  --log_freq=1 \
  --num_workers=0 \
  --trackio.enable=false \
  --output_dir=/outputs/rocm-train/smolvla-ddp2
EOF

chmod +x "$ROOT/jobs/"*.sh

run_job diffusion-1 0 diffusion-1.sh || true
run_job diffusion-ddp2 0,1 diffusion-ddp2.sh || true
run_job act-fsdp2 0,1 act-fsdp2.sh || true
if run_job dit-1 0 dit-1.sh; then
  run_job dit-ddp2 0,1 dit-ddp2.sh || true
fi
if run_job smolvla-1 0 smolvla-1.sh; then
  run_job smolvla-ddp2 0,1 smolvla-ddp2.sh || true
fi

write_status "finished" "$(paste -sd '; ' "$ROOT/results.txt" 2>/dev/null || echo none)"
echo "queue end $(date -Is)"
