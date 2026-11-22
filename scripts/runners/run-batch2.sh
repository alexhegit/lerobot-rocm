#!/bin/bash
# Second-batch Hub-finetune smokes for lerobot-rocm on DOUni.
set -u
ROOT=/data/lerobot/outputs/rocm-train
LOG="$ROOT/run-batch2.log"
STATUS="$ROOT/STATUS.md"
RESULTS="$ROOT/results.txt"
mkdir -p "$ROOT/jobs"
: >"$LOG"
exec >>"$LOG" 2>&1

# Archive batch-1 results if not already copied.
if [[ -f "$RESULTS" && ! -f "$ROOT/results-batch1.txt" ]]; then
  cp "$RESULTS" "$ROOT/results-batch1.txt"
fi

write_status() {
  local state="$1"
  local note="$2"
  cat >"$STATUS" <<EOF
# rocm-train smoke

- updated: $(date -Is)
- batch: 2
- state: ${state}
- note: ${note}

Batch1 archive: ${ROOT}/results-batch1.txt
Batch2 log: ${LOG}
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
  if [[ "$verdict" == "PASS" ]]; then
    return 0
  fi
  return 1
}

write_status "starting" "batch2 queue"
echo "batch2 begin $(date -Is)"

# Shared SO100 camera remaps.
PI_RENAME='{"observation.images.top":"observation.images.base_0_rgb","observation.images.wrist":"observation.images.left_wrist_0_rgb"}'
XVLA_RENAME='{"observation.images.top":"observation.images.image","observation.images.wrist":"observation.images.image2"}'

cat >"$ROOT/jobs/xvla-1.sh" <<EOF
#!/bin/bash
set -euo pipefail
lerobot-train \\
  --policy.path=lerobot/xvla-base \\
  --policy.device=cuda \\
  --policy.push_to_hub=false \\
  --policy.dtype=bfloat16 \\
  --policy.empty_cameras=1 \\
  --dataset.repo_id=lerobot/svla_so100_pickplace \\
  --dataset.episodes='[0]' \\
  --rename_map='${XVLA_RENAME}' \\
  --batch_size=1 \\
  --steps=2 \\
  --env_eval_freq=0 \\
  --save_freq=2 \\
  --log_freq=1 \\
  --num_workers=0 \\
  --trackio.enable=false \\
  --output_dir=/outputs/rocm-train/xvla-1
EOF

cat >"$ROOT/jobs/xvla-ddp2.sh" <<EOF
#!/bin/bash
set -euo pipefail
accelerate launch --num_processes=2 "\$(which lerobot-train)" \\
  --policy.path=lerobot/xvla-base \\
  --policy.device=cuda \\
  --policy.push_to_hub=false \\
  --policy.dtype=bfloat16 \\
  --policy.empty_cameras=1 \\
  --dataset.repo_id=lerobot/svla_so100_pickplace \\
  --dataset.episodes='[0]' \\
  --rename_map='${XVLA_RENAME}' \\
  --batch_size=1 \\
  --steps=2 \\
  --env_eval_freq=0 \\
  --save_freq=2 \\
  --log_freq=1 \\
  --num_workers=0 \\
  --trackio.enable=false \\
  --output_dir=/outputs/rocm-train/xvla-ddp2
EOF

cat >"$ROOT/jobs/g05-1.sh" <<'EOF'
#!/bin/bash
set -euo pipefail
lerobot-train \
  --policy.path=lerobot/g05_base \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --policy.embodiment=so100 \
  --policy.camera_order='["observation.images.top", null, "observation.images.wrist"]' \
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
  --policy.path=lerobot/g05_base \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --policy.embodiment=so100 \
  --policy.camera_order='["observation.images.top", null, "observation.images.wrist"]' \
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

cat >"$ROOT/jobs/pi0fast-1.sh" <<EOF
#!/bin/bash
set -euo pipefail
lerobot-train \\
  --policy.type=pi0_fast \\
  --policy.pretrained_path=lerobot/pi0fast-base \\
  --policy.device=cuda \\
  --policy.push_to_hub=false \\
  --policy.dtype=bfloat16 \\
  --policy.empty_cameras=1 \\
  --policy.compile_model=false \\
  --dataset.repo_id=lerobot/svla_so100_pickplace \\
  --dataset.episodes='[0]' \\
  --rename_map='${PI_RENAME}' \\
  --batch_size=1 \\
  --steps=2 \\
  --env_eval_freq=0 \\
  --save_freq=2 \\
  --log_freq=1 \\
  --num_workers=0 \\
  --trackio.enable=false \\
  --output_dir=/outputs/rocm-train/pi0fast-1
EOF

cat >"$ROOT/jobs/pi0fast-ddp2.sh" <<EOF
#!/bin/bash
set -euo pipefail
accelerate launch --num_processes=2 "\$(which lerobot-train)" \\
  --policy.type=pi0_fast \\
  --policy.pretrained_path=lerobot/pi0fast-base \\
  --policy.device=cuda \\
  --policy.push_to_hub=false \\
  --policy.dtype=bfloat16 \\
  --policy.empty_cameras=1 \\
  --policy.compile_model=false \\
  --dataset.repo_id=lerobot/svla_so100_pickplace \\
  --dataset.episodes='[0]' \\
  --rename_map='${PI_RENAME}' \\
  --batch_size=1 \\
  --steps=2 \\
  --env_eval_freq=0 \\
  --save_freq=2 \\
  --log_freq=1 \\
  --num_workers=0 \\
  --trackio.enable=false \\
  --output_dir=/outputs/rocm-train/pi0fast-ddp2
EOF

cat >"$ROOT/jobs/pi0-1.sh" <<EOF
#!/bin/bash
set -euo pipefail
lerobot-train \\
  --policy.type=pi0 \\
  --policy.pretrained_path=lerobot/pi0_base \\
  --policy.device=cuda \\
  --policy.push_to_hub=false \\
  --policy.dtype=bfloat16 \\
  --policy.empty_cameras=1 \\
  --policy.gradient_checkpointing=true \\
  --policy.compile_model=false \\
  --dataset.repo_id=lerobot/svla_so100_pickplace \\
  --dataset.episodes='[0]' \\
  --rename_map='${PI_RENAME}' \\
  --batch_size=1 \\
  --steps=2 \\
  --env_eval_freq=0 \\
  --save_freq=2 \\
  --log_freq=1 \\
  --num_workers=0 \\
  --trackio.enable=false \\
  --output_dir=/outputs/rocm-train/pi0-1
EOF

cat >"$ROOT/jobs/pi0-ddp2.sh" <<EOF
#!/bin/bash
set -euo pipefail
accelerate launch --num_processes=2 "\$(which lerobot-train)" \\
  --policy.type=pi0 \\
  --policy.pretrained_path=lerobot/pi0_base \\
  --policy.device=cuda \\
  --policy.push_to_hub=false \\
  --policy.dtype=bfloat16 \\
  --policy.empty_cameras=1 \\
  --policy.gradient_checkpointing=true \\
  --policy.compile_model=false \\
  --dataset.repo_id=lerobot/svla_so100_pickplace \\
  --dataset.episodes='[0]' \\
  --rename_map='${PI_RENAME}' \\
  --batch_size=1 \\
  --steps=2 \\
  --env_eval_freq=0 \\
  --save_freq=2 \\
  --log_freq=1 \\
  --num_workers=0 \\
  --trackio.enable=false \\
  --output_dir=/outputs/rocm-train/pi0-ddp2
EOF

cat >"$ROOT/jobs/pi05-1.sh" <<EOF
#!/bin/bash
set -euo pipefail
lerobot-train \\
  --policy.type=pi05 \\
  --policy.pretrained_path=lerobot/pi05_base \\
  --policy.device=cuda \\
  --policy.push_to_hub=false \\
  --policy.dtype=bfloat16 \\
  --policy.empty_cameras=1 \\
  --policy.gradient_checkpointing=true \\
  --policy.compile_model=false \\
  --dataset.repo_id=lerobot/svla_so100_pickplace \\
  --dataset.episodes='[0]' \\
  --rename_map='${PI_RENAME}' \\
  --batch_size=1 \\
  --steps=2 \\
  --env_eval_freq=0 \\
  --save_freq=2 \\
  --log_freq=1 \\
  --num_workers=0 \\
  --trackio.enable=false \\
  --output_dir=/outputs/rocm-train/pi05-1
EOF

cat >"$ROOT/jobs/pi05-ddp2.sh" <<EOF
#!/bin/bash
set -euo pipefail
accelerate launch --num_processes=2 "\$(which lerobot-train)" \\
  --policy.type=pi05 \\
  --policy.pretrained_path=lerobot/pi05_base \\
  --policy.device=cuda \\
  --policy.push_to_hub=false \\
  --policy.dtype=bfloat16 \\
  --policy.empty_cameras=1 \\
  --policy.gradient_checkpointing=true \\
  --policy.compile_model=false \\
  --dataset.repo_id=lerobot/svla_so100_pickplace \\
  --dataset.episodes='[0]' \\
  --rename_map='${PI_RENAME}' \\
  --batch_size=1 \\
  --steps=2 \\
  --env_eval_freq=0 \\
  --save_freq=2 \\
  --log_freq=1 \\
  --num_workers=0 \\
  --trackio.enable=false \\
  --output_dir=/outputs/rocm-train/pi05-ddp2
EOF

cat >"$ROOT/jobs/eo1-1.sh" <<'EOF'
#!/bin/bash
set -euo pipefail
lerobot-train \
  --policy.path=lerobot/eo1-base \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --policy.dtype=bfloat16 \
  --policy.attn_implementation=sdpa \
  --dataset.repo_id=lerobot/svla_so100_pickplace \
  --dataset.episodes='[0]' \
  --batch_size=1 \
  --steps=2 \
  --env_eval_freq=0 \
  --save_freq=2 \
  --log_freq=1 \
  --num_workers=0 \
  --trackio.enable=false \
  --output_dir=/outputs/rocm-train/eo1-1
EOF

cat >"$ROOT/jobs/eo1-ddp2.sh" <<'EOF'
#!/bin/bash
set -euo pipefail
accelerate launch --num_processes=2 "$(which lerobot-train)" \
  --policy.path=lerobot/eo1-base \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --policy.dtype=bfloat16 \
  --policy.attn_implementation=sdpa \
  --dataset.repo_id=lerobot/svla_so100_pickplace \
  --dataset.episodes='[0]' \
  --batch_size=1 \
  --steps=2 \
  --env_eval_freq=0 \
  --save_freq=2 \
  --log_freq=1 \
  --num_workers=0 \
  --trackio.enable=false \
  --output_dir=/outputs/rocm-train/eo1-ddp2
EOF

chmod +x "$ROOT/jobs/"*.sh

if run_job xvla-1 0 xvla-1.sh; then
  run_job xvla-ddp2 0,1 xvla-ddp2.sh || true
fi
if run_job g05-1 0 g05-1.sh; then
  run_job g05-ddp2 0,1 g05-ddp2.sh || true
fi
if run_job pi0fast-1 0 pi0fast-1.sh; then
  run_job pi0fast-ddp2 0,1 pi0fast-ddp2.sh || true
fi
if run_job pi0-1 0 pi0-1.sh; then
  run_job pi0-ddp2 0,1 pi0-ddp2.sh || true
fi
if run_job pi05-1 0 pi05-1.sh; then
  run_job pi05-ddp2 0,1 pi05-ddp2.sh || true
fi
if run_job eo1-1 0 eo1-1.sh; then
  run_job eo1-ddp2 0,1 eo1-ddp2.sh || true
fi

write_status "finished" "$(grep -E '^(xvla|g05|pi0|pi05|eo1)' "$RESULTS" | paste -sd '; ' - || echo none)"
echo "batch2 end $(date -Is)"
