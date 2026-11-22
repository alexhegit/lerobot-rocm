#!/bin/bash
# Next ROCm checks: resume, 8-GPU, in-training eval, dataloader, pi0 compile.
# Append-only. Does not delete the resume seed until resume finishes.
set -u
ROOT=/data/lerobot/outputs/rocm-train
LOG="$ROOT/run-next.log"
RESULTS="$ROOT/results.txt"
mkdir -p "$ROOT/jobs"
exec >>"$LOG" 2>&1
echo "next begin $(date -Is)"

free_weights() {
  local host_out="$1"
  find "$host_out" -type d -name training_state -prune -exec rm -rf {} + 2>/dev/null || true
  find "$host_out" -type f \( -name '*.safetensors' -o -name '*.distcp' \) -delete 2>/dev/null || true
}

run_job() {
  local name="$1"
  local devices="$2"
  local script="$3"
  local host_out="$ROOT/$4"
  local step_dir="$5"
  echo "----- $(date -Is) BEGIN ${name} devices=${devices} -----"
  echo "disk_free_gb=$(df -BG / | awk 'NR==2 {print $4}')"
  local started ec elapsed verdict has_ckpt=0
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
  ec=$?
  elapsed=$(( $(date +%s) - started ))
  local ckpt="$host_out/checkpoints/${step_dir}/pretrained_model/model.safetensors"
  local stepf="$host_out/checkpoints/${step_dir}/training_state/training_step.json"
  if [[ -f "$ckpt" || -f "$stepf" ]]; then has_ckpt=1; fi
  if find "$host_out/checkpoints/${step_dir}" -name '*.distcp' 2>/dev/null | grep -q .; then has_ckpt=1; fi
  verdict="FAIL"
  if [[ "$ec" -eq 124 ]]; then
    verdict="TIMEOUT"
    docker rm -f "rocm-${name}" >/dev/null 2>&1 || true
  elif [[ "$ec" -eq 0 && "$has_ckpt" -eq 1 ]]; then
    verdict="PASS"
  fi
  echo "RESULT ${name} ${verdict} exit=${ec} seconds=${elapsed}"
  if [[ -f "$stepf" ]]; then echo "STEP $(tr -d '\n' <"$stepf")"; else echo "STEP missing"; fi
  echo "${name} ${verdict} exit=${ec} seconds=${elapsed}" >>"$RESULTS"
  echo "----- $(date -Is) END ${name} -----"
  [[ "$verdict" == "PASS" ]]
}

cat >"$ROOT/jobs/dit-resume-seed.sh" <<'EOF'
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
  --output_dir=/outputs/rocm-train/dit-resume-seed
EOF

cat >"$ROOT/jobs/dit-resume.sh" <<'EOF'
#!/bin/bash
set -euo pipefail
lerobot-train \
  --resume=true \
  --config_path=/outputs/rocm-train/dit-resume-seed/checkpoints/000002/pretrained_model/train_config.json \
  --steps=4
EOF

cat >"$ROOT/jobs/diffusion-ddp8.sh" <<'EOF'
#!/bin/bash
set -euo pipefail
accelerate launch --num_processes=8 "$(which lerobot-train)" \
  --policy.type=diffusion \
  --policy.down_dims='[64,128,256]' \
  --policy.diffusion_step_embed_dim=32 \
  --policy.num_inference_steps=10 \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --dataset.repo_id=lerobot/aloha_sim_transfer_cube_human \
  --dataset.episodes='[0]' \
  --batch_size=1 \
  --steps=2 \
  --env_eval_freq=0 \
  --save_freq=2 \
  --log_freq=1 \
  --num_workers=0 \
  --trackio.enable=false \
  --output_dir=/outputs/rocm-train/diffusion-ddp8
EOF

cat >"$ROOT/jobs/act-fsdp8.sh" <<'EOF'
#!/bin/bash
set -euo pipefail
accelerate launch --num_processes=8 "$(which lerobot-train)" \
  --policy.type=act \
  --policy.dim_model=64 \
  --policy.n_action_steps=20 \
  --policy.chunk_size=20 \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --parallelism.dp_shard=8 \
  --dataset.repo_id=lerobot/aloha_sim_transfer_cube_human \
  --dataset.episodes='[0]' \
  --batch_size=1 \
  --steps=2 \
  --env_eval_freq=0 \
  --save_freq=2 \
  --log_freq=1 \
  --num_workers=0 \
  --trackio.enable=false \
  --output_dir=/outputs/rocm-train/act-fsdp8
EOF

cat >"$ROOT/jobs/dit-eval.sh" <<'EOF'
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
  --env.type=aloha \
  --env_eval_freq=2 \
  --eval.n_episodes=1 \
  --eval.batch_size=1 \
  --eval.max_episode_steps=2 \
  --batch_size=1 \
  --steps=2 \
  --save_freq=2 \
  --log_freq=1 \
  --num_workers=0 \
  --trackio.enable=false \
  --output_dir=/outputs/rocm-train/dit-eval
EOF

cat >"$ROOT/jobs/dit-workers.sh" <<'EOF'
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
  --steps=20 \
  --env_eval_freq=0 \
  --save_freq=20 \
  --log_freq=5 \
  --num_workers=2 \
  --trackio.enable=false \
  --output_dir=/outputs/rocm-train/dit-workers
EOF

cat >"$ROOT/jobs/pi0-compile-1.sh" <<'EOF'
#!/bin/bash
set -euo pipefail
lerobot-train \
  --policy.type=pi0 \
  --policy.pretrained_path=lerobot/pi0_base \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --policy.dtype=bfloat16 \
  --policy.empty_cameras=1 \
  --policy.gradient_checkpointing=true \
  --policy.compile_model=true \
  --dataset.repo_id=lerobot/svla_so100_pickplace \
  --dataset.episodes='[0]' \
  --rename_map='{"observation.images.top":"observation.images.base_0_rgb","observation.images.wrist":"observation.images.left_wrist_0_rgb"}' \
  --batch_size=1 \
  --steps=2 \
  --env_eval_freq=0 \
  --save_freq=2 \
  --log_freq=1 \
  --num_workers=0 \
  --trackio.enable=false \
  --output_dir=/outputs/rocm-train/pi0-compile-1
EOF
chmod +x "$ROOT/jobs/"*.sh

rm -rf "$ROOT/dit-resume-seed"
if run_job dit-resume-seed 0 dit-resume-seed.sh dit-resume-seed 000002; then
  run_job dit-resume 0 dit-resume.sh dit-resume-seed 000004 || true
else
  echo "RESULT dit-resume SKIP exit= seed-failed"
  echo "dit-resume SKIP seed-failed" >>"$RESULTS"
fi
free_weights "$ROOT/dit-resume-seed"

rm -rf "$ROOT/diffusion-ddp8"
run_job diffusion-ddp8 0,1,2,3,4,5,6,7 diffusion-ddp8.sh diffusion-ddp8 000002 || true
free_weights "$ROOT/diffusion-ddp8"

rm -rf "$ROOT/act-fsdp8"
run_job act-fsdp8 0,1,2,3,4,5,6,7 act-fsdp8.sh act-fsdp8 000002 || true
free_weights "$ROOT/act-fsdp8"

rm -rf "$ROOT/dit-eval"
run_job dit-eval 0 dit-eval.sh dit-eval 000002 || true
free_weights "$ROOT/dit-eval"

rm -rf "$ROOT/dit-workers"
run_job dit-workers 0 dit-workers.sh dit-workers 000020 || true
free_weights "$ROOT/dit-workers"

rm -rf "$ROOT/pi0-compile-1"
run_job pi0-compile-1 0 pi0-compile-1.sh pi0-compile-1 000002 || true
free_weights "$ROOT/pi0-compile-1"

echo "next end $(date -Is)"
