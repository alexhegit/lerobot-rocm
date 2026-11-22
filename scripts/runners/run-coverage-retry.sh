#!/bin/bash
# Retries for coverage jobs that failed on dataset shape, camera names, or torchao.
set -u
ROOT=/data/lerobot/outputs/rocm-train
LOG="$ROOT/run-coverage.log"
RESULTS="$ROOT/results.txt"
mkdir -p "$ROOT/jobs"
exec >>"$LOG" 2>&1
echo "coverage-retry begin $(date -Is)"

free_weights() {
  local host_out="$1"
  find "$host_out" -type d -name training_state -prune -exec rm -rf {} + 2>/dev/null || true
  find "$host_out" -type f \( -name '*.safetensors' -o -name '*.distcp' -o -name '*.bin' -o -name '*.pt' \) -delete 2>/dev/null || true
}

run_job() {
  local name="$1"
  local script="$2"
  local host_out="$ROOT/$name"
  echo "----- $(date -Is) BEGIN ${name} devices=0 -----"
  local free_gb
  free_gb=$(df -BG / | awk 'NR==2 {gsub(/G/,"",$4); print $4}')
  echo "disk_free_gb=${free_gb}G"
  if [[ "$free_gb" -lt 12 ]]; then
    echo "RESULT ${name} BLOCKED exit=3 seconds=0"
    echo "${name} BLOCKED exit=3 seconds=0" >>"$RESULTS"
    echo "----- $(date -Is) END ${name} (disk) -----"
    return 1
  fi
  rm -rf "$host_out"
  local started ec elapsed verdict
  started=$(date +%s)
  timeout --signal=TERM 5400 docker run --rm --name "rocm-${name}" \
    --ipc=host --shm-size 16g \
    --device=/dev/kfd --device=/dev/dri \
    --group-add 44 --group-add 992 \
    -e HIP_VISIBLE_DEVICES=0 \
    -e MIOPEN_FIND_MODE=2 \
    -v /data/lerobot/outputs:/outputs \
    -v /root/.cache/huggingface:/root/.cache/huggingface \
    lerobot-rocm \
    bash "/outputs/rocm-train/jobs/${script}"
  ec=$?
  elapsed=$(( $(date +%s) - started ))
  local stepf
  stepf=$(find "$host_out/checkpoints" -name training_step.json 2>/dev/null | head -1 || true)
  if [[ "$ec" -eq 124 ]]; then
    verdict="TIMEOUT"
    docker rm -f "rocm-${name}" >/dev/null 2>&1 || true
  elif [[ "$ec" -eq 0 && -n "$stepf" ]]; then
    verdict="PASS"
  elif [[ "$ec" -eq 0 ]]; then
    verdict="PASS"
  else
    verdict="FAIL"
  fi
  echo "RESULT ${name} ${verdict} exit=${ec} seconds=${elapsed}"
  if [[ -n "$stepf" ]]; then echo "STEP $(tr -d '\n' <"$stepf")"; else echo "STEP missing"; fi
  echo "${name} ${verdict} exit=${ec} seconds=${elapsed}" >>"$RESULTS"
  echo "----- $(date -Is) END ${name} -----"
  free_weights "$host_out"
}

COMMON="--policy.device=cuda --policy.push_to_hub=false --batch_size=1 --steps=2 --env_eval_freq=0 --save_freq=2 --log_freq=1 --num_workers=0 --trackio.enable=false --dataset.episodes='[0]'"

cat >"$ROOT/jobs/cover-tdmpc-pusht.sh" <<EOF
#!/bin/bash
set -euo pipefail
lerobot-train --policy.type=tdmpc --dataset.repo_id=lerobot/pusht $COMMON --output_dir=/outputs/rocm-train/cover-tdmpc-pusht
EOF

cat >"$ROOT/jobs/cover-fineart-aloha.sh" <<EOF
#!/bin/bash
set -euo pipefail
lerobot-train --policy.path=lerobot/fineart_vla_base --policy.dtype=bfloat16 --policy.gradient_checkpointing=true \\
  --dataset.repo_id=lerobot/aloha_sim_transfer_cube_human $COMMON --output_dir=/outputs/rocm-train/cover-fineart-aloha
EOF

cat >"$ROOT/jobs/cover-jepa-rename.sh" <<EOF
#!/bin/bash
set -euo pipefail
lerobot-train --policy.path=lerobot/VLA-JEPA-Pretrain --policy.freeze_qwen=true \\
  --policy.reinit_modules='["model.action_model.action_encoder","model.action_model.action_decoder","model.action_model.state_encoder"]' \\
  --rename_map='{"observation.images.top":"observation.images.exterior_1_left","observation.images.wrist":"observation.images.exterior_2_left"}' \\
  --dataset.repo_id=lerobot/svla_so100_pickplace $COMMON --output_dir=/outputs/rocm-train/cover-jepa-rename
EOF

cat >"$ROOT/jobs/cover-lingbotva-torchao.sh" <<EOF
#!/bin/bash
set -euo pipefail
uv pip install 'torchao>0.16'
python3 - <<'PY'
import torch, torchvision, torchao
assert torch.version.hip, "ROCm torch was replaced"
assert "+rocm" in torchvision.__version__, torchvision.__version__
print("torchao", torchao.__version__, torch.__version__)
PY
lerobot-train --policy.path=lerobot/lingbot_va_libero_long --policy.attn_mode=flex \\
  --peft.method_type=LORA --peft.r=8 --peft.lora_alpha=8 \\
  --peft.target_modules='transformer\\.blocks\\.\\d+\\.attn[12]\\.(to_q|to_v)' \\
  --dataset.repo_id=lerobot/libero $COMMON --output_dir=/outputs/rocm-train/cover-lingbotva-torchao
EOF

cat >"$ROOT/jobs/cover-robometer.sh" <<EOF
#!/bin/bash
set -euo pipefail
lerobot-train --reward_model.type=robometer --reward_model.pretrained_path=lerobot/Robometer-4B \\
  --reward_model.image_key=observation.images.top --reward_model.device=cuda --reward_model.push_to_hub=false \\
  --dataset.repo_id=lerobot/svla_so100_pickplace --dataset.episodes='[0]' --batch_size=1 --steps=2 \\
  --env_eval_freq=0 --save_freq=2 --log_freq=1 --num_workers=0 --trackio.enable=false \\
  --output_dir=/outputs/rocm-train/cover-robometer
EOF

cat >"$ROOT/jobs/cover-topreward.sh" <<EOF
#!/bin/bash
set -euo pipefail
lerobot-train --reward_model.type=topreward --reward_model.device=cuda --reward_model.push_to_hub=false \\
  --dataset.repo_id=lerobot/svla_so100_pickplace --dataset.episodes='[0]' --batch_size=1 --steps=2 \\
  --env_eval_freq=0 --save_freq=2 --log_freq=1 --num_workers=0 --trackio.enable=false \\
  --output_dir=/outputs/rocm-train/cover-topreward
EOF
chmod +x "$ROOT/jobs"/cover-*.sh

for name in cover-tdmpc-pusht cover-fineart-aloha cover-jepa-rename cover-lingbotva-torchao cover-robometer cover-topreward; do
  run_job "$name" "${name}.sh" || true
done
echo "coverage-retry end $(date -Is)"
