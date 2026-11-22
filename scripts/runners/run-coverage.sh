#!/bin/bash
# Coverage smokes for policies not in the earlier ROCm queue.
# Single GPU, 2 steps, push_to_hub false. One failure does not stop the queue.
set -u
ROOT=/data/lerobot/outputs/rocm-train
LOG="$ROOT/run-coverage.log"
RESULTS="$ROOT/results.txt"
mkdir -p "$ROOT/jobs"
exec >>"$LOG" 2>&1
echo "coverage begin $(date -Is)"

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
  if [[ "$free_gb" -lt 20 ]]; then
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
  local has_loss=0
  if awk "/BEGIN ${name} /{p=1} p" "$LOG" | grep -E -q 'loss|Loss'; then has_loss=1; fi
  if [[ "$ec" -eq 124 ]]; then
    verdict="TIMEOUT"
    docker rm -f "rocm-${name}" >/dev/null 2>&1 || true
  elif [[ "$ec" -eq 0 && ( -n "$stepf" || "$has_loss" -eq 1 ) ]]; then
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

write_job() {
  local name="$1"
  cat >"$ROOT/jobs/${name}.sh"
  chmod +x "$ROOT/jobs/${name}.sh"
}

COMMON="--policy.device=cuda --policy.push_to_hub=false --batch_size=1 --steps=2 --env_eval_freq=0 --save_freq=2 --log_freq=1 --num_workers=0 --trackio.enable=false --dataset.episodes='[0]'"

write_job cover-vqbet <<EOF
#!/bin/bash
set -euo pipefail
lerobot-train --policy.type=vqbet --dataset.repo_id=lerobot/aloha_sim_transfer_cube_human $COMMON --output_dir=/outputs/rocm-train/cover-vqbet
EOF

write_job cover-tdmpc <<EOF
#!/bin/bash
set -euo pipefail
lerobot-train --policy.type=tdmpc --dataset.repo_id=lerobot/aloha_sim_transfer_cube_human $COMMON --output_dir=/outputs/rocm-train/cover-tdmpc
EOF

write_job cover-gaussian <<EOF
#!/bin/bash
set -euo pipefail
lerobot-train --policy.type=gaussian_actor --dataset.repo_id=lerobot/svla_so100_pickplace $COMMON --output_dir=/outputs/rocm-train/cover-gaussian
EOF

write_job cover-dm05 <<EOF
#!/bin/bash
set -euo pipefail
lerobot-train --policy.path=lerobot/dm05_base --policy.add_state=false --policy.chunk_size=10 --policy.n_action_steps=10 \\
  --rename_map='{"observation.images.image":"observation.images.front","observation.images.image2":"observation.images.wrist"}' \\
  --dataset.repo_id=lerobot/libero $COMMON --output_dir=/outputs/rocm-train/cover-dm05
EOF

write_job cover-evo1 <<EOF
#!/bin/bash
set -euo pipefail
lerobot-train --policy.type=evo1 --policy.training_stage=stage1 --policy.vlm_model_name=OpenGVLab/InternVL3-1B-hf \\
  --policy.chunk_size=16 --policy.n_action_steps=16 --policy.max_state_dim=24 --policy.max_action_dim=24 \\
  --dataset.repo_id=lerobot/svla_so100_pickplace $COMMON --output_dir=/outputs/rocm-train/cover-evo1
EOF

write_job cover-sarm <<EOF
#!/bin/bash
set -euo pipefail
lerobot-train --reward_model.type=sarm --reward_model.annotation_mode=single_stage \\
  --reward_model.image_key=observation.images.top --reward_model.device=cuda --reward_model.push_to_hub=false \\
  --dataset.repo_id=lerobot/svla_so100_pickplace --dataset.episodes='[0]' --batch_size=1 --steps=2 \\
  --env_eval_freq=0 --save_freq=2 --log_freq=1 --num_workers=0 --trackio.enable=false \\
  --output_dir=/outputs/rocm-train/cover-sarm
EOF

write_job cover-wallx <<EOF
#!/bin/bash
set -euo pipefail
lerobot-train --policy.type=wall_x --policy.pretrained_name_or_path=x-square-robot/wall-oss-flow \\
  --policy.prediction_mode=diffusion --policy.attn_implementation=eager --policy.recipe=null \\
  --dataset.repo_id=lerobot/svla_so100_pickplace $COMMON --output_dir=/outputs/rocm-train/cover-wallx
EOF

write_job cover-fineart <<EOF
#!/bin/bash
set -euo pipefail
lerobot-train --policy.path=lerobot/fineart_vla_base --policy.dtype=bfloat16 --policy.gradient_checkpointing=true \\
  --dataset.repo_id=lerobot/svla_so100_pickplace $COMMON --output_dir=/outputs/rocm-train/cover-fineart
EOF

write_job cover-jepa <<EOF
#!/bin/bash
set -euo pipefail
lerobot-train --policy.path=lerobot/VLA-JEPA-Pretrain --policy.freeze_qwen=true \\
  --policy.reinit_modules='["model.action_model.action_encoder","model.action_model.action_decoder","model.action_model.state_encoder"]' \\
  --dataset.repo_id=lerobot/svla_so100_pickplace $COMMON --output_dir=/outputs/rocm-train/cover-jepa
EOF

write_job cover-longwam <<EOF
#!/bin/bash
set -euo pipefail
lerobot-train --policy.path=Efficient-Large-Model/Long-WAM-LIBERO-IDM-LeRobot --policy.dtype=bfloat16 \\
  --accelerator.mixed_precision=bf16 --dataset.repo_id=lerobot/libero $COMMON \\
  --output_dir=/outputs/rocm-train/cover-longwam
EOF

write_job cover-groot <<EOF
#!/bin/bash
set -euo pipefail
lerobot-train --policy.type=groot --policy.base_model_path=nvidia/GR00T-N1.7-3B --policy.embodiment_tag=new_embodiment \\
  --policy.use_bf16=true --policy.use_relative_actions=false --policy.chunk_size=16 --policy.n_action_steps=16 \\
  --dataset.repo_id=lerobot/svla_so100_pickplace $COMMON --output_dir=/outputs/rocm-train/cover-groot
EOF

write_job cover-lingbotva <<EOF
#!/bin/bash
set -euo pipefail
lerobot-train --policy.path=lerobot/lingbot_va_libero_long --policy.attn_mode=flex \\
  --peft.method_type=LORA --peft.r=8 --peft.lora_alpha=8 \\
  --peft.target_modules='transformer\\.blocks\\.\\d+\\.attn[12]\\.(to_q|to_v)' \\
  --dataset.repo_id=lerobot/libero $COMMON --output_dir=/outputs/rocm-train/cover-lingbotva
EOF

write_job cover-robometer <<EOF
#!/bin/bash
set -euo pipefail
lerobot-train --reward_model.type=robometer --reward_model.pretrained_path=lerobot/Robometer-4B \\
  --reward_model.image_key=observation.images.top --reward_model.device=cuda --reward_model.push_to_hub=false \\
  --dataset.repo_id=lerobot/svla_so100_pickplace --dataset.episodes='[0]' --batch_size=1 --steps=2 \\
  --env_eval_freq=0 --save_freq=2 --log_freq=1 --num_workers=0 --trackio.enable=false \\
  --output_dir=/outputs/rocm-train/cover-robometer
EOF

write_job cover-topreward <<EOF
#!/bin/bash
set -euo pipefail
lerobot-train --reward_model.type=topreward --reward_model.device=cuda --reward_model.push_to_hub=false \\
  --dataset.repo_id=lerobot/svla_so100_pickplace --dataset.episodes='[0]' --batch_size=1 --steps=2 \\
  --env_eval_freq=0 --save_freq=2 --log_freq=1 --num_workers=0 --trackio.enable=false \\
  --output_dir=/outputs/rocm-train/cover-topreward
EOF

for name in cover-vqbet cover-tdmpc cover-gaussian cover-dm05 cover-evo1 cover-sarm cover-wallx cover-fineart cover-jepa cover-longwam cover-groot cover-lingbotva cover-robometer cover-topreward; do
  run_job "$name" "${name}.sh" || true
done
echo "coverage end $(date -Is)"
