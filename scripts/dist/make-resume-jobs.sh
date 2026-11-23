#!/bin/bash
# Write the distributed resume jobs into $ROOT/jobs: train 2 steps on 2 GPUs, then resume to step 4
# from the step-2 checkpoint with the same topology, and require checkpoints/000004.
set -euo pipefail
ROOT=${ROOT:-/data/lerobot/outputs/rocm-train}

DIT_ARGS="--policy.type=multi_task_dit --policy.device=cuda --policy.push_to_hub=false \
  --policy.horizon=32 --policy.n_action_steps=24 --policy.objective=diffusion --policy.num_inference_steps=10 \
  --dataset.repo_id=lerobot/aloha_sim_transfer_cube_human --dataset.episodes='[0]'"
SMOLVLA_ARGS="--policy.path=lerobot/smolvla_base --policy.device=cuda --policy.push_to_hub=false \
  --dataset.repo_id=lerobot/svla_so100_pickplace --dataset.episodes='[0]' \
  --rename_map='{\"observation.images.top\": \"observation.images.camera1\", \"observation.images.wrist\": \"observation.images.camera2\"}'"
ACT_ARGS="--policy.type=act --policy.dim_model=64 --policy.n_action_steps=20 --policy.chunk_size=20 \
  --policy.device=cuda --policy.push_to_hub=false \
  --dataset.repo_id=lerobot/aloha_sim_transfer_cube_human --dataset.episodes='[0]'"
COMMON="--batch_size=1 --steps=2 --env_eval_freq=0 --save_freq=2 --log_freq=1 --num_workers=0 --trackio.enable=false"

write_job() {
  local name=$1 args=$2 par=$3 entry=${4:-'"$(which lerobot-train)"'} pre=${5:-}
  local out=/outputs/rocm-train/$name
  cat >"$ROOT/jobs/$name.sh" <<EOF
#!/bin/bash
set -euo pipefail
$pre
rm -rf $out
echo "== seed: 2 steps"
accelerate launch --num_processes=2 $entry $args $par $COMMON --output_dir=$out
test -d $out/checkpoints/000002
echo "== resume: steps 3-4"
accelerate launch --num_processes=2 $entry --resume=true \\
  --config_path=$out/checkpoints/000002/pretrained_model/train_config.json --steps=4
test -d $out/checkpoints/000004
find $out/checkpoints/000004 -maxdepth 2 | sort
cat $out/checkpoints/000004/training_state/training_step.json 2>/dev/null || true
echo "RESUME_OK"
EOF
  chmod +x "$ROOT/jobs/$name.sh"
}

write_job dit-resume-ddp2 "$DIT_ARGS" ""
AUTOWRAP=/outputs/rocm-train/jobs/fsdp_autowrap_train.py
write_job dit-resume-fsdp2 "$DIT_ARGS" "--parallelism.dp_shard=2" "$AUTOWRAP"
# SmolVLA runs its language layers through submodules, so only the vision encoder layers are units.
write_job smolvla-resume-fsdp2 "$SMOLVLA_ARGS" \
  "--parallelism.dp_shard=2 --accelerator.mixed_precision=bf16 --accelerator.fsdp.wrap_modules='[\"SmolVLMEncoderLayer\"]'" \
  "$AUTOWRAP" "export FSDP_CAST_DTYPE=float32"
# ACT declares its wrap units and uses every parameter: a control for the FSDP resume path.
write_job act-resume-fsdp2 "$ACT_ARGS" "--parallelism.dp_shard=2"
write_job act-resume-ddp2 "$ACT_ARGS" ""
# EMA (sharded EMA state under FSDP), fp16 loss-scaler state, and gradient accumulation across resume.
write_job act-ema-resume-ddp2 "$ACT_ARGS" "--ema.type=constant"
write_job act-ema-resume-fsdp2 "$ACT_ARGS" "--parallelism.dp_shard=2 --ema.type=constant"
write_job act-fp16-resume-ddp2 "$ACT_ARGS" "--accelerator.mixed_precision=fp16"
write_job act-gradacc-resume-fsdp2 "$ACT_ARGS" "--parallelism.dp_shard=2 --accelerator.gradient_accumulation.steps=2"
echo "resume jobs written"
