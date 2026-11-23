#!/bin/bash
# N independent single-GPU ACT trainings at once (no DDP), one per visible GPU, to tell node-level
# contention apart from DDP synchronisation. Usage: concurrent-act.sh N
set -uo pipefail
N=$1
export MIOPEN_USER_DB_PATH=/outputs/rocm-train/miopen/fast/db
export MIOPEN_CUSTOM_CACHE_DIR=/outputs/rocm-train/miopen/fast/cache
GPUS=(${HIP_VISIBLE_DEVICES//,/ })
pids=()
for i in $(seq 0 $((N - 1))); do
  OUT=/outputs/rocm-train/concurrent-act-${N}x1-$i
  rm -rf "$OUT"
  HIP_VISIBLE_DEVICES=${GPUS[$i]} lerobot-train --policy.type=act --policy.dim_model=64 --policy.n_action_steps=20 \
    --policy.chunk_size=20 --policy.device=cuda --policy.push_to_hub=false \
    --dataset.repo_id=lerobot/aloha_sim_transfer_cube_human --batch_size=32 --steps=60 --log_freq=20 \
    --save_freq=100000 --env_eval_freq=0 --num_workers=8 --trackio.enable=false --output_dir="$OUT" \
    >"$OUT.log" 2>&1 &
  pids+=($!)
done
ec=0
for p in "${pids[@]}"; do wait "$p" || ec=1; done
for i in $(seq 0 $((N - 1))); do
  echo "CONCURRENT ${N}x1 gpu=${GPUS[$i]} $(grep -aoE 'step:60 .*' /outputs/rocm-train/concurrent-act-${N}x1-$i.log | grep -oE 'updt_s:[0-9.]+|smp/s:[0-9.]+' | tr '\n' ' ')"
done
exit $ec
