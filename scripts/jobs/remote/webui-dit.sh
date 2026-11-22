#!/bin/bash
set -u
LOG=/data/lerobot/outputs/rocm-train/webui-dit.log
exec >"$LOG" 2>&1
echo "webui-dit begin $(date -Is)"
rm -rf /data/lerobot/outputs/rocm-train/webui-dit
docker rm -f rocm-webui-dit >/dev/null 2>&1 || true
docker run --name rocm-webui-dit \
  --ipc=host --shm-size 16g \
  --device=/dev/kfd --device=/dev/dri \
  --group-add 44 --group-add 992 \
  -e HIP_VISIBLE_DEVICES=0 \
  -e MIOPEN_FIND_MODE=2 \
  -v /data/lerobot/outputs:/outputs \
  -v /root/.cache/huggingface:/root/.cache/huggingface \
  lerobot-rocm \
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
  --output_dir=/outputs/rocm-train/webui-dit
ec=$?
echo "EXIT:${ec}"
docker rm -f rocm-webui-dit >/dev/null 2>&1 || true
