#!/bin/bash
set -euo pipefail
OUT=/outputs/rocm-train/dit-fsdp2-deploy
rm -rf "$OUT"
accelerate launch --num_processes=2 /outputs/rocm-train/jobs/fsdp_autowrap_train.py --policy.type=multi_task_dit --policy.device=cuda --policy.push_to_hub=false   --policy.horizon=32 --policy.n_action_steps=24 --policy.objective=diffusion --policy.num_inference_steps=10   --dataset.repo_id=lerobot/aloha_sim_transfer_cube_human --dataset.episodes='[0]' --parallelism.dp_shard=2   --batch_size=1 --steps=2 --env_eval_freq=0 --save_freq=2 --log_freq=1 --num_workers=0 --trackio.enable=false --output_dir=$OUT
CUDA_VISIBLE_DEVICES=0 HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES%%,*} python3 - $OUT/checkpoints/000002/pretrained_model <<'PY'
import sys
import torch
from lerobot.datasets import LeRobotDataset
from lerobot.policies.factory import get_policy_class, make_pre_post_processors

path = sys.argv[1]
policy = get_policy_class('multi_task_dit').from_pretrained(path).to('cuda').eval()
pre, post = make_pre_post_processors(policy.config, pretrained_path=path)
item = LeRobotDataset('lerobot/aloha_sim_transfer_cube_human', episodes=[0])[0]
obs = {k: v for k, v in item.items() if k.startswith('observation') or k == 'task'}
with torch.no_grad():
    action = post(policy.select_action(pre(obs)))
print('DEPLOY_OK params', sum(p.numel() for p in policy.parameters()), 'action', tuple(action.shape), float(action.abs().mean()))
PY
