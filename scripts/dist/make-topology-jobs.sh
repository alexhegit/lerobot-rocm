#!/bin/bash
# Write checkpoint-topology jobs into $ROOT/jobs:
#   act-reshard-resume      FSDP 2 -> resume on FSDP 4 (DCP reshards the optimizer)
#   act-ddp2-to-1-resume    DDP 2  -> resume on 1 GPU
#   act-fsdp-to-ddp-reject  FSDP 2 -> resume on DDP 2 must fail with LeRobot's topology error
#   act-fsdp2-dcp           FSDP 2 with --checkpoint_format=dcp, then resume
#   act-fsdp2-stdcp         FSDP 2 with --checkpoint_format=safetensors_dcp, then resume
#   dit-fsdp2-deploy        FSDP 2 training, then load the checkpoint on 1 GPU and predict an action
set -euo pipefail
ROOT=${ROOT:-/data/lerobot/outputs/rocm-train}
TRAIN='"$(which lerobot-train)"'
ACT="--policy.type=act --policy.dim_model=64 --policy.n_action_steps=20 --policy.chunk_size=20 \
  --policy.device=cuda --policy.push_to_hub=false \
  --dataset.repo_id=lerobot/aloha_sim_transfer_cube_human --dataset.episodes='[0]'"
COMMON="--batch_size=2 --steps=2 --env_eval_freq=0 --save_freq=2 --log_freq=1 --num_workers=0 --trackio.enable=false"

job() {
  local name=$1 body=$2
  printf '#!/bin/bash\nset -euo pipefail\nOUT=/outputs/rocm-train/%s\nrm -rf "$OUT"\n%s\n' "$name" "$body" >"$ROOT/jobs/$name.sh"
  chmod +x "$ROOT/jobs/$name.sh"
}

seed() {  # nproc, parallelism args, extra args
  echo "accelerate launch --num_processes=$1 $TRAIN $ACT $2 $COMMON $3 --output_dir=\$OUT"
}
resume() {  # nproc, overrides
  echo "accelerate launch --num_processes=$1 $TRAIN --resume=true --config_path=\$OUT/checkpoints/000002/pretrained_model/train_config.json --steps=4 $2"
}
check4='test -d "$OUT/checkpoints/000004"; find "$OUT/checkpoints/000004" -maxdepth 2 | sort; echo TOPOLOGY_OK'

job act-reshard-resume "$(seed 2 --parallelism.dp_shard=2 '')
$(resume 4 --parallelism.dp_shard=4)
$check4"

job act-ddp2-to-1-resume "$(seed 2 '' '')
lerobot-train --resume=true --config_path=\$OUT/checkpoints/000002/pretrained_model/train_config.json --steps=4 \
  --parallelism.dp_replicate=1
$check4"

job act-fsdp-to-ddp-reject "$(seed 2 --parallelism.dp_shard=2 '')
set +e
$(resume 2 --parallelism.dp_shard=1) >\$OUT.reject.log 2>&1
ec=\$?
set -e
grep -m1 'Cannot resume' \$OUT.reject.log
test \$ec -ne 0 && grep -q 'Cannot resume: the checkpoint was written with a sharded topology' \$OUT.reject.log
echo REJECT_OK"

job act-fsdp2-dcp "$(seed 2 --parallelism.dp_shard=2 --checkpoint_format=dcp)
find \$OUT/checkpoints/000002 -maxdepth 2 | sort
test -d \$OUT/checkpoints/000002/pretrained_model/pytorch_model_fsdp_0
$(resume 2 '')
$check4"

job act-fsdp2-stdcp "$(seed 2 --parallelism.dp_shard=2 --checkpoint_format=safetensors_dcp)
test -f \$OUT/checkpoints/000002/pretrained_model/model.safetensors
test -d \$OUT/checkpoints/000002/pretrained_model/pytorch_model_fsdp_0
$(resume 2 '')
$check4"

DIT="--policy.type=multi_task_dit --policy.device=cuda --policy.push_to_hub=false \
  --policy.horizon=32 --policy.n_action_steps=24 --policy.objective=diffusion --policy.num_inference_steps=10 \
  --dataset.repo_id=lerobot/aloha_sim_transfer_cube_human --dataset.episodes='[0]'"
job dit-fsdp2-deploy "accelerate launch --num_processes=2 /outputs/rocm-train/jobs/fsdp_autowrap_train.py $DIT --parallelism.dp_shard=2 \
  --batch_size=1 --steps=2 --env_eval_freq=0 --save_freq=2 --log_freq=1 --num_workers=0 --trackio.enable=false --output_dir=\$OUT
HIP_VISIBLE_DEVICES=\${HIP_VISIBLE_DEVICES%%,*} python3 - \$OUT/checkpoints/000002/pretrained_model <<'PY'
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
PY"
echo "topology jobs written"
