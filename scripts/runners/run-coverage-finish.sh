#!/bin/bash
# Finish the remaining coverage items. Passed-model Hub caches may be deleted.
set -u
ROOT=/data/lerobot/outputs/rocm-train
LOG="$ROOT/run-coverage.log"
RESULTS="$ROOT/results.txt"
mkdir -p "$ROOT/jobs"
exec >>"$LOG" 2>&1
echo "coverage-finish begin $(date -Is)"

python3 - <<'PY'
import os, shutil
root = "/root/.cache/huggingface/hub"
drop = {
    "models--lerobot--Cosmos3-Nano",
    "models--lerobot--Cosmos3-Edge",
    "models--lerobot--lingbot_vla_v2_base",
    "models--lerobot--MolmoAct2-SO100_101-LeRobot",
    "models--allenai--MolmoAct2-SO100_101",
    "models--black-forest-labs--flux-3-action-so101",
    "models--black-forest-labs--flux-3-action-base",
    "models--black-forest-labs--flux-3-action-droid",
    "models--lerobot--pi0_base",
    "models--lerobot--pi05_base",
    "models--lerobot--pi0fast-base",
    "models--lerobot--g05_so101",
    "models--lerobot--fastwam_base",
    "models--Wan-AI--Wan2.2-TI2V-5B-Diffusers",
    "models--robbyant--lingbot-va-posttrain-libero-long",
    "models--lerobot--wall-oss-0.5",
    "models--x-square-robot--wall-oss-flow",
    "models--lerobot--dm05_base",
    "models--lerobot--lingbot_va_libero_long",
    "models--lerobot--VLA-JEPA-Pretrain",
    "models--jialei02--lawam-pretrain-lerobot",
    "models--lerobot--xvla-base",
    "models--HuggingFaceTB--SmolVLM2-500M-Video-Instruct",
    "models--lerobot--smolvla_base",
    "models--OpenGVLab--InternVL3-1B-hf",
}
keep_blobs, drop_blobs = set(), set()
for repo in os.listdir(root):
    if not repo.startswith("models--"):
        continue
    blobs = set()
    base = os.path.join(root, repo)
    for dirpath, _, files in os.walk(base):
        for name in files:
            path = os.path.join(dirpath, name)
            if os.path.islink(path):
                blobs.add(os.path.realpath(path))
    if repo in drop:
        drop_blobs |= blobs
        shutil.rmtree(base, ignore_errors=True)
        print("dropped", repo)
    else:
        keep_blobs |= blobs
freed = 0
for blob in drop_blobs - keep_blobs:
    if os.path.isfile(blob):
        freed += os.path.getsize(blob)
        os.remove(blob)
print(f"freed_gb={freed/1e9:.1f}")
PY
df -h / | tail -1

# Three camera names, one shared video, 14-dim aloha actions.
python3 - <<'PY'
import json, os, shutil
src = "/root/.cache/huggingface/lerobot/hub/datasets--lerobot--aloha_sim_transfer_cube_human/snapshots/6a43d500f101255823a9d2b9dc244eeb01a2cd31"
dst = "/data/lerobot/outputs/rocm-train/fineart-aloha3"
if os.path.isdir(dst):
    shutil.rmtree(dst)
os.makedirs(dst)
for part in ("meta", "data"):
    shutil.copytree(src + "/" + part, dst + "/" + part, symlinks=False, copy_function=shutil.copy2)
# copytree follows? copy2 on symlink copies target content if we don't use symlinks=True.
# shutil.copytree default follow_symlinks True for copy2 in 3.12? copy2 copies file content through symlink. Good.
cams = [
    "observation.images.cam_high",
    "observation.images.cam_left_wrist",
    "observation.images.cam_right_wrist",
]
src_vid = os.path.realpath(src + "/videos/observation.images.top")
for cam in cams:
    target = os.path.join(dst, "videos", cam)
    shutil.copytree(src_vid, target, copy_function=lambda s, d: os.symlink(os.path.realpath(s), d))
info_path = dst + "/meta/info.json"
info = json.load(open(info_path))
feat = info["features"].pop("observation.images.top")
for cam in cams:
    info["features"][cam] = feat
json.dump(info, open(info_path, "w"), indent=2)
stats_path = dst + "/meta/stats.json"
stats = json.load(open(stats_path))
if "observation.images.top" in stats:
    image_stats = stats.pop("observation.images.top")
    for cam in cams:
        stats[cam] = image_stats
    json.dump(stats, open(stats_path, "w"))
print("fineart dataset ready", dst)
PY

free_weights() {
  local host_out="$1"
  find "$host_out" -type d -name training_state -prune -exec rm -rf {} + 2>/dev/null || true
  find "$host_out" -type f \( -name '*.safetensors' -o -name '*.distcp' -o -name '*.bin' -o -name '*.pt' \) -delete 2>/dev/null || true
}

run_docker() {
  local name="$1"
  local script="$2"
  local host_out="$ROOT/$name"
  echo "----- $(date -Is) BEGIN ${name} devices=0 -----"
  echo "disk_free_gb=$(df -BG / | awk 'NR==2 {print $4}')"
  rm -rf "$host_out"
  local started ec elapsed verdict stepf
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
  stepf=$(find "$host_out" -name training_step.json 2>/dev/null | head -1 || true)
  if [[ "$ec" -eq 124 ]]; then
    verdict="TIMEOUT"
    docker rm -f "rocm-${name}" >/dev/null 2>&1 || true
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

cat >"$ROOT/jobs/cover-fineart3.sh" <<'EOF'
#!/bin/bash
set -euo pipefail
lerobot-train \
  --policy.path=lerobot/fineart_vla_base \
  --policy.dtype=bfloat16 \
  --policy.gradient_checkpointing=true \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --dataset.repo_id=local/fineart_aloha3 \
  --dataset.root=/outputs/rocm-train/fineart-aloha3 \
  --dataset.episodes='[0]' \
  --batch_size=1 --steps=2 --env_eval_freq=0 --save_freq=2 --log_freq=1 \
  --num_workers=0 --trackio.enable=false \
  --output_dir=/outputs/rocm-train/cover-fineart3
EOF

cat >"$ROOT/jobs/cover-groot.sh" <<'EOF'
#!/bin/bash
set -euo pipefail
lerobot-train \
  --policy.type=groot \
  --policy.base_model_path=nvidia/GR00T-N1.7-3B \
  --policy.embodiment_tag=new_embodiment \
  --policy.use_bf16=true \
  --policy.use_relative_actions=false \
  --policy.chunk_size=16 \
  --policy.n_action_steps=16 \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --dataset.repo_id=lerobot/svla_so100_pickplace \
  --dataset.episodes='[0]' \
  --batch_size=1 --steps=2 --env_eval_freq=0 --save_freq=2 --log_freq=1 \
  --num_workers=0 --trackio.enable=false \
  --output_dir=/outputs/rocm-train/cover-groot
EOF

cat >"$ROOT/jobs/cover-hilserl.sh" <<'EOF'
#!/bin/bash
set -euo pipefail
python3 - <<'PY'
import torch
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.gaussian_actor.configuration_gaussian_actor import GaussianActorConfig
from lerobot.policies.gaussian_actor.modeling_gaussian_actor import GaussianActorPolicy
from lerobot.rl.algorithms.sac import SACAlgorithm, SACAlgorithmConfig
from lerobot.utils.constants import ACTION, OBS_STATE

state_dim, action_dim = 10, 6
cfg = GaussianActorConfig(
    device="cuda",
    input_features={OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(state_dim,))},
    output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(action_dim,))},
    dataset_stats={
        OBS_STATE: {"min": [0.0] * state_dim, "max": [1.0] * state_dim},
        ACTION: {"min": [0.0] * action_dim, "max": [1.0] * action_dim},
    },
)
cfg.validate_features()
policy = GaussianActorPolicy(cfg)
algo_cfg = SACAlgorithmConfig.from_policy_config(cfg)
algo_cfg.utd_ratio = 1
algo_cfg.policy_update_freq = 1
algorithm = SACAlgorithm(policy=policy, config=algo_cfg)
algorithm.make_optimizers_and_scheduler()
assert torch.cuda.is_available() and torch.version.hip

def batch():
    b, s, a = 4, state_dim, action_dim
    obs = {OBS_STATE: torch.randn(b, s, device="cuda")}
    nxt = {OBS_STATE: torch.randn(b, s, device="cuda")}
    return {
        ACTION: torch.randn(b, a, device="cuda"),
        "reward": torch.randn(b, device="cuda"),
        "state": obs,
        "next_state": nxt,
        "done": torch.zeros(b, device="cuda"),
        "complementary_info": {},
    }

losses = []
for step in range(2):
    stats = algorithm.update(iter([batch(), batch()]))
    losses.append(stats.losses)
    print(f"step {step+1} {stats.losses}")
print("HIL_PASS", losses)
PY
EOF

cat >"$ROOT/jobs/cover-robometer-infer.sh" <<'EOF'
#!/bin/bash
set -euo pipefail
python3 - <<'PY'
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.rewards import make_reward_model, make_reward_model_config, make_reward_pre_post_processors

ds = LeRobotDataset("lerobot/svla_so100_pickplace", episodes=[0])
item = ds[0]
cfg = make_reward_model_config(
    "robometer",
    pretrained_path="lerobot/Robometer-4B",
    device="cuda",
    image_key="observation.images.top",
)
assert torch.version.hip
model = make_reward_model(cfg)
print("trainable", model.is_trainable)
pre, _post = make_reward_pre_post_processors(cfg)
batch = pre(item)
reward = model.compute_reward(batch)
print("ROBOMETER_REWARD", reward.detach().float().cpu().tolist())
PY
EOF

cat >"$ROOT/jobs/cover-topreward-infer.sh" <<'EOF'
#!/bin/bash
set -euo pipefail
python3 - <<'PY'
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.rewards import make_reward_model, make_reward_model_config, make_reward_pre_post_processors

ds = LeRobotDataset("lerobot/svla_so100_pickplace", episodes=[0])
item = ds[0]
cfg = make_reward_model_config(
    "topreward",
    device="cuda",
    image_key="observation.images.top",
    vlm_name="Qwen/Qwen3-VL-8B-Instruct",
)
assert torch.version.hip
model = make_reward_model(cfg)
print("trainable", model.is_trainable)
pre, _post = make_reward_pre_post_processors(cfg)
batch = pre(item)
reward = model.compute_reward(batch)
print("TOPREWARD_REWARD", reward.detach().float().cpu().tolist())
PY
EOF
chmod +x "$ROOT/jobs"/cover-fineart3.sh "$ROOT/jobs"/cover-groot.sh "$ROOT/jobs"/cover-hilserl.sh \
  "$ROOT/jobs"/cover-robometer-infer.sh "$ROOT/jobs"/cover-topreward-infer.sh

for name in cover-fineart3 cover-groot cover-hilserl cover-robometer-infer cover-topreward-infer; do
  run_docker "$name" "${name}.sh" || true
done
echo "coverage-finish end $(date -Is)"
df -h / | tail -1
