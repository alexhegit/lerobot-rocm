#!/bin/bash
# MolmoAct2 LoRA: base weights load first, adapters attach afterwards.
set -u
ROOT=/data/lerobot/outputs/rocm-train
LOG="$ROOT/run-followup.log"
RESULTS="$ROOT/results.txt"
SNAP=/root/.cache/huggingface/hub/models--lerobot--MolmoAct2-SO100_101-LeRobot/snapshots/a93b5fcdce4c1e3cc7688c939fc943bf4b092a8a
exec >>"$LOG" 2>&1
echo "molmo-lora-defer begin $(date -Is)"

rm -rf "$ROOT/molmo-lora-compat"
mkdir -p "$ROOT/molmo-lora-compat"
find "$SNAP" -mindepth 1 -maxdepth 1 ! -name 'model.safetensors' -exec cp -aL {} "$ROOT/molmo-lora-compat/" \;
ln -s "$SNAP/model.safetensors" "$ROOT/molmo-lora-compat/model.safetensors"

cat >"$ROOT/jobs/molmo-lora-defer-body.sh" <<'EOF'
#!/bin/bash
set -euo pipefail
python3 - <<'PY'
import dataclasses, json
from pathlib import Path
from lerobot.policies.molmoact2.configuration_molmoact2 import MolmoAct2Config
from lerobot.policies.molmoact2.processor_molmoact2 import make_molmoact2_pre_post_processors
root = Path("/outputs/rocm-train/molmo-lora-compat")
cfg_json = json.loads((root / "config.json").read_text())
allowed = {field.name for field in dataclasses.fields(MolmoAct2Config)} | {"type"}
for key in sorted(set(cfg_json) - allowed):
    cfg_json.pop(key)
cfg_json["train_mode_vlm"] = "lora"
(root / "config.json").write_text(json.dumps(cfg_json))
cfg = MolmoAct2Config.from_pretrained(root)
cfg.device = "cpu"
_, post = make_molmoact2_pre_post_processors(cfg)
post.save_pretrained(root)
print("postprocessor", (root / "policy_postprocessor.json").is_file(), flush=True)
PY
uv pip install 'torchao>0.16'
python3 - <<'PY'
import torch, torchvision
assert torch.version.hip, "ROCm torch was replaced"
assert "+rocm" in torchvision.__version__, torchvision.__version__
print("hip", torch.version.hip, flush=True)
PY
exec "$@"
EOF
chmod +x "$ROOT/jobs/molmo-lora-defer-body.sh"

run_one() {
  local name="$1"
  local devices="$2"
  local launcher="$3"
  echo "----- $(date -Is) BEGIN ${name} devices=${devices} -----"
  echo "disk_free_gb=$(df -BG / | awk 'NR==2 {print $4}')"
  rm -rf "$ROOT/$name"
  local started ec elapsed verdict=FAIL
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
    bash /outputs/rocm-train/jobs/molmo-lora-defer-body.sh \
    ${launcher} \
    --policy.path=/outputs/rocm-train/molmo-lora-compat \
    --policy.device=cuda \
    --policy.push_to_hub=false \
    --policy.dtype=bfloat16 \
    --policy.train_mode_vlm=lora \
    --policy.gradient_checkpointing=true \
    --rename_map='{"observation.images.top":"observation.images.cam0","observation.images.wrist":"observation.images.cam1"}' \
    --dataset.repo_id=lerobot/svla_so100_pickplace \
    --dataset.episodes='[0]' \
    --batch_size=1 --steps=2 --env_eval_freq=0 --save_freq=2 --log_freq=1 \
    --num_workers=0 --trackio.enable=false \
    --output_dir=/outputs/rocm-train/${name}
  ec=$?
  elapsed=$(( $(date +%s) - started ))
  local stepf="$ROOT/$name/checkpoints/000002/training_state/training_step.json"
  if [[ "$ec" -eq 124 ]]; then
    verdict=TIMEOUT
    docker rm -f "rocm-${name}" >/dev/null 2>&1 || true
  elif [[ "$ec" -eq 0 && -f "$stepf" ]] && awk "/BEGIN ${name} /{p=1} p" "$LOG" | grep -q "trainable params"; then
    verdict=PASS
  fi
  echo "RESULT ${name} ${verdict} exit=${ec} seconds=${elapsed}"
  if [[ -f "$stepf" ]]; then echo "STEP $(tr -d '\n' <"$stepf")"; else echo "STEP missing"; fi
  echo "${name} ${verdict} exit=${ec} seconds=${elapsed}" >>"$RESULTS"
  find "$ROOT/$name" -type d -name training_state -prune -exec rm -rf {} + 2>/dev/null || true
  find "$ROOT/$name" -type f \( -name '*.safetensors' -o -name '*.distcp' \) -delete 2>/dev/null || true
  echo "----- $(date -Is) END ${name} -----"
  [[ "$verdict" == PASS ]]
}

if run_one molmo-lora-defer 0 "python3 /outputs/rocm-train/jobs/molmo_lora_defer.py"; then
  run_one molmo-lora-defer-ddp2 0,1 "accelerate launch --num_processes=2 /outputs/rocm-train/jobs/molmo_lora_defer.py" || true
fi
rm -rf "$ROOT/molmo-lora-compat"
echo "molmo-lora-defer end $(date -Is)"
