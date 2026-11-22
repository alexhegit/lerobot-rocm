#!/bin/bash
# One more Molmo LoRA attempt: strict=False plus a generated postprocessor.
set -u
ROOT=/data/lerobot/outputs/rocm-train
LOG="$ROOT/run-followup.log"
RESULTS="$ROOT/results.txt"
SNAP=/root/.cache/huggingface/hub/models--lerobot--MolmoAct2-SO100_101-LeRobot/snapshots/a93b5fcdce4c1e3cc7688c939fc943bf4b092a8a
exec >>"$LOG" 2>&1
echo "molmo-lora-retry2 begin $(date -Is)"

rm -rf "$ROOT/molmo-lora-compat" "$ROOT/molmo-lora-nonstrict"
mkdir -p "$ROOT/molmo-lora-compat"
find "$SNAP" -mindepth 1 -maxdepth 1 ! -name 'model.safetensors' -exec cp -aL {} "$ROOT/molmo-lora-compat/" \;
ln -s "$SNAP/model.safetensors" "$ROOT/molmo-lora-compat/model.safetensors"

cat >"$ROOT/jobs/molmo-lora-retry2.sh" <<'EOF'
#!/bin/bash
set -euo pipefail
python3 - <<'PY'
import dataclasses, json
from pathlib import Path
from lerobot.policies.molmoact2.configuration_molmoact2 import MolmoAct2Config
from lerobot.policies.molmoact2.processor_molmoact2 import make_molmoact2_pre_post_processors
p = Path("/outputs/rocm-train/molmo-lora-compat/config.json")
cfg_json = json.loads(p.read_text())
allowed = {f.name for f in dataclasses.fields(MolmoAct2Config)} | {"type"}
for key in sorted(set(cfg_json) - allowed):
    cfg_json.pop(key)
cfg_json["train_mode_vlm"] = "lora"
p.write_text(json.dumps(cfg_json))
cfg = MolmoAct2Config.from_pretrained("/outputs/rocm-train/molmo-lora-compat")
_, post = make_molmoact2_pre_post_processors(cfg)
post.save_pretrained("/outputs/rocm-train/molmo-lora-compat")
print("postprocessor written", (p.parent / "policy_postprocessor.json").is_file())
PY
uv pip install 'torchao>0.16'
python3 - <<'PY'
import torch, torchvision
assert torch.version.hip and "+rocm" in torchvision.__version__
print("hip", torch.version.hip)
PY
python3 /outputs/rocm-train/jobs/molmo_lora_nonstrict.py \
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
  --output_dir=/outputs/rocm-train/molmo-lora-nonstrict
EOF
chmod +x "$ROOT/jobs/molmo-lora-retry2.sh"

echo "----- $(date -Is) BEGIN molmo-lora-retry2 devices=0 -----"
started=$(date +%s)
timeout --signal=TERM 5400 docker run --rm --name rocm-molmo-lora-retry2 \
  --ipc=host --shm-size 16g \
  --device=/dev/kfd --device=/dev/dri \
  --group-add 44 --group-add 992 \
  -e HIP_VISIBLE_DEVICES=0 \
  -e MIOPEN_FIND_MODE=2 \
  -v /data/lerobot/outputs:/outputs \
  -v /root/.cache/huggingface:/root/.cache/huggingface \
  lerobot-rocm bash /outputs/rocm-train/jobs/molmo-lora-retry2.sh
ec=$?
elapsed=$(( $(date +%s) - started ))
host_out="$ROOT/molmo-lora-nonstrict"
stepf="$host_out/checkpoints/000002/training_state/training_step.json"
verdict=FAIL
if [[ "$ec" -eq 124 ]]; then
  verdict=TIMEOUT
  docker rm -f rocm-molmo-lora-retry2 >/dev/null 2>&1 || true
elif [[ "$ec" -eq 0 && -f "$stepf" ]] && awk '/BEGIN molmo-lora-retry2 /{p=1} p' "$LOG" | grep -q "trainable params"; then
  verdict=PASS
elif [[ "$ec" -eq 0 && -f "$stepf" ]]; then
  verdict=FAIL
  echo "marker missing: trainable params"
fi
echo "RESULT molmo-lora-retry2 ${verdict} exit=${ec} seconds=${elapsed}"
if [[ -f "$stepf" ]]; then echo "STEP $(tr -d '\n' <"$stepf")"; fi
echo "molmo-lora-retry2 ${verdict} exit=${ec} seconds=${elapsed}" >>"$RESULTS"
find "$host_out" -type d -name training_state -prune -exec rm -rf {} + 2>/dev/null || true
find "$host_out" -type f \( -name '*.safetensors' -o -name '*.distcp' \) -delete 2>/dev/null || true
rm -rf "$ROOT/molmo-lora-compat"
echo "----- $(date -Is) END molmo-lora-retry2 -----"
echo "molmo-lora-retry2 end $(date -Is)"
