#!/bin/bash
# flux3 with torch attention, then Molmo LoRA loaded with strict=False.
set -u
ROOT=/data/lerobot/outputs/rocm-train
LOG="$ROOT/run-followup.log"
RESULTS="$ROOT/results.txt"
MOLMO_SNAP=/root/.cache/huggingface/hub/models--lerobot--MolmoAct2-SO100_101-LeRobot/snapshots/a93b5fcdce4c1e3cc7688c939fc943bf4b092a8a
exec >>"$LOG" 2>&1
echo "followup-retry begin $(date -Is)"

free_weights() {
  find "$1" -type d -name training_state -prune -exec rm -rf {} + 2>/dev/null || true
  find "$1" -type f \( -name '*.safetensors' -o -name '*.distcp' \) -delete 2>/dev/null || true
}

run_job() {
  local name="$1" devices="$2" script="$3" out_name="$4" step_dir="$5" marker="${6:-}"
  local host_out="$ROOT/$out_name"
  echo "----- $(date -Is) BEGIN ${name} devices=${devices} -----"
  echo "disk_free_gb=$(df -BG / | awk 'NR==2 {print $4}')"
  rm -rf "$host_out"
  local started ec elapsed verdict=FAIL has_ckpt=0
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
    bash "/outputs/rocm-train/jobs/${script}"
  ec=$?
  elapsed=$(( $(date +%s) - started ))
  local ckpt="$host_out/checkpoints/${step_dir}/pretrained_model/model.safetensors"
  local stepf="$host_out/checkpoints/${step_dir}/training_state/training_step.json"
  if [[ -f "$ckpt" || -f "$stepf" ]]; then has_ckpt=1; fi
  if [[ "$ec" -eq 124 ]]; then
    verdict=TIMEOUT
    docker rm -f "rocm-${name}" >/dev/null 2>&1 || true
  elif [[ "$ec" -eq 0 && "$has_ckpt" -eq 1 ]]; then
    verdict=PASS
  fi
  if [[ -n "$marker" && "$verdict" == PASS ]]; then
    if ! awk "/BEGIN ${name} /{p=1} p" "$LOG" | grep -q "$marker"; then
      verdict=FAIL
      echo "marker missing: ${marker}"
    fi
  fi
  echo "RESULT ${name} ${verdict} exit=${ec} seconds=${elapsed}"
  if [[ -f "$stepf" ]]; then echo "STEP $(tr -d '\n' <"$stepf")"; else echo "STEP missing"; fi
  echo "${name} ${verdict} exit=${ec} seconds=${elapsed}" >>"$RESULTS"
  echo "----- $(date -Is) END ${name} -----"
}

cat >"$ROOT/jobs/flux3-torch.sh" <<'EOF'
#!/bin/bash
set -euo pipefail
python3 - <<'PY'
try:
    from flash_attn_interface import flash_attn_func
    print("IMPORT_OK", flash_attn_func)
except Exception as exc:
    print("IMPORT_FAIL", type(exc).__name__, exc)
print("attn_mode torch")
PY
lerobot-train \
  --policy.path=black-forest-labs/flux-3-action-so101 \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --policy.attn_mode=torch \
  --policy.gradient_checkpointing=true \
  --peft.method_type=LORA \
  --peft.r=8 \
  --peft.lora_alpha=8 \
  --accelerator.mixed_precision=bf16 \
  --rename_map='{"observation.images.top":"observation.images.scene"}' \
  --dataset.repo_id=lerobot/svla_so100_pickplace \
  --dataset.episodes='[0]' \
  --batch_size=1 \
  --steps=2 \
  --env_eval_freq=0 \
  --save_freq=2 \
  --log_freq=1 \
  --num_workers=0 \
  --trackio.enable=false \
  --output_dir=/outputs/rocm-train/flux3-torch
EOF

cat >"$ROOT/jobs/molmo_lora_nonstrict.py" <<'PY'
"""Load MolmoAct2 with strict=False so fresh LoRA keys are not required."""
import sys

from lerobot.policies.molmoact2.modeling_molmoact2 import MolmoAct2Policy

_orig = MolmoAct2Policy.from_pretrained.__func__


def _loose(cls, pretrained_name_or_path, *args, strict=True, **kwargs):
    print("molmo strict_override False", flush=True)
    return _orig(cls, pretrained_name_or_path, strict=False, **kwargs)


MolmoAct2Policy.from_pretrained = classmethod(_loose)

from lerobot.scripts.lerobot_train import main

if __name__ == "__main__":
    raise SystemExit(main())
PY

cat >"$ROOT/jobs/molmo-lora-nonstrict.sh" <<'EOF'
#!/bin/bash
set -euo pipefail
python3 - <<'PY'
import dataclasses, json
from pathlib import Path
from lerobot.policies.molmoact2.configuration_molmoact2 import MolmoAct2Config
p = Path("/outputs/rocm-train/molmo-lora-compat/config.json")
cfg = json.loads(p.read_text())
allowed = {f.name for f in dataclasses.fields(MolmoAct2Config)} | {"type"}
for key in sorted(set(cfg) - allowed):
    cfg.pop(key)
cfg["train_mode_vlm"] = "lora"
p.write_text(json.dumps(cfg))
print("molmo stripped for nonstrict")
PY
uv pip install 'torchao>0.16'
python3 - <<'PY'
import torch, torchvision, torchao
assert torch.version.hip
assert "+rocm" in torchvision.__version__
print("torchao", torchao.__version__, "hip", torch.version.hip)
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
  --batch_size=1 \
  --steps=2 \
  --env_eval_freq=0 \
  --save_freq=2 \
  --log_freq=1 \
  --num_workers=0 \
  --trackio.enable=false \
  --output_dir=/outputs/rocm-train/molmo-lora-nonstrict
EOF
chmod +x "$ROOT/jobs/flux3-torch.sh" "$ROOT/jobs/molmo-lora-nonstrict.sh"

run_job flux3-torch 0 flux3-torch.sh flux3-torch 000002
free_weights "$ROOT/flux3-torch"

rm -rf "$ROOT/molmo-lora-compat"
mkdir -p "$ROOT/molmo-lora-compat"
find "$MOLMO_SNAP" -mindepth 1 -maxdepth 1 ! -name 'model.safetensors' -exec cp -aL {} "$ROOT/molmo-lora-compat/" \;
ln -s "$MOLMO_SNAP/model.safetensors" "$ROOT/molmo-lora-compat/model.safetensors"

run_job molmo-lora-nonstrict 0 molmo-lora-nonstrict.sh molmo-lora-nonstrict 000002 "trainable params"
free_weights "$ROOT/molmo-lora-nonstrict"
rm -rf "$ROOT/molmo-lora-compat"
echo "followup-retry end $(date -Is)"
