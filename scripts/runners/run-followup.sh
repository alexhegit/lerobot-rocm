#!/bin/bash
# Follow-up: MolmoAct2 LoRA, flux3, g05 CoT. Hub push is skipped (no repo id).
set -u
ROOT=/data/lerobot/outputs/rocm-train
LOG="$ROOT/run-followup.log"
RESULTS="$ROOT/results.txt"
MOLMO_SNAP=/root/.cache/huggingface/hub/models--lerobot--MolmoAct2-SO100_101-LeRobot/snapshots/a93b5fcdce4c1e3cc7688c939fc943bf4b092a8a
SO100_SNAP=/root/.cache/huggingface/lerobot/hub/datasets--lerobot--svla_so100_pickplace/snapshots/728583b5eaf9e739a7f119e2def466fa1d552402
mkdir -p "$ROOT/jobs"
exec >>"$LOG" 2>&1
echo "followup begin $(date -Is)"

free_weights() {
  local host_out="$1"
  find "$host_out" -type d -name training_state -prune -exec rm -rf {} + 2>/dev/null || true
  find "$host_out" -type f \( -name '*.safetensors' -o -name '*.distcp' \) -delete 2>/dev/null || true
}

run_job() {
  local name="$1"
  local devices="$2"
  local script="$3"
  local host_out="$ROOT/$4"
  local step_dir="$5"
  local marker="${6:-}"
  echo "----- $(date -Is) BEGIN ${name} devices=${devices} -----"
  echo "disk_free_gb=$(df -BG / | awk 'NR==2 {print $4}')"
  rm -rf "$host_out"
  local started ec elapsed verdict has_ckpt=0
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
  if find "$host_out/checkpoints/${step_dir}" -name '*.distcp' 2>/dev/null | grep -q .; then has_ckpt=1; fi
  if [[ "$ec" -eq 124 ]]; then
    verdict="TIMEOUT"
    docker rm -f "rocm-${name}" >/dev/null 2>&1 || true
  elif [[ "$ec" -eq 0 && "$has_ckpt" -eq 1 ]]; then
    verdict="PASS"
  elif [[ "$ec" -eq 3 ]]; then
    verdict="BLOCKED"
  else
    verdict="FAIL"
  fi
  if [[ -n "$marker" && "$verdict" == "PASS" ]]; then
    if ! awk "/BEGIN ${name} /{p=1} p" "$LOG" | grep -q "$marker"; then
      verdict="FAIL"
      echo "marker missing: ${marker}"
    fi
  fi
  echo "RESULT ${name} ${verdict} exit=${ec} seconds=${elapsed}"
  if [[ -f "$stepf" ]]; then echo "STEP $(tr -d '\n' <"$stepf")"; else echo "STEP missing"; fi
  echo "${name} ${verdict} exit=${ec} seconds=${elapsed}" >>"$RESULTS"
  echo "----- $(date -Is) END ${name} -----"
  [[ "$verdict" == "PASS" ]]
}

# --- Molmo LoRA local config: symlink the 21GB weight, copy the small files ---
rm -rf "$ROOT/molmo-lora-compat"
mkdir -p "$ROOT/molmo-lora-compat"
find "$MOLMO_SNAP" -mindepth 1 -maxdepth 1 ! -name 'model.safetensors' -exec cp -aL {} "$ROOT/molmo-lora-compat/" \;
ln -s "$MOLMO_SNAP/model.safetensors" "$ROOT/molmo-lora-compat/model.safetensors"
python3 - <<'PY'
import json
from pathlib import Path
p = Path("/data/lerobot/outputs/rocm-train/molmo-lora-compat/config.json")
cfg = json.loads(p.read_text())
for key in ("enable_lora_vlm", "enable_lora_action_expert", "train_action_expert_only", "model_dtype"):
    cfg.pop(key, None)
cfg["train_mode_vlm"] = "lora"
p.write_text(json.dumps(cfg))
w = Path("/data/lerobot/outputs/rocm-train/molmo-lora-compat/model.safetensors")
print("molmo link", w.stat().st_size, "mode", cfg["train_mode_vlm"])
PY

MOLMO_PREFIX='python3 - <<'"'"'PY'"'"'
import dataclasses, json
from pathlib import Path
from lerobot.policies.molmoact2.configuration_molmoact2 import MolmoAct2Config
p = Path("/outputs/rocm-train/molmo-lora-compat/config.json")
cfg = json.loads(p.read_text())
allowed = {f.name for f in dataclasses.fields(MolmoAct2Config)} | {"type"}
extra = sorted(set(cfg) - allowed)
for key in extra:
    cfg.pop(key)
cfg["train_mode_vlm"] = "lora"
p.write_text(json.dumps(cfg))
print("molmo stripped", extra)
PY
uv pip install "torchao>0.16"
python3 - <<'"'"'PY'"'"'
import torch, torchvision, torchao
assert torch.version.hip, "ROCm torch was replaced"
assert "+rocm" in torchvision.__version__, torchvision.__version__
print("torchao", torchao.__version__, "hip", torch.version.hip)
PY'

cat >"$ROOT/jobs/molmo-lora-1.sh" <<EOF
#!/bin/bash
set -euo pipefail
${MOLMO_PREFIX}
lerobot-train \\
  --policy.path=/outputs/rocm-train/molmo-lora-compat \\
  --policy.device=cuda \\
  --policy.push_to_hub=false \\
  --policy.dtype=bfloat16 \\
  --policy.train_mode_vlm=lora \\
  --policy.gradient_checkpointing=true \\
  --rename_map='{"observation.images.top":"observation.images.cam0","observation.images.wrist":"observation.images.cam1"}' \\
  --dataset.repo_id=lerobot/svla_so100_pickplace \\
  --dataset.episodes='[0]' \\
  --batch_size=1 \\
  --steps=2 \\
  --env_eval_freq=0 \\
  --save_freq=2 \\
  --log_freq=1 \\
  --num_workers=0 \\
  --trackio.enable=false \\
  --output_dir=/outputs/rocm-train/molmo-lora-1
EOF

cat >"$ROOT/jobs/molmo-lora-ddp2.sh" <<EOF
#!/bin/bash
set -euo pipefail
${MOLMO_PREFIX}
accelerate launch --num_processes=2 "\$(which lerobot-train)" \\
  --policy.path=/outputs/rocm-train/molmo-lora-compat \\
  --policy.device=cuda \\
  --policy.push_to_hub=false \\
  --policy.dtype=bfloat16 \\
  --policy.train_mode_vlm=lora \\
  --policy.gradient_checkpointing=true \\
  --rename_map='{"observation.images.top":"observation.images.cam0","observation.images.wrist":"observation.images.cam1"}' \\
  --dataset.repo_id=lerobot/svla_so100_pickplace \\
  --dataset.episodes='[0]' \\
  --batch_size=1 \\
  --steps=2 \\
  --env_eval_freq=0 \\
  --save_freq=2 \\
  --log_freq=1 \\
  --num_workers=0 \\
  --trackio.enable=false \\
  --output_dir=/outputs/rocm-train/molmo-lora-ddp2
EOF

# --- flux3: import or install flash, then one single-GPU LoRA step. attn_mode=torch if flash stays missing ---
cat >"$ROOT/jobs/flux3-1.sh" <<'EOF'
#!/bin/bash
set -euo pipefail
python3 - <<'PY'
import torch, torchvision
assert torch.version.hip
assert "+rocm" in torchvision.__version__
try:
    from flash_attn_interface import flash_attn_func
    print("IMPORT_OK", flash_attn_func)
except Exception as exc:
    print("IMPORT_FAIL", type(exc).__name__, exc)
PY
if ! grep -q IMPORT_OK <(python3 -c 'from flash_attn_interface import flash_attn_func; print("IMPORT_OK")' 2>/dev/null || true); then
  set +e
  uv pip install 'flash-attn' 
  install_ec=$?
  set -e
  python3 - <<'PY'
import torch, torchvision
assert torch.version.hip, "ROCm torch was replaced"
assert "+rocm" in torchvision.__version__, torchvision.__version__
try:
    from flash_attn_interface import flash_attn_func
    print("IMPORT_OK", torch.__version__)
except Exception as exc:
    print("IMPORT_STILL_FAIL", type(exc).__name__, exc)
    raise SystemExit(3)
PY
fi
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
  --output_dir=/outputs/rocm-train/flux3-1
EOF

# --- g05 CoT: local dataset copy with a constant atomic_task column ---
cat >"$ROOT/jobs/g05-cot-prep.py" <<'PY'
"""Copy the SO100 snapshot and add a constant atomic_task column."""
import json
import shutil
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

src = Path("/root/.cache/huggingface/lerobot/hub/datasets--lerobot--svla_so100_pickplace/snapshots/728583b5eaf9e739a7f119e2def466fa1d552402")
dst = Path("/outputs/rocm-train/g05-cot-data")
if dst.exists():
    shutil.rmtree(dst)
for path in src.rglob("*"):
    rel = path.relative_to(src)
    target = dst / rel
    if path.is_symlink() or path.is_file():
        target.parent.mkdir(parents=True, exist_ok=True)
        if path.name == "file-000.parquet" and "data" in rel.parts:
            table = pq.read_table(path.resolve())
            column = pa.array(["pick up the cube"] * table.num_rows)
            if "atomic_task" in table.column_names:
                table = table.drop(["atomic_task"])
            table = table.append_column("atomic_task", column)
            pq.write_table(table, target)
            print("rows", table.num_rows)
        else:
            target.symlink_to(path.resolve())
    elif path.is_dir():
        target.mkdir(parents=True, exist_ok=True)
info_path = dst / "meta" / "info.json"
info = json.loads(info_path.read_text())
info["features"]["atomic_task"] = {"dtype": "string", "shape": [1], "names": None}
info_path.write_text(json.dumps(info))
print("g05 data ready", dst)
PY

cat >"$ROOT/jobs/g05-cot-1.sh" <<'EOF'
#!/bin/bash
set -euo pipefail
python3 /outputs/rocm-train/jobs/g05-cot-prep.py
lerobot-train \
  --policy.path=lerobot/g05_so101 \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --policy.dtype=bfloat16 \
  --policy.predict_cot=true \
  --policy.use_language_recipe=true \
  --policy.camera_order='["observation.images.top", "observation.images.wrist_left", "observation.images.wrist"]' \
  --policy.n_obs_steps=1 \
  --dataset.repo_id=lerobot/svla_so100_pickplace \
  --dataset.root=/outputs/rocm-train/g05-cot-data \
  --dataset.episodes='[0]' \
  --batch_size=1 \
  --steps=2 \
  --env_eval_freq=0 \
  --save_freq=2 \
  --log_freq=1 \
  --num_workers=0 \
  --trackio.enable=false \
  --output_dir=/outputs/rocm-train/g05-cot-1
EOF
chmod +x "$ROOT/jobs/"*.sh

echo "hub-push SKIP no private repo id"
echo "hub-push SKIP no-repo" >>"$RESULTS"

if run_job molmo-lora-1 0 molmo-lora-1.sh molmo-lora-1 000002 "trainable params"; then
  run_job molmo-lora-ddp2 0,1 molmo-lora-ddp2.sh molmo-lora-ddp2 000002 "trainable params" || true
fi
free_weights "$ROOT/molmo-lora-1"
free_weights "$ROOT/molmo-lora-ddp2"
rm -rf "$ROOT/molmo-lora-compat"

run_job flux3-1 0 flux3-1.sh flux3-1 000002 || true
free_weights "$ROOT/flux3-1"

run_job g05-cot-1 0 g05-cot-1.sh g05-cot-1 000002 || true
free_weights "$ROOT/g05-cot-1"
rm -rf "$ROOT/g05-cot-data"

echo "followup end $(date -Is)"
