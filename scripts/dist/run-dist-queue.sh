#!/bin/bash
# Run a queue of multi-GPU jobs on the ROCm node.
# Usage: run-dist-queue.sh <queue-file> <log-name>
# Queue line: name|base|gpus|nproc|extra|pre
#   base  = jobs/<base>.sh, a passing single-GPU script with one `lerobot-train` call,
#           or CUSTOM to run jobs/<name>.sh unchanged.
#   extra = arguments appended after `lerobot-train`.
#   pre   = shell lines (separated by ';;') run before training.
# A PASS deletes the job's checkpoints. Jobs are skipped below MIN_FREE_GB of free disk.
set -u
OUT=${OUT:-/data/lerobot/outputs}
ROOT=$OUT/rocm-train
QUEUE="$1"
LOG="$ROOT/$2.log"
MIN_FREE_GB=${MIN_FREE_GB:-60}

while IFS='|' read -r name base gpus nproc extra pre; do
  [[ -z "$name" || "$name" == \#* ]] && continue
  job="$ROOT/jobs/${name}.sh"
  if [[ "$base" != CUSTOM ]]; then
    python3 - "$ROOT/jobs/${base}.sh" "$job" "$name" "$nproc" "$extra" "$pre" <<'PY'
import re, sys
src, dst, name, nproc, extra, pre = sys.argv[1:]
# @AUTOWRAP: launch through fsdp_autowrap_train.py, which picks FSDP wrap classes from the model.
entry = '"$(which lerobot-train)"'
if "@AUTOWRAP" in extra:
    extra = extra.replace("@AUTOWRAP", "").strip()
    entry = "/outputs/rocm-train/jobs/fsdp_autowrap_train.py"
lines = open(src).read().splitlines()
out, done, outdir = [], False, 0
for line in lines:
    if not done and line.startswith("lerobot-train"):
        line = f"accelerate launch --num_processes={nproc} {entry}" + line[len("lerobot-train"):]
        done = True
    # Extra args go after --output_dir, the last argument of every base script, so they win over
    # the base script's own values.
    line, n = re.subn(
        r"--output_dir=/outputs/rocm-train/\S+",
        lambda m: f"--output_dir=/outputs/rocm-train/{name} {extra}".rstrip(),
        line,
    )
    outdir += n
    out.append(line)
if not done or outdir != 1:
    sys.exit(f"{src}: expected one lerobot-train line and one --output_dir, got {done=} {outdir=}")
idx = next((i + 1 for i, l in enumerate(out) if l.startswith("set -euo pipefail")), 1)
pre_lines = [p.strip() for p in pre.split(";;") if p.strip()]
out[idx:idx] = pre_lines
open(dst, "w").write("\n".join(out) + "\n")
PY
    [[ $? -ne 0 ]] && { echo "GENFAIL $name" | tee -a "$LOG"; echo "$name GENFAIL" >>"$ROOT/results.txt"; continue; }
  fi
  chmod +x "$job"
  if [[ -n "${DRY_RUN:-}" ]]; then
    echo "== $name ($gpus, $nproc procs)"; grep -E "accelerate launch|output_dir|^[a-z]" "$job" | grep -v "^set " | head -n 6
    continue
  fi

  free_gb=$(df -BG --output=avail / | tail -1 | tr -dc '0-9')
  if (( free_gb < MIN_FREE_GB )); then
    echo "SKIP ${name} free=${free_gb}G" | tee -a "$LOG"
    echo "${name} SKIP free=${free_gb}G" >>"$ROOT/results.txt"
    continue
  fi
  mounts=()
  [[ "$name" == flux3* ]] && mounts=(-v "$ROOT/pr5213:/pr")
  echo "----- $(date -Is) BEGIN ${name} devices=${gpus} free=${free_gb}G -----" >>"$LOG"
  s=$(date +%s)
  timeout --signal=TERM 5400 docker run --rm --name "rocm-${name}" --ipc=host --shm-size 32g \
    --device=/dev/kfd --device=/dev/dri --group-add 44 --group-add 992 \
    -e HIP_VISIBLE_DEVICES="$gpus" -e MIOPEN_FIND_MODE=2 \
    -v "$OUT":/outputs -v /root/.cache/huggingface:/root/.cache/huggingface \
    "${mounts[@]}" lerobot-rocm bash "/outputs/rocm-train/jobs/${name}.sh" </dev/null >>"$LOG" 2>&1
  ec=$?
  e=$(( $(date +%s) - s ))
  v=FAIL; (( ec == 0 )) && v=PASS; (( ec == 124 )) && v=TIMEOUT
  seg=$(awk "/BEGIN ${name} /,0" "$LOG")
  rep=$(grep -ao "'dp_replicate': [0-9]*" <<<"$seg" | tail -1 | tr -dc '0-9')
  shard=$(grep -ao "'dp_shard': [0-9-]*" <<<"$seg" | tail -1 | tr -dc '0-9-')
  steps=$(grep -aoE "step:[0-9]+ " <<<"$seg" | tr -dc '0-9 ' | xargs)
  echo "RESULT ${name} ${v} exit=${ec} seconds=${e} dp_replicate=${rep} dp_shard=${shard} steps=[${steps}]" >>"$LOG"
  echo "${name} ${v} exit=${ec} seconds=${e} dp_replicate=${rep} dp_shard=${shard}" >>"$ROOT/results.txt"
  [[ "$v" == PASS ]] && rm -rf "$ROOT/$name/checkpoints"
done <"$QUEUE"
echo "QUEUE END $(date -Is) $(df -h / | tail -1)" >>"$LOG"
