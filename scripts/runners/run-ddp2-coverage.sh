#!/bin/bash
# 2-GPU DDP for models that only had single-GPU coverage (smoke-report section 11) and FLUX3 on PR #5213.
# Derives each job from the passing single-GPU script; deletes the checkpoint after a PASS.
set -u
ROOT=/data/lerobot/outputs/rocm-train
LOG="$ROOT/run-ddp2.log"
JOBS=(cover-vqbet cover-tdmpc-pusht cover-dm05 cover-evo1 cover-jepa-rename cover-wallx
      cover-groot2 cover-fineart3c cover-lingbotva-torchao cover-longwam2 flux3-pr5213)

for base in "${JOBS[@]}"; do
  name="${base}-ddp2"
  src="$ROOT/jobs/${base}.sh"
  dst="$ROOT/jobs/${name}.sh"
  sed -e 's|^lerobot-train|accelerate launch --num_processes=2 "$(which lerobot-train)"|' \
      -e "s|--output_dir=/outputs/rocm-train/${base}|--output_dir=/outputs/rocm-train/${name}|" \
      "$src" >"$dst"
  chmod +x "$dst"
done

for base in "${JOBS[@]}"; do
  name="${base}-ddp2"
  free_gb=$(df -BG --output=avail / | tail -1 | tr -dc '0-9')
  if (( free_gb < 60 )); then
    echo "SKIP ${name} free=${free_gb}G" | tee -a "$LOG"
    echo "${name} SKIP free=${free_gb}G" >>"$ROOT/results.txt"
    continue
  fi
  extra=()
  [[ "$base" == flux3-pr5213 ]] && extra=(-v "$ROOT/pr5213:/pr")
  echo "----- $(date -Is) BEGIN ${name} devices=0,1 free=${free_gb}G -----" >>"$LOG"
  s=$(date +%s)
  timeout --signal=TERM 5400 docker run --rm --name "rocm-${name}" --ipc=host --shm-size 16g \
    --device=/dev/kfd --device=/dev/dri --group-add 44 --group-add 992 \
    -e HIP_VISIBLE_DEVICES=0,1 -e MIOPEN_FIND_MODE=2 \
    -v /data/lerobot/outputs:/outputs -v /root/.cache/huggingface:/root/.cache/huggingface \
    "${extra[@]}" lerobot-rocm bash "/outputs/rocm-train/jobs/${name}.sh" >>"$LOG" 2>&1
  ec=$?
  e=$(( $(date +%s) - s ))
  v=FAIL; (( ec == 0 )) && v=PASS
  ws=$(awk "/BEGIN ${name} /,0" "$LOG" | grep -ao 'dp_world_size": *[0-9]*' | tail -1)
  echo "RESULT ${name} ${v} exit=${ec} seconds=${e} ${ws}" >>"$LOG"
  echo "${name} ${v} exit=${ec} seconds=${e} ${ws}" >>"$ROOT/results.txt"
  [[ "$v" == PASS ]] && rm -rf "$ROOT/$name/checkpoints"
done
echo "ddp2 coverage end $(date -Is) $(df -h / | tail -1)" >>"$LOG"
