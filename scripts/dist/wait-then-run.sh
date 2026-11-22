#!/bin/bash
# Wait until the given queue logs report QUEUE END, then run a queue on the freed GPUs.
# Usage: wait-then-run.sh <queue-file> <log-name> <log-to-wait>...
set -u
ROOT=${ROOT:-/data/lerobot/outputs/rocm-train}
queue=$1 log=$2
shift 2
for w in "$@"; do
  until grep -q "QUEUE END" "$ROOT/$w.log" 2>/dev/null; do sleep 60; done
done
cd "$ROOT/dist" && exec bash run-dist-queue.sh "$queue" "$log"
