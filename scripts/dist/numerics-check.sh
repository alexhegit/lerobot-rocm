#!/bin/bash
# ROCm GPU vs CPU action chunks on dataset frames: ACT (official transfer-cube, deterministic) and
# SmolVLA (smolvla_libero at the documented revision, fixed noise).
set -uo pipefail
export AMD_CPU_AFFINITY=0
J=/outputs/rocm-train/jobs/numerics_check.py
# ACT done in the first run (max abs diff 2.7e-6).
python3 $J lerobot/smolvla_libero lerobot/libero \
  '{"observation.images.image": "observation.images.camera1", "observation.images.image2": "observation.images.camera2"}' \
  a24f1f380461b665158d52acd140152c79e8b36d 2>&1 | grep -aiE "NUMERICS|Error|error:|missing|newly initialized|not initialized" | grep -v Warning | cut -c1-300
