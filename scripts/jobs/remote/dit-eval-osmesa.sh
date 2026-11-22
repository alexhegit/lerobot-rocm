#!/bin/bash
set -euo pipefail
apt-get update -qq
apt-get install -y -qq libosmesa6 libglu1-mesa
export MUJOCO_GL=osmesa
bash /outputs/rocm-train/jobs/dit-eval.sh
