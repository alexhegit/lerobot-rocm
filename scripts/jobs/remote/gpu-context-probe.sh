#!/bin/bash
# Which import makes a CPU-only process (like a DataLoader worker) open a HIP context on the GPU?
# Each probe imports something, reports torch.cuda.is_initialized(), and sleeps so the host can list
# the processes holding GPU memory.
set -uo pipefail
probe() {
  local name=$1 code=$2
  python3 -c "
import os, time
$code
import torch
print('PROBE $name pid', os.getpid(), 'cuda_initialized', torch.cuda.is_initialized(), flush=True)
time.sleep(25)
" &
}
probe torch "import torch"
probe aiter "import torch, aiter"
probe dataset "from lerobot.datasets import LeRobotDataset"
probe act "from lerobot.policies.act.modeling_act import ACTPolicy"
probe train "import lerobot.scripts.lerobot_train"
sleep 20
echo PROBES_RUNNING
wait
