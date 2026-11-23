"""Print the process CPU affinity size after each step that might change it."""

import os
import sys


def show(label: str) -> None:
    print(f"AFFINITY {label}: {len(os.sched_getaffinity(0))} cpus", flush=True)


show("start")
import torch  # noqa: E402

show("import torch")
torch.cuda.device_count()
show("torch.cuda.device_count")
torch.zeros(1, device="cuda")
show("cuda tensor")
if "--aiter" in sys.argv:
    import aiter  # noqa: E402, F401

    show("import aiter")
import accelerate.commands.launch  # noqa: E402, F401

show("import accelerate.commands.launch")
