"""lerobot-train with torch.profiler around update_policy for steps 30-39 on rank 0.

Prints the top ops by device time and by CPU time, and the wall time per profiled step.
"""

import os
import time

import torch
from torch.profiler import ProfilerActivity, profile

from lerobot.scripts import lerobot_train

_original = lerobot_train.update_policy
_state = {"step": 0, "prof": None, "t": []}
_RANK0 = os.environ.get("RANK", "0") == "0"


def _update_policy(*args, **kwargs):
    _state["step"] += 1
    step = _state["step"]
    if _RANK0 and step == 30:
        _state["prof"] = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA])
        _state["prof"].__enter__()
    torch.cuda.synchronize()
    start = time.perf_counter()
    out = _original(*args, **kwargs)
    torch.cuda.synchronize()
    if 30 <= step < 40:
        _state["t"].append(time.perf_counter() - start)
    if _RANK0 and step == 39 and _state["prof"] is not None:
        prof = _state["prof"]
        prof.__exit__(None, None, None)
        print(f"PROFILE_STEP_S {[round(t, 4) for t in _state['t']]}", flush=True)
        print("PROFILE_BY_DEVICE\n" + prof.key_averages().table(sort_by="self_device_time_total", row_limit=15), flush=True)
        print("PROFILE_BY_CPU\n" + prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=15), flush=True)
        _state["prof"] = None
    return out


lerobot_train.update_policy = _update_policy

if __name__ == "__main__":
    lerobot_train.main()
