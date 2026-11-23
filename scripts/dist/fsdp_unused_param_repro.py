"""CPU-only repro: FSDP2 optimizer resume fails when a trainable parameter never gets a gradient.

Follows accelerate's save_fsdp_optimizer / load_fsdp_optimizer: get_optimizer_state_dict + dcp.save,
then get_optimizer_state_dict on a fresh optimizer (which initialises state for every parameter) +
dcp.load. Run: torchrun --nproc-per-node=2 fsdp_unused_param_repro.py <checkpoint_dir> [--all-used]
"""

import sys

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch import nn
from torch.distributed.checkpoint.state_dict import get_optimizer_state_dict, set_optimizer_state_dict
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard

ALL_USED = "--all-used" in sys.argv


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.used = nn.Linear(8, 8)
        self.unused = nn.Linear(8, 8)  # trainable, but its output never reaches the loss

    def forward(self, x):
        return self.used(x) + (self.unused(x) if ALL_USED else 0)


def build(mesh):
    torch.manual_seed(0)
    model = Model()
    fully_shard(model.used, mesh=mesh)
    fully_shard(model, mesh=mesh)
    return model, torch.optim.AdamW(model.parameters(), lr=1e-3)


dist.init_process_group("gloo")
mesh = init_device_mesh("cpu", (dist.get_world_size(),))
ckpt = sys.argv[1]

model, opt = build(mesh)
model(torch.randn(4, 8)).sum().backward()
opt.step()
opt.zero_grad()
dcp.save({"optimizer": get_optimizer_state_dict(model, opt)}, checkpoint_id=ckpt)

model2, opt2 = build(mesh)
state = {"optimizer": get_optimizer_state_dict(model2, opt2)}
try:
    dcp.load(state, checkpoint_id=ckpt)
    set_optimizer_state_dict(model2, opt2, state["optimizer"])
    result = "LOAD_OK"
except Exception as exc:  # noqa: BLE001
    result = f"LOAD_FAILED {type(exc).__name__}: {str(exc).splitlines()[-1][:200]}"
if dist.get_rank() == 0:
    print(f"REPRO all_used={ALL_USED} torch={torch.__version__} {result}", flush=True)
dist.destroy_process_group()
