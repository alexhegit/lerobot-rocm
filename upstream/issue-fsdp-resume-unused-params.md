## What happens

Resuming an FSDP2 run fails while loading the sharded optimizer state if the policy has trainable parameters that never receive a gradient. With Multi-task DiT:

```
RuntimeError: Missing key in checkpoint state_dict: optimizer.state.observation_encoder.vision_encoder.model.post_layernorm.weight.step.
```

## Why

- `CLIPVisionEncoder.forward` uses `last_hidden_state[:, 0]`. `CLIPVisionModel` still computes its pooled output through `post_layernorm`, but that output does not reach the loss, so `post_layernorm` never gets a gradient and AdamW never creates state for it.
- `save_sharded_optimizer` (accelerate's `save_fsdp_optimizer`) writes the optimizer state that exists, without `post_layernorm`.
- On resume, `load_sharded_optimizer` (accelerate's `load_fsdp_optimizer`) calls torch's `get_optimizer_state_dict`, whose `_init_optim_state` creates state for every trainable parameter, so `dcp.load` expects `post_layernorm` keys the checkpoint does not have.

DDP resume of the same policy works (non-sharded optimizer path). FSDP resume of ACT, whose parameters all get gradients, works.

## Reproduce

With LeRobot (2 GPUs; Multi-task DiT declares no `_fsdp_wrap_modules`, hence `wrap_modules`):

```bash
accelerate launch --num_processes=2 $(which lerobot-train) \
  --policy.type=multi_task_dit --policy.horizon=32 --policy.n_action_steps=24 \
  --dataset.repo_id=lerobot/aloha_sim_transfer_cube_human --dataset.episodes='[0]' \
  --parallelism.dp_shard=2 --accelerator.fsdp.wrap_modules='["CLIPEncoderLayer","TransformerBlock"]' \
  --batch_size=1 --steps=2 --save_freq=2 --output_dir=outputs/dit-fsdp
accelerate launch --num_processes=2 $(which lerobot-train) --resume=true \
  --config_path=outputs/dit-fsdp/checkpoints/000002/pretrained_model/train_config.json --steps=4
```

Device-independent, with plain torch on CPU (`torchrun --nproc-per-node=2 repro.py /tmp/ckpt`, gloo), following the same save/load calls:

```python
import sys, torch, torch.distributed as dist, torch.distributed.checkpoint as dcp
from torch import nn
from torch.distributed.checkpoint.state_dict import get_optimizer_state_dict, set_optimizer_state_dict
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.used = nn.Linear(8, 8)
        self.unused = nn.Linear(8, 8)  # trainable, but never reaches the loss
    def forward(self, x):
        return self.used(x)

def build(mesh):
    model = Model()
    fully_shard(model.used, mesh=mesh)
    fully_shard(model, mesh=mesh)
    return model, torch.optim.AdamW(model.parameters())

dist.init_process_group("gloo")
mesh = init_device_mesh("cpu", (dist.get_world_size(),))
model, opt = build(mesh)
model(torch.randn(4, 8)).sum().backward(); opt.step(); opt.zero_grad()
dcp.save({"optimizer": get_optimizer_state_dict(model, opt)}, checkpoint_id=sys.argv[1])

model2, opt2 = build(mesh)
state = {"optimizer": get_optimizer_state_dict(model2, opt2)}
dcp.load(state, checkpoint_id=sys.argv[1])  # Missing key ... optimizer.state.unused.weight.step
set_optimizer_state_dict(model2, opt2, state["optimizer"])
```

The load raises `Missing key in checkpoint state_dict: optimizer.state.unused.weight.step.`; with `unused` included in `forward` it loads.

## Possible fixes

- Initialise the optimizer state for all trainable parameters before the first sharded save (e.g. the same `_init_optim_state` the load path runs), so saved and expected keys match.
- Or load the optimizer shards non-strictly for parameters without state.
- For this policy specifically, freezing `post_layernorm` (or not computing the pooled output) avoids it.

The underlying behaviour is in torch's `get_optimizer_state_dict` + DCP; LeRobot is where FSDP resume breaks for users, so filing here first.

## Environment

- LeRobot `main` (`e5be0c16`), accelerate 1.15, torch 2.12.
- Seen on 2× AMD MI300X (ROCm 7.14); the CPU repro above reproduces it without any GPU. Not run on NVIDIA.
