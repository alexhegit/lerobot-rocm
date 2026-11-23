## What happens

`--accelerator.fsdp.min_num_params` (the size-based wrap policy) fails at `accelerator.prepare()` for policies whose layers live in an `nn.ModuleList`, which is most of them:

```
ValueError: fully_shard does not support containers that do not implement forward: ModuleList(
```

## Why

`FSDPConfig.build_plugin` maps `min_num_params` to accelerate's `size_based_wrap`. accelerate's FSDP2 size policy (`fsdp2_prepare_auto_wrap_policy`) returns `True` for any module whose total parameter count exceeds the threshold, including the `nn.ModuleList` holding the layers, and torch's `fully_shard` rejects modules without a `forward`.

Only six policies declare `_fsdp_wrap_modules` (ACT, WALL-OSS 0.5, FastWAM, LingBot-VLA v2, LaWAM, Cosmos3). For the others, `set_fsdp_wrap_modules` says to pass `--accelerator.fsdp.wrap_modules` or `--accelerator.fsdp.min_num_params`, but in practice only class names work.

## Reproduce

```bash
accelerate launch --num_processes=2 $(which lerobot-train) \
  --policy.type=multi_task_dit --policy.horizon=32 --policy.n_action_steps=24 \
  --dataset.repo_id=lerobot/aloha_sim_transfer_cube_human --dataset.episodes='[0]' \
  --parallelism.dp_shard=2 --accelerator.fsdp.min_num_params=100000 \
  --batch_size=1 --steps=2
```

Also failed for VQ-BeT and SmolVLA (and, with other options set, for DiT with gradient accumulation, HSDP, fp16/bf16 and `reshard_after_forward=false`). The same runs pass with class names, e.g. `--accelerator.fsdp.wrap_modules='["CLIPEncoderLayer","TransformerBlock"]'` for DiT, `'["Block"]'` for VQ-BeT.

## Possible fixes

- Wrap the size policy so it skips containers (`nn.ModuleList`, `nn.ModuleDict`) and modules without their own `forward`, falling through to their children.
- Or document that `min_num_params` does not work with FSDP2 and point to `wrap_modules`, perhaps with the class names of the policies that declare none.

Related, separately: FSDP `cpu_offload=true` with the default `init_process_group.backend="nccl"` fails with `No backend type associated with device type cpu`; #5065 addresses it (`cpu:gloo,cuda:nccl` worked for us).

## Environment

- LeRobot `e5be0c16`-era code in the image (accelerate 1.15, torch 2.12).
- 2× AMD MI300X (ROCm 7.14). The failing check is torch's own `fully_shard` validation, so it is device-independent. Not run on NVIDIA.
