# Issue draft for huggingface/lerobot

**Title:** FSDP resume fails when a policy has trainable parameters that get no gradient (`Missing key in checkpoint state_dict: optimizer.state...`)

## What happens

Resuming an FSDP2 run fails while loading the sharded optimizer state if the policy has trainable parameters that never receive a gradient. With Multi-task DiT:

```
RuntimeError: Missing key in checkpoint state_dict: optimizer.state.observation_encoder.vision_encoder.model.post_layernorm.weight.step.
```

## Why

- `CLIPVisionEncoder.forward` uses `last_hidden_state[:, 0]`. `CLIPVisionModel` still computes the pooled output through `post_layernorm`, but it does not reach the loss, so `post_layernorm` never gets a gradient and AdamW never creates state for it.
- At save time, `save_sharded_optimizer` writes the state that exists, without `post_layernorm`.
- At load time, torch's `_init_optim_state` (via accelerate's `load_fsdp_optimizer`) builds state for every trainable parameter, so the DCP load expects `post_layernorm` keys that the checkpoint lacks.

DDP resume of the same policy works (non-sharded optimizer path). FSDP resume of ACT, whose parameters all get gradients, works.

## Reproduce

```bash
accelerate launch --num_processes=2 $(which lerobot-train) \
  --policy.type=multi_task_dit --policy.horizon=32 --policy.n_action_steps=24 \
  --dataset.repo_id=lerobot/aloha_sim_transfer_cube_human --dataset.episodes='[0]' \
  --parallelism.dp_shard=2 --accelerator.fsdp.wrap_modules='["CLIPEncoderLayer","TransformerBlock"]' \
  --batch_size=1 --steps=2 --save_freq=2 --output_dir=outputs/dit-fsdp
accelerate launch --num_processes=2 $(which lerobot-train) --resume=true \
  --config_path=outputs/dit-fsdp/checkpoints/000002/pretrained_model/train_config.json --steps=4
```

(Multi-task DiT declares no `_fsdp_wrap_modules`, hence the explicit `wrap_modules`.)

## Possible fixes

- Initialise the optimizer state for all trainable parameters before the first sharded save, so the saved and expected keys match.
- Or load the optimizer shards non-strictly for parameters without state.
- For this policy specifically, freezing `post_layernorm` (or not computing the pooled output) avoids it.

Seen on 2× AMD MI300X (ROCm 7.14, torch 2.12, accelerate 1.15). The logic is device-independent.
