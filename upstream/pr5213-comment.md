Tested this PR on AMD GPUs (ROCm), where NATTEN has no wheels and only its Flex backend builds from source. With this PR, FLUX3 trains without NATTEN.

**Setup:** 1x AMD Instinct MI300X, ROCm 7.14, `torch 2.12.0+rocm7.14`, `transformers 5.22.2`, `torchao 0.18`, NATTEN not installed. PR head `141d22b8`.

**Training smoke (1 GPU, 2 steps):**

```bash
lerobot-train --policy.path=black-forest-labs/flux-3-action-so101 --policy.device=cuda \
  --policy.attn_mode=torch --policy.gradient_checkpointing=true \
  --peft.method_type=LORA --peft.r=8 --peft.lora_alpha=8 --accelerator.mixed_precision=bf16 \
  --rename_map='{"observation.images.top":"observation.images.scene"}' \
  --dataset.repo_id=lerobot/svla_so100_pickplace --dataset.episodes='[0]' \
  --batch_size=1 --steps=2 --save_freq=2 --policy.push_to_hub=false
```

- Passed in 87 s. The VAE warned once that it uses plain PyTorch attention, as intended.
- 63M learnable params (LoRA). Loss 0.137 → 0.083 (`video_mse` 0.119 → 0.072, `action_mse` 0.050 → 0.031). Peak memory 27.6 GB.
- Step 2 wrote `adapter_model.safetensors`.

**Tests:** `pytest tests/policies/flux3` on the GPU: 155 passed, 5 skipped (3 need MPS, 2 need NATTEN and CUDA).

**For comparison, on `main` with NATTEN 0.21.7 built from source:** the backend in `attention_kwargs` is ignored (commit 1 here), NATTEN falls back to uncompiled `flex-fna`, and `create_block_mask` tries to allocate 882 GiB for the `(1, 42, 64, 128, 4, 64)` `na3d` call. Compiled flex is gated off on ROCm by NATTEN itself (`get_device_cc()` returns 0 when `torch.version.cuda` is `None`). So the plain PyTorch path in commit 2 is currently the only way to run FLUX3 on AMD GPUs.

I don't have an NVIDIA GPU, so I could not compare against the CUTLASS kernels.
