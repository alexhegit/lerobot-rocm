## Summary

On ROCm, NATTEN only builds its Flex backends (no ROCm wheels on whl.natten.org). Two problems make `flex-fna` impractical there:

1. Compiled Flex (`torch_compile=True`) is rejected on ROCm even after `natten.allow_flex_compile()`, because `get_device_cc()` returns 0 when `torch.version.cuda` is `None`.
2. When that gate is forced open, large 3D masks sometimes fail to launch while inductor autotunes `create_block_mask`, and compile is very slow when it works.

Without compilation, `create_block_mask` materialises the dense mask: one `(1, 42, 64, 128, 4, 64)` `na3d` call tried to allocate 882 GiB.

## Environment

- AMD Instinct MI300X, ROCm 7.14 (HIP 7.14.60850)
- torch `2.12.0+rocm7.14.0a20260608`
- natten `0.21.7`, installed from the sdist (`pip install natten==0.21.7`)

## 1. `torch_compile=True` is rejected on ROCm

`src/natten/utils/device.py`:

```python
def get_device_cc(device=None) -> int:
    if torch.cuda.is_available() and torch.version.cuda and (device is None or is_cuda(device)):
        ...
    return 0
```

On ROCm `torch.version.cuda` is `None`, so this returns 0. `utils/environment.py` then sets `_IS_TORCH_COMPILE_SUPPORTED = _TORCH_VERSION >= [2, 6] and get_device_cc() >= 70` to False, so `_FLEX_COMPILE_SUPPORTED` is False and

```
RuntimeError: Can't run NATTEN with Flex Attention (compiled).)
```

`torch.compile` and Flex Attention themselves work on ROCm. For example, this passes on the same machine at 1,048,576 tokens in 7.1 s:

```python
import torch
from torch.nn.attention.flex_attention import create_block_mask

n = 1_048_576
local = lambda b, h, q, kv: (q - kv).abs() <= 2048
torch.compile(create_block_mask)(local, None, None, n, n, device="cuda", BLOCK_SIZE=(128, 128))
```

Suggestion: treat HIP devices (`torch.version.hip`) as compile-capable, or gate on `torch.compile` availability rather than on CUDA compute capability.

## 2. Forced on, large 3D masks fail to launch or compile very slowly

```python
import sys
import natten, torch
from natten.backends.configs import checks
from natten.functional import na3d

T, causal = int(sys.argv[1]), sys.argv[2] == "causal"
checks._FLEX_COMPILE_SUPPORTED = True  # bypass issue 1
natten.allow_flex_compile()
q, k, v = (torch.randn(1, T, 64, 128, 4, 64, device="cuda", dtype=torch.bfloat16) for _ in range(3))
with torch.no_grad():
    na3d(q, k, v, kernel_size=(5, 5, 5), is_causal=(causal, False, False),
         backend="flex-fna", torch_compile=True)
torch.cuda.synchronize()
```

Each row is a fresh process:

| T | tokens | causal | result |
|---|---|---|---|
| 8 | 65,536 | yes | OK, 42.9 s including compile |
| 42 | 344,064 | no | `RuntimeError: CUDA driver error: 1` |
| 42 | 344,064 | yes | OK, but 861.4 s including compile |

Inside the FLUX3 Video VAE (same shape and window, causal, inside a training step with the rest of the model on the GPU), the causal case failed twice with `CUDA driver error: 1`, so the failure does not depend on shape alone.

The error is raised while inductor autotunes the compiled `create_block_mask` kernel (`triton_heuristics.benchmark_all_configs` → `kernel_call`). With `TORCHINDUCTOR_USE_STATIC_CUDA_LAUNCHER=0` it becomes `Triton Error [HIP]: Code: 1, Messsage: invalid argument`. Since the plain 1-D mask above compiles at 1M tokens in seconds, this looks specific to the 3D NATTEN mask under inductor on ROCm, perhaps one of the autotune configs requesting an invalid launch.

## Context

Found while running the FLUX3 Video VAE in LeRobot on AMD GPUs (shape and window above are the VAE's). LeRobot PR huggingface/lerobot#5213 adds a plain PyTorch neighborhood attention used when NATTEN is missing, which is how FLUX3 trains on ROCm today. Working compiled `flex-fna` would let ROCm users use NATTEN instead.

I don't have an NVIDIA GPU to compare against.
