"""Minimal NATTEN repro on ROCm: na3d with flex-fna + torch_compile at the flux3 Video VAE shape.

Usage: python repro-natten-flex3d.py <T> [causal|noncausal]
"""

import sys
import time

import natten
import torch
from natten.backends.configs import checks
from natten.functional import na3d

t = int(sys.argv[1])
causal = (sys.argv[2] if len(sys.argv) > 2 else "causal") == "causal"
print("torch", torch.__version__, "hip", torch.version.hip, "natten", natten.__version__)
print("_FLEX_COMPILE_SUPPORTED (stock)", checks._FLEX_COMPILE_SUPPORTED)
checks._FLEX_COMPILE_SUPPORTED = True
natten.allow_flex_compile()

shape = (1, t, 64, 128, 4, 64)
q, k, v = (torch.randn(shape, device="cuda", dtype=torch.bfloat16) for _ in range(3))
print("shape", shape, "tokens", t * 64 * 128, "causal", causal)
start = time.perf_counter()
with torch.no_grad():
    out = na3d(
        q,
        k,
        v,
        kernel_size=(5, 5, 5),
        is_causal=(causal, False, False),
        backend="flex-fna",
        torch_compile=True,
    )
torch.cuda.synchronize()
print("OK", tuple(out.shape), "seconds %.1f" % (time.perf_counter() - start))
