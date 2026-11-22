"""Launch lawam training after casting every floating parameter to float32.

FSDP rejects the released checkpoint because Qwen stays bf16 while the rest of
the policy is fp32. Casting the safetensors file is not enough: those bf16
parameters are created again when the backbone loads.
"""

import sys

import torch
from lerobot.policies.lawam.modeling_lawam import LaWAMPolicy

_orig = LaWAMPolicy._load_as_safetensor.__func__


def _load_fp32(cls, model, model_file, map_location, strict):
    model = _orig(cls, model, model_file, map_location, strict)
    converted = 0
    for param in model.parameters():
        if param.is_floating_point() and param.dtype != torch.float32:
            param.data = param.data.float()
            converted += 1
    dtypes = {str(param.dtype) for param in model.parameters() if param.is_floating_point()}
    print(f"lawam cast_params={converted} dtypes={sorted(dtypes)}", flush=True)
    return model


LaWAMPolicy._load_as_safetensor = classmethod(_load_fp32)

from lerobot.scripts.lerobot_train import main

if __name__ == "__main__":
    raise SystemExit(main())
