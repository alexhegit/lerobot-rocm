"""Compare a policy's action chunks on the ROCm GPU and on the CPU for the same dataset frames.

Flow-matching policies get the same noise on both devices (drawn on the CPU with a fixed seed), so any
difference comes from the device's arithmetic. Usage:
  python numerics_check.py <policy_path> <dataset_repo_id> [rename_json] [revision]
"""

import inspect
import json
import os
import sys

import torch
from huggingface_hub import snapshot_download

from lerobot.datasets import LeRobotDataset
from lerobot.policies.factory import get_policy_class, make_pre_post_processors

path, repo_id = sys.argv[1], sys.argv[2]
rename = json.loads(sys.argv[3]) if len(sys.argv) > 3 and sys.argv[3] else {}
revision = sys.argv[4] if len(sys.argv) > 4 else None
if not os.path.isdir(path):
    path = snapshot_download(path, revision=revision)

policy_type = json.load(open(f"{path}/config.json"))["type"]
policy_cls = get_policy_class(policy_type)

ds = LeRobotDataset(repo_id, episodes=[0, 1])
frames = [ds[i] for i in torch.linspace(0, len(ds) - 1, 16).long().tolist()]


def observations(item):
    obs = {rename.get(k, k): v for k, v in item.items() if k.startswith("observation") or k == "task"}
    return obs


def run(device: str, dtype: torch.dtype | None):
    # Any weights that loading leaves randomly initialised must be the same in every run.
    torch.manual_seed(1234)
    policy = policy_cls.from_pretrained(path).to(device).eval()
    if dtype is not None:
        policy = policy.to(dtype)
    pre, _ = make_pre_post_processors(policy.config, pretrained_path=path,
                                      preprocessor_overrides={"device_processor": {"device": device}})
    chunks = []
    gen = torch.Generator().manual_seed(0)
    for item in frames:
        batch = pre(observations(item))
        kwargs = {}
        # inspect.signature follows the decorator's __wrapped__; __code__ would see only *args/**kwargs.
        if "noise" in inspect.signature(policy.predict_action_chunk).parameters:
            shape = (1, policy.config.chunk_size, policy.config.max_action_dim)
            kwargs["noise"] = torch.randn(shape, generator=gen).to(device)
        with torch.no_grad():
            chunk = policy.predict_action_chunk(batch, **kwargs)
        chunks.append(chunk.float().cpu())
    del policy
    return torch.cat(chunks)


cpu32 = run("cpu", torch.float32)
gpu32 = run("cuda", torch.float32)
gpu32_again = run("cuda", torch.float32)
gpu_default = run("cuda", None)
# Same fp32 run with SDPA restricted to the math kernel (no flash / memory-efficient kernels).
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
gpu32_math = run("cuda", torch.float32)
scale = cpu32.abs().mean().item()
d = (gpu32_again - gpu32).abs()
print(f"NUMERICS {policy_type} gpu_fp32 repeat vs gpu_fp32: max_abs={d.max().item():.3e} mean_abs={d.mean().item():.3e}", flush=True)
for name, other in [("gpu_fp32", gpu32), ("gpu_default_dtype", gpu_default), ("gpu_fp32_math_sdpa", gpu32_math)]:
    diff = (other - cpu32).abs()
    print(f"NUMERICS {policy_type} {name} vs cpu_fp32: max_abs={diff.max().item():.3e} "
          f"mean_abs={diff.mean().item():.3e} mean_abs/mean|a|={diff.mean().item() / scale:.3e} "
          f"chunks={tuple(cpu32.shape)}", flush=True)
