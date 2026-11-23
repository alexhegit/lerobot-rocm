"""Which parameters of a checkpoint come out different when it is loaded twice with different seeds?

Those are left randomly initialised by from_pretrained. Usage: load_determinism.py <repo_id> <revision>
"""

import json
import sys

import torch
from huggingface_hub import snapshot_download

from lerobot.policies.factory import get_policy_class

path = snapshot_download(sys.argv[1], revision=sys.argv[2] if len(sys.argv) > 2 else None)
cls = get_policy_class(json.load(open(f"{path}/config.json"))["type"])


def load(seed: int) -> dict[str, torch.Tensor]:
    torch.manual_seed(seed)
    return {k: v.detach().float().cpu().clone() for k, v in cls.from_pretrained(path).state_dict().items()}


a, b = load(1), load(2)
differ = [k for k in a if not torch.equal(a[k], b[k])]
total = sum(a[k].numel() for k in differ)
print(f"LOAD_DETERMINISM params={len(a)} differing={len(differ)} numel={total}", flush=True)
for k in differ[:40]:
    print(f"LOAD_DIFF {k} {tuple(a[k].shape)}", flush=True)
