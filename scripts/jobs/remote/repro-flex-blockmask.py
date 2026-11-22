"""Minimal repro: compiled flex-attention create_block_mask at ~1M tokens on ROCm.

Usage: python repro-flex-blockmask.py <seq_len> [block_size]
"""

import sys
import time

import torch
from torch.nn.attention.flex_attention import create_block_mask

seq_len = int(sys.argv[1])
block = int(sys.argv[2]) if len(sys.argv) > 2 else 128
window = 2048

print("torch", torch.__version__, "hip", torch.version.hip, "cuda", torch.version.cuda)
print("device", torch.cuda.get_device_name(0), "seq_len", seq_len, "block", block)


def local_window(b, h, q_idx, kv_idx):
    return (q_idx - kv_idx).abs() <= window


compiled = torch.compile(create_block_mask)
start = time.perf_counter()
bm = compiled(local_window, None, None, seq_len, seq_len, device="cuda", BLOCK_SIZE=(block, block))
torch.cuda.synchronize()
print("OK", bm.shape, "sparsity %.4f" % bm.sparsity(), "seconds %.1f" % (time.perf_counter() - start))
