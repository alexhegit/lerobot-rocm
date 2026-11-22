#!/bin/bash
# flux3 on ROCm: NATTEN flex-fna with a compiled block mask, patched at runtime (no source change).
set -euo pipefail
uv pip install 'natten==0.21.7' 'torchao>0.16'
python3 - <<'PY'
import torch, torchvision
assert torch.version.hip, "ROCm torch was replaced"
assert "+rocm" in torchvision.__version__, torchvision.__version__
import natten, torchao
print("env", torch.__version__, torch.version.hip, torchvision.__version__, "natten", natten.__version__, "torchao", torchao.__version__)
PY
export F3_NATTEN_BACKEND=flex-fna
python3 - "$@" <<'PY'
import sys
import natten
from lerobot.policies.flux3.f3 import video_vae

from natten.backends.configs import checks as natten_checks

# NATTEN's get_device_cc() returns 0 when torch.version.cuda is None, which disables compiled flex on ROCm.
natten_checks._FLEX_COMPILE_SUPPORTED = True
natten.allow_flex_compile()


# NATTEN 0.21 reads backend/torch_compile as top-level na{2,3}d arguments;
# attention_kwargs only reaches the self-attention / additional-KV FMHA path.
def _lift(fn):
    def wrapped(q, k, v, *, attention_kwargs=None, **kw):
        ak = dict(attention_kwargs or {})
        backend = ak.pop("backend", None)
        if backend == "flex-fna":
            kw["torch_compile"] = True
        elif "run_persistent_kernel" in ak:
            kw["run_persistent_kernel"] = ak.pop("run_persistent_kernel")
        print("NA_CALL", fn.__name__, tuple(q.shape), q.dtype, kw.get("kernel_size"), kw.get("is_causal"), flush=True)
        return fn(q, k, v, backend=backend, **kw)

    return wrapped


video_vae.na2d = _lift(video_vae.na2d)
video_vae.na3d = _lift(video_vae.na3d)
print("PATCH na2d/na3d backend top-level, flex-fna torch_compile=True")

from lerobot.scripts.lerobot_train import main

sys.argv = [
    "lerobot-train",
    "--policy.path=black-forest-labs/flux-3-action-so101",
    "--policy.device=cuda",
    "--policy.push_to_hub=false",
    "--policy.attn_mode=torch",
    "--policy.gradient_checkpointing=true",
    "--peft.method_type=LORA",
    "--peft.r=8",
    "--peft.lora_alpha=8",
    "--accelerator.mixed_precision=bf16",
    '--rename_map={"observation.images.top":"observation.images.scene"}',
    "--dataset.repo_id=lerobot/svla_so100_pickplace",
    "--dataset.episodes=[0]",
    "--batch_size=1",
    "--steps=2",
    "--env_eval_freq=0",
    "--save_freq=2",
    "--log_freq=1",
    "--num_workers=0",
    "--trackio.enable=false",
    "--output_dir=/outputs/rocm-train/flux3-flexcompile5",
]
main()
PY
