"""Load MolmoAct2 base weights first, then attach LoRA."""

import sys

from lerobot.policies.molmoact2.modeling_molmoact2 import MolmoAct2Policy

_orig_init = MolmoAct2Policy.__init__
_orig_from = MolmoAct2Policy.from_pretrained.__func__
_defer = {"lora": False}


def _init(self, config, *args, **kwargs):
    _defer["lora"] = config.train_mode_vlm == "lora"
    if _defer["lora"]:
        config.train_mode_vlm = "fft"
    _orig_init(self, config, *args, **kwargs)
    if _defer["lora"]:
        config.train_mode_vlm = "lora"


def _from_pretrained(cls, pretrained_name_or_path, *args, strict=True, **kwargs):
    policy = _orig_from(cls, pretrained_name_or_path, strict=True, **kwargs)
    if _defer["lora"]:
        policy._apply_lora_adapters()
        trainable = sum(param.numel() for param in policy.parameters() if param.requires_grad)
        print(f"trainable params {trainable}", flush=True)
    return policy


MolmoAct2Policy.__init__ = _init
MolmoAct2Policy.from_pretrained = classmethod(_from_pretrained)

from lerobot.scripts.lerobot_train import main

if __name__ == "__main__":
    raise SystemExit(main())
