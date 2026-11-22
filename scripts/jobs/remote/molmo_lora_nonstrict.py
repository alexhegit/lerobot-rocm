"""Load MolmoAct2 with strict=False so fresh LoRA keys are not required."""
import sys

from lerobot.policies.molmoact2.modeling_molmoact2 import MolmoAct2Policy

_orig = MolmoAct2Policy.from_pretrained.__func__


def _loose(cls, pretrained_name_or_path, *args, strict=True, **kwargs):
    print("molmo strict_override False", flush=True)
    return _orig(cls, pretrained_name_or_path, strict=False, **kwargs)


MolmoAct2Policy.from_pretrained = classmethod(_loose)

from lerobot.scripts.lerobot_train import main

if __name__ == "__main__":
    raise SystemExit(main())
