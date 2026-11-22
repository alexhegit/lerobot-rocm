"""Find None fields in one Molmo/SO100 batch, before and after torchao."""

import dataclasses
import json
import subprocess
import sys
from pathlib import Path

SNAP = Path(
    "/root/.cache/huggingface/hub/models--lerobot--MolmoAct2-SO100_101-LeRobot/"
    "snapshots/a93b5fcdce4c1e3cc7688c939fc943bf4b092a8a"
)
COMPAT = Path("/outputs/rocm-train/molmo-lora-compat")


def prepare_config() -> None:
    from lerobot.policies.molmoact2.configuration_molmoact2 import MolmoAct2Config

    COMPAT.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((SNAP / "config.json").read_text())
    allowed = {field.name for field in dataclasses.fields(MolmoAct2Config)} | {"type"}
    for key in sorted(set(cfg) - allowed):
        cfg.pop(key)
    cfg["train_mode_vlm"] = "lora"
    (COMPAT / "config.json").write_text(json.dumps(cfg))


def none_keys(item, prefix=""):
    found = []
    if item is None:
        return [prefix or "<root>"]
    if isinstance(item, dict):
        for key, value in item.items():
            found.extend(none_keys(value, f"{prefix}.{key}" if prefix else key))
    elif isinstance(item, list | tuple):
        for index, value in enumerate(item):
            if value is None:
                found.append(f"{prefix}[{index}]")
    return found


def scan(label: str) -> None:
    from lerobot.datasets.factory import resolve_delta_timestamps
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.molmoact2.configuration_molmoact2 import MolmoAct2Config

    cfg = MolmoAct2Config.from_pretrained(COMPAT)
    base = LeRobotDataset("lerobot/svla_so100_pickplace", episodes=[0])
    rename = {
        "observation.images.top": "observation.images.cam0",
        "observation.images.wrist": "observation.images.cam1",
    }
    deltas = resolve_delta_timestamps(cfg, base.meta, rename)
    print(label, "deltas", None if deltas is None else {k: len(v) for k, v in deltas.items()}, flush=True)
    ds = LeRobotDataset("lerobot/svla_so100_pickplace", episodes=[0], delta_timestamps=deltas)
    hits = 0
    for index in range(min(len(ds), 8)):
        item = ds[index]
        found = none_keys(item)
        if found:
            hits += 1
            print(label, "index", index, "none", found, flush=True)
    print(label, "none_samples", hits, "checked", min(len(ds), 8), flush=True)


if __name__ == "__main__":
    prepare_config()
    scan("before")
    subprocess.check_call([sys.executable, "-m", "uv", "pip", "install", "torchao>0.16"])
    import torch
    import torchvision

    assert torch.version.hip
    assert "+rocm" in torchvision.__version__
    print("torchao_ok", torch.__version__, flush=True)
    scan("after")
