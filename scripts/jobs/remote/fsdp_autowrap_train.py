"""lerobot-train with FSDP wrap units picked from the model when none are declared.

Equivalent to passing --accelerator.fsdp.wrap_modules='[...]' by hand: when the run is sharded,
no wrap classes were given and the policy declares no _fsdp_wrap_modules, the classes repeated
inside nn.ModuleList blocks become the FSDP units. The chosen names are printed as AUTO_WRAP so
they can be reused on the command line.

FSDP_CAST_DTYPE=float32|bfloat16 (or FSDP_CAST_FP32=1) additionally casts every floating-point
parameter and buffer to one dtype before sharding, for policies that load mixed bf16/fp32 weights
(FSDP2 needs one dtype per unit). float32 pairs with --accelerator.mixed_precision=bf16; bfloat16
is for policies whose code feeds bf16 tensors into their own layers.
"""

import os

import torch
from torch import nn

from lerobot.distributed import factory
from lerobot.scripts import lerobot_train

_CONTAINERS = (nn.ModuleList, nn.ModuleDict)
_original = lerobot_train.set_fsdp_wrap_modules


def _repeated_block_classes(model: nn.Module) -> list[str]:
    counts: dict[str, int] = {}
    for module in model.modules():
        if not isinstance(module, nn.ModuleList) or len(module) < 2:
            continue
        kinds = {type(child) for child in module}
        if len(kinds) != 1:
            continue
        kind = kinds.pop()
        # Blocks only: skip containers, parameter-free modules and leaf layers such as Linear,
        # which can hold weights tied to modules in other units (e.g. lm_head and embeddings).
        if (
            issubclass(kind, _CONTAINERS)
            or not any(True for _ in module[0].parameters())
            or not any(True for _ in module[0].children())
        ):
            continue
        counts[kind.__name__] = counts.get(kind.__name__, 0) + len(module)
    return sorted(counts)


def _cast(model: nn.Module, dtype: torch.dtype) -> None:
    casted = 0
    for tensor in [*model.parameters(), *model.buffers()]:
        if tensor.is_floating_point() and tensor.dtype != dtype:
            tensor.data = tensor.data.to(dtype)
            casted += 1
    dtypes = {p.dtype for p in model.parameters() if p.is_floating_point()}
    print(f"CAST_DTYPE {dtype} casted={casted} param_dtypes={sorted(map(str, dtypes))}", flush=True)


def _dtype_report(model: nn.Module, names: list[str]) -> None:
    """Print the dtypes inside each wrap class and of the parameters left in the root unit."""
    covered: set[int] = set()
    per_class: dict[str, set[str]] = {}
    for module in model.modules():
        if type(module).__name__ in names:
            dtypes = {str(p.dtype) for p in module.parameters()}
            per_class.setdefault(type(module).__name__, set()).update(dtypes)
            covered.update(id(p) for p in module.parameters())
    root: dict[tuple[str, str], int] = {}
    for fqn, param in model.named_parameters():
        if id(param) not in covered:
            key = (str(param.dtype), ".".join(fqn.split(".")[:4]))
            root[key] = root.get(key, 0) + param.numel()
    print(f"DTYPE_REPORT units {dict((k, sorted(v)) for k, v in per_class.items())}", flush=True)
    for (dtype, prefix), numel in sorted(root.items()):
        print(f"DTYPE_REPORT root {dtype} {prefix} {numel}", flush=True)


def _set_fsdp_wrap_modules(accelerator, policy) -> None:
    cast = os.environ.get("FSDP_CAST_DTYPE") or ("float32" if os.environ.get("FSDP_CAST_FP32") == "1" else "")
    if cast:
        _cast(factory.unwrap_peft_model(policy), getattr(torch, cast))
    plugin = getattr(accelerator.state, "fsdp_plugin", None)
    if plugin is not None and not plugin.min_num_params and not plugin.transformer_cls_names_to_wrap:
        base = factory.unwrap_peft_model(policy)
        if not getattr(type(base), "_fsdp_wrap_modules", None):
            names = _repeated_block_classes(base)
            print(f"AUTO_WRAP {names}", flush=True)
            if names:
                plugin.transformer_cls_names_to_wrap = names
    if plugin is not None and os.environ.get("FSDP_DTYPE_REPORT") == "1" and accelerator.is_main_process:
        _dtype_report(factory.unwrap_peft_model(policy), list(plugin.transformer_cls_names_to_wrap or []))
    _original(accelerator, policy)


def _param_owner_policy(model: nn.Module):
    """FSDP_WRAP_STRATEGY=params: every module that directly owns parameters is its own unit.

    Each unit then holds one module's parameters, so it has one dtype, and a submodule called on its
    own (not through its parent's forward) gathers its own shard. Modules owning tied parameters stay
    in their parent's unit. Slow (many small units); meant to check that sharding works at all.
    """
    uses: dict[int, int] = {}
    for _, param in model.named_parameters(remove_duplicate=False):
        uses[id(param)] = uses.get(id(param), 0) + 1
    chosen: set[int] = set()
    for name, module in model.named_modules():
        if not name or isinstance(module, _CONTAINERS):
            continue
        own = list(module.parameters(recurse=False))
        if own and all(uses[id(p)] == 1 for p in own):
            chosen.add(id(module))
    print(f"PARAM_OWNER_WRAP units={len(chosen)}", flush=True)
    return lambda module: id(getattr(module, "_checkpoint_wrapped_module", module)) in chosen


def _patch_accelerate_wrap_policy(policy_model: nn.Module) -> None:
    from accelerate.utils import fsdp_utils

    policy = _param_owner_policy(policy_model)
    fsdp_utils.fsdp2_prepare_auto_wrap_policy = lambda plugin, model: policy


_orig_set = _set_fsdp_wrap_modules


def _set_fsdp_wrap_modules_with_strategy(accelerator, policy) -> None:
    if os.environ.get("FSDP_WRAP_STRATEGY") == "params" and getattr(accelerator.state, "fsdp_plugin", None):
        _patch_accelerate_wrap_policy(factory.unwrap_peft_model(policy))
    _orig_set(accelerator, policy)


lerobot_train.set_fsdp_wrap_modules = _set_fsdp_wrap_modules_with_strategy

if __name__ == "__main__":
    lerobot_train.main()
