# PR draft: fix(flux3): pass the NATTEN backend to na2d/na3d as a top-level argument

Local branch: `fix/flux3-natten-backend` (worktree `~/Repo/lerobot-flux3-natten`), commit `4df276b8`, based on `main` `d490e472`. Not pushed.

## What this does

The FLUX3 Video VAE resolves a NATTEN backend per tensor layout (`blackwell-fna` > `hopper-fna` > `cutlass-fna` > `flex-fna`, overridable with `F3_NATTEN_BACKEND`) and passes it to `na2d` / `na3d` through `attention_kwargs`.

In NATTEN 0.21.x, `backend` and `run_persistent_kernel` are top-level `na2d` / `na3d` arguments. `attention_kwargs` only configures the FMHA that NATTEN uses for the self-attention shortcut and for additional KV. So the resolved backend never reached the neighborhood-attention path:

- `F3_NATTEN_BACKEND` has no effect on the main path.
- NATTEN falls back to its own `choose_backend()`. On Hopper/Blackwell that picks the same kernel, so the bug is invisible there. On devices without a CUTLASS kernel it picks uncompiled `flex-fna`.

This PR passes the kwargs at the top level and adds a test that records what `na2d` / `na3d` receive.

## Tests

- New `tests/policies/flux3/test_flux3_natten_backend.py` (8 cases: 4 backends x 2D/3D). Uses fake `na2d` / `na3d`, so it runs without NATTEN installed.
  - Before the fix: 8 failed (`KeyError: 'backend'`).
  - After the fix: 8 passed.
- `tests/policies/flux3` with the fix: 133 passed, 3 skipped, 4 failed. The 4 failures are `ImportError: Found an incompatible version of torchao ... 0.15.0` from PEFT in the test container, unrelated to this change.
- `ruff check` / `ruff format --check` pass on the changed files.

Tested on AMD MI300X (ROCm 7.14, torch 2.12). No NVIDIA run yet.

## Out of scope

This does not make FLUX3 trainable on ROCm. With the backend honoured, `flex-fna` still materialises a dense block mask unless it is compiled, and NATTEN disables compiled flex on non-CUDA devices. See the NATTEN issue draft in this folder.
