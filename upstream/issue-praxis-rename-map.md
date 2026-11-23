## What happens

The DROID fine-tune in `docs/source/praxis.mdx` ("Training" → "Data") fails at startup with the documented `rename_map`:

```bash
RENAME='{"observation.state.cartesian":"observation.state","action.cartesian":"action",
"observation.images.exterior_1_left":"observation.images.exterior","observation.images.wrist_left":"observation.images.wrist"}'
```

```
ValueError: Both 'observation.state' and 'observation.state.cartesian' are named 'observation.state' after rename_map. Map one of them to another name as well, e.g. add {'observation.state': 'observation.state.original'} to rename a key onto an existing 'observation.state'.
```

The docs say "a renamed key replaces an existing one", which was the behaviour before #5316 (merged the same day as Praxis, #5321) made collisions an error.

## Following the error message does not work either

Moving the joint-space columns out of the way, as the message suggests:

```bash
RENAME='{"observation.state":"observation.state.joint_unused","action":"action.joint_unused",
"observation.state.cartesian":"observation.state","action.cartesian":"action", ...cameras...}'
```

fails with

```
ValueError: Both 'observation.state.joint_unused' and 'observation.state' are named 'observation.state.joint_unused' after rename_map. ...
```

As the doc notes, `lerobot-train` applies the map to the raw batch and once more inside the processor pipeline. On the second pass, the `observation.state` produced by the first pass (formerly `observation.state.cartesian`) is renamed again by the `observation.state → ...` entry. (`observation.state.joint` cannot be used as the target: DROID already has that column.)

So there seems to be no `rename_map` that puts the Cartesian columns in place of the joint-space ones under the current collision check.

## Reproduce

The command in the doc's "Full fine-tune" / "LoRA" sections, with `--dataset.revision=refs/pr/11`, `--dataset.episodes` from `select_droid_episodes.py`, and `--policy.path=runwayml/praxis-1-droid-lerobot`. It fails before the first step, on any device.

## Possible fixes

- Apply `rename_map` once, or make the second pass a no-op for keys the first pass produced.
- Or allow an explicit replace, e.g. a key mapped onto a name whose current holder is also mapped away within the same map.
- Or a DROID revision whose Cartesian columns are already named `observation.state` / `action`.

## Environment

LeRobot `main` (`e5be0c16`), run from source on 1-4× AMD MI300X (ROCm 7.14). The failure is in key handling, not device code.
