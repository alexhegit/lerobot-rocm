"""Delete Hugging Face model caches whose files live in a shared hub/blobs directory.

Usage: python3 prune-hf-models.py [--apply] models--org--name ...
Only blobs referenced by the listed models and by no other cached repo are removed.
Without --apply it only reports what would be freed.
"""

import os
import shutil
import sys
from pathlib import Path

HUB = Path("/root/.cache/huggingface/hub")


def blob_targets(repo: Path) -> set[Path]:
    targets = set()
    for path in repo.rglob("*"):
        if path.is_symlink():
            target = (path.parent / os.readlink(path)).resolve()
            if target.is_file():
                targets.add(target)
    return targets


def main() -> None:
    apply = "--apply" in sys.argv
    names = [a for a in sys.argv[1:] if a != "--apply"]
    drop = {HUB / n for n in names if (HUB / n).is_dir()}
    keep = {p for p in HUB.iterdir() if p.is_dir() and p.name != "blobs" and p not in drop}
    kept_blobs = set().union(*(blob_targets(p) for p in keep)) if keep else set()
    total = 0
    for repo in sorted(drop):
        own = {b for b in blob_targets(repo) if b not in kept_blobs}
        size = sum(b.stat().st_size for b in own)
        total += size
        print(f"{repo.name}: {len(own)} blobs, {size / 1e9:.1f} GB")
        if apply:
            for b in own:
                b.unlink()
            shutil.rmtree(repo)
    print(f"{'freed' if apply else 'would free'} {total / 1e9:.1f} GB")


if __name__ == "__main__":
    main()
