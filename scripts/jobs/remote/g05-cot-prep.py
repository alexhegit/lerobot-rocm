"""Copy the SO100 snapshot and add a constant atomic_task column."""
import json
import shutil
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

src = Path("/root/.cache/huggingface/lerobot/hub/datasets--lerobot--svla_so100_pickplace/snapshots/728583b5eaf9e739a7f119e2def466fa1d552402")
dst = Path("/outputs/rocm-train/g05-cot-data")
if dst.exists():
    shutil.rmtree(dst)
for path in src.rglob("*"):
    rel = path.relative_to(src)
    target = dst / rel
    if path.is_symlink() or path.is_file():
        target.parent.mkdir(parents=True, exist_ok=True)
        if path.name == "file-000.parquet" and "data" in rel.parts:
            table = pq.read_table(path.resolve())
            column = pa.array(["pick up the cube"] * table.num_rows)
            if "atomic_task" in table.column_names:
                table = table.drop(["atomic_task"])
            table = table.append_column("atomic_task", column)
            pq.write_table(table, target)
            print("rows", table.num_rows)
        else:
            target.symlink_to(path.resolve())
    elif path.is_dir():
        target.mkdir(parents=True, exist_ok=True)
info_path = dst / "meta" / "info.json"
info = json.loads(info_path.read_text())
info["features"]["atomic_task"] = {"dtype": "string", "shape": [1], "names": None}
info_path.write_text(json.dumps(info))
print("g05 data ready", dst)
