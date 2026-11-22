#!/bin/bash
set -euo pipefail
python3 -c 'import torch,torchvision; print("before", torch.__version__, torch.version.hip, torchvision.__version__)'
set +e
timeout 600 uv pip install 'natten==0.21.7'
ec=$?
set -e
python3 - <<'PY'
import torch, torchvision
assert torch.version.hip, "ROCm torch was replaced"
assert "+rocm" in torchvision.__version__, torchvision.__version__
print("after", torch.__version__, torch.version.hip, torchvision.__version__)
PY
python3 - <<'PY'
import natten
print("natten", getattr(natten, "__version__", "?"))
print("file", natten.__file__)
PY
echo "INSTALL_EC=$ec"
exit "$ec"
