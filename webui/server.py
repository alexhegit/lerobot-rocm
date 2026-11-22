#!/usr/bin/env python3
"""Serve the training UI and one fixed DiT smoke on a remote ROCm host.

Start and stop only control the container named rocm-webui-dit.
The command is fixed in this file. Request bodies are ignored.

Environment:
  LEROBOT_ROCM_HOST  ssh host alias of the GPU node (default: DOUni)
  LEROBOT_OUTPUTS    outputs directory on that node, mounted at /outputs (default: /data/lerobot/outputs)
"""

from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
HOST = os.environ.get("LEROBOT_ROCM_HOST", "DOUni")
OUTPUTS = os.environ.get("LEROBOT_OUTPUTS", "/data/lerobot/outputs")
CONTAINER = "rocm-webui-dit"
REMOTE_LOG = f"{OUTPUTS}/rocm-train/webui-dit.log"
REMOTE_SCRIPT = f"{OUTPUTS}/rocm-train/jobs/webui-dit.sh"

REMOTE_SCRIPT_BODY = r"""#!/bin/bash
set -u
LOG=@OUTPUTS@/rocm-train/webui-dit.log
exec >"$LOG" 2>&1
echo "webui-dit begin $(date -Is)"
rm -rf @OUTPUTS@/rocm-train/webui-dit
docker rm -f rocm-webui-dit >/dev/null 2>&1 || true
docker run --name rocm-webui-dit \
  --ipc=host --shm-size 16g \
  --device=/dev/kfd --device=/dev/dri \
  --group-add 44 --group-add 992 \
  -e HIP_VISIBLE_DEVICES=0 \
  -e MIOPEN_FIND_MODE=2 \
  -v @OUTPUTS@:/outputs \
  -v /root/.cache/huggingface:/root/.cache/huggingface \
  lerobot-rocm \
  lerobot-train \
  --policy.type=multi_task_dit \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --policy.horizon=32 \
  --policy.n_action_steps=24 \
  --policy.objective=diffusion \
  --policy.num_inference_steps=10 \
  --dataset.repo_id=lerobot/aloha_sim_transfer_cube_human \
  --dataset.episodes='[0]' \
  --batch_size=1 \
  --steps=2 \
  --env_eval_freq=0 \
  --save_freq=2 \
  --log_freq=1 \
  --num_workers=0 \
  --trackio.enable=false \
  --output_dir=/outputs/rocm-train/webui-dit
ec=$?
echo "EXIT:${ec}"
docker rm -f rocm-webui-dit >/dev/null 2>&1 || true
""".replace("@OUTPUTS@", OUTPUTS)


def ssh(script: str, timeout: int = 30) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["ssh", "-o", "BatchMode=yes", HOST, "bash", "-s"],
        input=script,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT), **kwargs)

    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, status: int, payload: dict) -> None:
        self._send(status, json.dumps(payload).encode(), "application/json")

    def do_GET(self) -> None:
        if self.path.split("?", 1)[0] != "/api/log":
            super().do_GET()
            return
        try:
            result = ssh(f"tail -n 200 {REMOTE_LOG} 2>/dev/null || true")
        except subprocess.TimeoutExpired:
            self._send(504, b"ssh timeout\n", "text/plain; charset=utf-8")
            return
        text = result.stdout or result.stderr
        self._send(200, text.encode(), "text/plain; charset=utf-8")

    def do_POST(self) -> None:
        path = self.path.split("?", 1)[0]
        length = int(self.headers.get("Content-Length", "0") or "0")
        if length:
            self.rfile.read(length)
        if path == "/api/start":
            self._start()
        elif path == "/api/stop":
            self._stop()
        else:
            self._json(404, {"ok": False, "error": "not found"})

    def _start(self) -> None:
        try:
            busy = ssh("if docker ps -q | grep -q .; then echo BUSY; fi")
        except subprocess.TimeoutExpired:
            self._json(504, {"ok": False, "error": "ssh timeout"})
            return
        if "BUSY" in busy.stdout:
            self._json(409, {"ok": False, "error": f"{HOST} already has a container"})
            return
        install = f"mkdir -p {OUTPUTS}/rocm-train/jobs\ncat > " + REMOTE_SCRIPT + " << 'ENDSCRIPT'\n" + REMOTE_SCRIPT_BODY + "ENDSCRIPT\nchmod +x " + REMOTE_SCRIPT + "\nnohup bash " + REMOTE_SCRIPT + " >/dev/null 2>&1 &\n"
        try:
            result = ssh(install, timeout=20)
        except subprocess.TimeoutExpired:
            self._json(504, {"ok": False, "error": "ssh timeout"})
            return
        if result.returncode != 0:
            self._json(500, {"ok": False, "error": (result.stderr or "ssh failed")[-400:]})
            return
        self._json(200, {"ok": True})

    def _stop(self) -> None:
        try:
            result = ssh(f"docker rm -f {CONTAINER} >/dev/null 2>&1 || true; echo stopped")
        except subprocess.TimeoutExpired:
            self._json(504, {"ok": False, "error": "ssh timeout"})
            return
        if result.returncode != 0:
            self._json(500, {"ok": False, "error": (result.stderr or "stop failed")[-400:]})
            return
        self._json(200, {"ok": True})


def main() -> None:
    host, port = "127.0.0.1", 8765
    server = ThreadingHTTPServer((host, port), Handler)
    print(f"http://{host}:{port}/", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
