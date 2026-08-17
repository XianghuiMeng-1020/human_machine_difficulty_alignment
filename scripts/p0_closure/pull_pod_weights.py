#!/usr/bin/env python3
"""Resilient SSH download of pod encoder weights."""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

KEY = Path.home() / ".ssh" / "id_ed25519"
HOST = "root@103.196.86.167"
PORT = "50410"
REMOTE = "/workspace/fode/outputs/encoder/seed_runs"
LOCAL = Path(__file__).resolve().parents[2] / "outputs/encoder/seed_runs"

SSH = [
    "ssh",
    "-o",
    "BatchMode=yes",
    "-o",
    "ConnectTimeout=30",
    "-o",
    "ServerAliveInterval=15",
    "-p",
    PORT,
    "-i",
    str(KEY),
    HOST,
]


def remote_size(rel: str) -> int:
    out = subprocess.check_output(SSH + [f"stat -c %s {REMOTE}/{rel}"], text=True)
    return int(out.strip())


def pull(rel: str, expected: int, attempts: int = 6) -> None:
    dest = LOCAL / rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file() and dest.stat().st_size == expected:
        print(f"SKIP {rel}", flush=True)
        return
    if dest.is_file():
        dest.unlink()
    for i in range(1, attempts + 1):
        print(f"GET {rel} try {i}/{attempts} expect={expected}", flush=True)
        with dest.open("wb") as fh:
            p = subprocess.run(SSH + [f"cat {REMOTE}/{rel}"], stdout=fh)
        got = dest.stat().st_size if dest.is_file() else 0
        print(f"  got={got} rc={p.returncode}", flush=True)
        if p.returncode == 0 and got == expected:
            print(f"OK {rel}", flush=True)
            return
        time.sleep(3)
    raise SystemExit(f"FAILED {rel} last_size={got}")


def main():
    files = []
    for seed in (0, 1, 2):
        files.append(f"longformer_seed{seed}/model_epoch4.pt")
        files.append(f"longformer_seed{seed}/hf_model/model.safetensors")
    if "--all-epochs" in sys.argv:
        for seed in (0, 1, 2):
            for ep in (1, 2, 3):
                files.append(f"longformer_seed{seed}/model_epoch{ep}.pt")
    for rel in files:
        expected = remote_size(rel)
        pull(rel, expected)
    print("ALL_OK", flush=True)


if __name__ == "__main__":
    main()
