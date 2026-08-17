#!/usr/bin/env python3
"""Resilient seed runner: survives parent shell death; one seed at a time; low VRAM.

Does not launch a new seed while another e1_train_mc is alive.
Writes PID files under each seed dir.
"""
from __future__ import annotations

import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/encoder/seed_runs"
OUT.mkdir(parents=True, exist_ok=True)
SEEDS = [0, 1, 2]


def train_cmd(seed: int, out_dir: Path) -> list[str]:
    return [
        sys.executable,
        "-u",
        "scripts/revision/e1_train_mc.py",
        "--model_name",
        "allenai/longformer-base-4096",
        "--seed",
        str(seed),
        "--out_dir",
        str(out_dir),
        "--max_len",
        "1024",
        "--article_words",
        "400",
        "--epochs",
        "4",
        "--lr",
        "2e-5",
        "--batch_size",
        "1",
        "--grad_accum",
        "16",
        "--eval_batch_size",
        "2",
        "--flat_out",
        "--amp",
        "--grad_checkpoint",
        "--cuda_memory_fraction",
        "0.38",
    ]


def live_train_pids() -> list[int]:
    pids = []
    try:
        out = subprocess.check_output(
            ["wmic", "process", "where", "name='python.exe'", "get", "ProcessId,CommandLine", "/FORMAT:CSV"],
            text=True,
            stderr=subprocess.DEVNULL,
            cwd=ROOT,
        )
    except Exception:
        # fallback: PowerShell
        ps = (
            "Get-CimInstance Win32_Process -Filter \"name='python.exe'\" | "
            "Where-Object { $_.CommandLine -match 'e1_train_mc' } | "
            "ForEach-Object { $_.ProcessId }"
        )
        try:
            out = subprocess.check_output(["powershell", "-NoProfile", "-Command", ps], text=True)
            for line in out.splitlines():
                line = line.strip()
                if line.isdigit():
                    pids.append(int(line))
        except Exception:
            pass
        return pids
    for line in out.splitlines():
        if "e1_train_mc" in line:
            parts = line.strip().split(",")
            if parts and parts[-1].isdigit():
                pids.append(int(parts[-1]))
    return pids


def summarize():
    rows, accs = [], []
    for seed in SEEDS:
        d = OUT / f"longformer_seed{seed}"
        meta_p = d / "run_meta.json"
        row = {"seed": seed, "complete": meta_p.is_file()}
        if meta_p.is_file():
            meta = json.loads(meta_p.read_text(encoding="utf-8"))
            row["val_accuracy"] = meta.get("val_accuracy")
            if meta.get("val_accuracy") is not None:
                accs.append(float(meta["val_accuracy"]))
        rows.append(row)
    summary = {
        "aggregation_rule": "mean/SD/min/max across seeds 0,1,2; do not pick a single seed",
        "n_complete": len(accs),
        "mean_val_accuracy": statistics.mean(accs) if accs else None,
        "sd_val_accuracy": statistics.stdev(accs) if len(accs) > 1 else (0.0 if accs else None),
        "min_val_accuracy": min(accs) if accs else None,
        "max_val_accuracy": max(accs) if accs else None,
        "seeds": rows,
        "live_train_pids": live_train_pids(),
    }
    import pandas as pd

    pd.DataFrame(rows).to_csv(ROOT / "outputs/encoder/seed_summary.csv", index=False)
    (ROOT / "outputs/encoder/seed_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def start_seed(seed: int) -> int:
    d = OUT / f"longformer_seed{seed}"
    d.mkdir(parents=True, exist_ok=True)
    cmd = train_cmd(seed, d)
    (d / "launch_cmd.txt").write_text(" ".join(cmd), encoding="utf-8")
    log = d / "train.log"
    # append marker
    with log.open("a", encoding="utf-8") as fh:
        fh.write(f"\n# watcher launch seed={seed} cwd={ROOT}\n")
    creationflags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0  # type: ignore[attr-defined]
    fh = open(log, "a", encoding="utf-8")
    proc = subprocess.Popen(
        cmd,
        cwd=str(ROOT),
        stdout=fh,
        stderr=subprocess.STDOUT,
        creationflags=creationflags,
    )
    (d / "train.pid").write_text(str(proc.pid), encoding="utf-8")
    print(f"[watch] started seed={seed} pid={proc.pid}", flush=True)
    return proc.pid


def main():
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    print("[watch] low-VRAM sequential Longformer seeds; poll every 60s", flush=True)
    while True:
        summary = summarize()
        print(json.dumps({"n_complete": summary["n_complete"], "live": summary["live_train_pids"], "seeds": summary["seeds"]}), flush=True)
        if summary["n_complete"] >= 3:
            print("[watch] all seeds complete", flush=True)
            print(json.dumps(summary, indent=2), flush=True)
            return
        live = summary["live_train_pids"]
        if live:
            time.sleep(60)
            continue
        # start next incomplete seed
        next_seed = next((s for s in SEEDS if not (OUT / f"longformer_seed{s}" / "run_meta.json").is_file()), None)
        if next_seed is None:
            print("[watch] nothing to start", flush=True)
            return
        start_seed(next_seed)
        time.sleep(90)  # allow model load before next poll


if __name__ == "__main__":
    main()
