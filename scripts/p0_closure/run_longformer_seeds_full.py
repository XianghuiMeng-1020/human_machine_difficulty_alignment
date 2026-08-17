#!/usr/bin/env python3
"""Full-speed Longformer 3-seed runner for 4090 (24GB). Sequential seeds, effective batch 16."""
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
        # 4090 max throughput: bs=8+ckpt ~98% util; bs=16 fills more VRAM but slower wall-clock
        "--batch_size",
        "8",
        "--grad_accum",
        "2",
        "--eval_batch_size",
        "8",
        "--num_workers",
        "4",
        "--flat_out",
        "--amp",
        "--grad_checkpoint",
        "--cuda_memory_fraction",
        "0.92",
    ]


def summarize():
    rows, accs = [], []
    for seed in SEEDS:
        d = OUT / f"longformer_seed{seed}"
        meta_p = d / "run_meta.json"
        row = {"seed": seed, "complete": meta_p.is_file(), "dir": str(d)}
        if meta_p.is_file():
            meta = json.loads(meta_p.read_text(encoding="utf-8"))
            row["val_accuracy"] = meta.get("val_accuracy")
            row["best_val_accuracy"] = meta.get("best_val_accuracy")
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
        "hardware_note": "4090 max-throughput: bs=8 accum=2 amp grad_checkpoint workers=4 frac=0.92",
    }
    import pandas as pd

    pd.DataFrame(rows).to_csv(ROOT / "outputs/encoder/seed_summary.csv", index=False)
    (ROOT / "outputs/encoder/seed_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main():
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    for seed in SEEDS:
        d = OUT / f"longformer_seed{seed}"
        d.mkdir(parents=True, exist_ok=True)
        if (d / "run_meta.json").is_file():
            print(f"[skip] seed {seed} complete", flush=True)
            continue
        cmd = train_cmd(seed, d)
        (d / "launch_cmd.txt").write_text(" ".join(cmd), encoding="utf-8")
        log = d / "train.log"
        print("[run]", " ".join(cmd), flush=True)
        t0 = time.time()
        with log.open("w", encoding="utf-8") as fh:
            rc = subprocess.call(cmd, cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT)
        print(f"[done] seed={seed} rc={rc} hours={(time.time()-t0)/3600:.2f}", flush=True)
        if rc != 0:
            summarize()
            raise SystemExit(rc)
    print(json.dumps(summarize(), indent=2), flush=True)


if __name__ == "__main__":
    main()
