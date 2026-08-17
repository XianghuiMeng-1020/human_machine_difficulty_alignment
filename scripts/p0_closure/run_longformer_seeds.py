#!/usr/bin/env python3
"""Sequentially train primary Longformer seeds 0/1/2 and write seed_summary.csv."""
from __future__ import annotations

import json
import statistics
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/encoder/seed_runs"
OUT.mkdir(parents=True, exist_ok=True)


def cmd_for(seed: int, out_dir: Path) -> list[str]:
    return [
        sys.executable,
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
        # Low VRAM: bs=1 + accum=16 keeps effective batch=16; hard-cap ~40% GPU for parallel jobs
        "--batch_size",
        "1",
        "--grad_accum",
        "16",
        "--eval_batch_size",
        "2",
        "--flat_out",
        "--amp",
        "--cuda_memory_fraction",
        "0.40",
    ]


def summarize():
    rows = []
    accs = []
    for seed in [0, 1, 2]:
        d = OUT / f"longformer_seed{seed}"
        meta_p = d / "run_meta.json"
        row = {"seed": seed, "dir": str(d.relative_to(ROOT)).replace("\\", "/"), "complete": meta_p.is_file()}
        if meta_p.is_file():
            meta = json.loads(meta_p.read_text(encoding="utf-8"))
            row["val_accuracy"] = meta.get("val_accuracy")
            row["best_val_accuracy"] = meta.get("best_val_accuracy")
            row["checkpoint_note"] = "best_by_val_accuracy in e1_train_mc"
            if meta.get("val_accuracy") is not None:
                accs.append(float(meta["val_accuracy"]))
        rows.append(row)
    summary = {
        "aggregation_rule": "report mean/SD/min/max across seeds 0,1,2; do not pick a single seed as sole result",
        "n_complete": len(accs),
        "mean_val_accuracy": statistics.mean(accs) if accs else None,
        "sd_val_accuracy": statistics.stdev(accs) if len(accs) > 1 else (0.0 if accs else None),
        "min_val_accuracy": min(accs) if accs else None,
        "max_val_accuracy": max(accs) if accs else None,
        "seeds": rows,
    }
    import pandas as pd

    pd.DataFrame(rows).to_csv(ROOT / "outputs/encoder/seed_summary.csv", index=False)
    (ROOT / "outputs/encoder/seed_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main():
    for seed in [0, 1, 2]:
        d = OUT / f"longformer_seed{seed}"
        d.mkdir(parents=True, exist_ok=True)
        if (d / "run_meta.json").is_file():
            print(f"[skip] seed {seed} complete", flush=True)
            continue
        cmd = cmd_for(seed, d)
        (d / "launch_cmd.txt").write_text(" ".join(cmd), encoding="utf-8")
        print("[run]", " ".join(cmd), flush=True)
        log = d / "train.log"
        with log.open("w", encoding="utf-8") as fh:
            rc = subprocess.call(cmd, cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT)
        if rc != 0:
            print(f"[FAIL] seed {seed} rc={rc}", flush=True)
            summarize()
            raise SystemExit(rc)
        print(f"[OK] seed {seed}", flush=True)
    print(json.dumps(summarize(), indent=2))


if __name__ == "__main__":
    main()
