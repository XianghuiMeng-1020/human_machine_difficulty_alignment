#!/usr/bin/env python3
"""After E1 Longformer finishes, train BigBird then rebuild integrated analyses."""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PY = sys.executable
LF_PRED = ROOT / "revision/artifacts/encoder_competitive/allenai_longformer-base-4096/val_predictions.csv"
LF_TD = ROOT / "revision/artifacts/encoder_competitive/allenai_longformer-base-4096/training_dynamics_val.csv"
BB_PRED = ROOT / "revision/artifacts/encoder_competitive/google_bigbird-roberta-base/val_predictions.csv"


def wait_file(path: Path, poll=60):
    print(f"[watch] waiting for {path}")
    while not path.is_file():
        time.sleep(poll)
    print(f"[watch] found {path}")


def run(cmd: list[str]):
    print("[watch] RUN", " ".join(cmd))
    subprocess.check_call(cmd, cwd=str(ROOT))


def main():
    wait_file(LF_PRED)
    wait_file(LF_TD)
    if not BB_PRED.is_file():
        run(
            [
                PY,
                "scripts/RACE_train_bert_models_trainer.py",
                "--data_dir",
                "race_prepared",
                "--out_dir",
                "revision/artifacts/encoder_competitive",
                "--max_len",
                "2048",
                "--epochs",
                "5",
                "--lr",
                "2e-05",
                "--batch_size",
                "2",
                "--eval_batch_size",
                "4",
                "--seed",
                "42",
                "--model_names",
                "google/bigbird-roberta-base",
                "--log_train_dynamics",
                "--train_td_max_examples",
                "5000",
            ]
        )
    # rebuild with competitive encoder paths
    run(
        [
            PY,
            "scripts/revision/e0_build_integrated_table.py",
            "--bert_pred_csv",
            str(LF_PRED),
            "--bert_td_csv",
            str(LF_TD),
        ]
    )
    run([PY, "scripts/revision/e1_encoder_cartography.py"])
    run([PY, "scripts/revision/e4_agreement_sensitivity.py"])
    run([PY, "scripts/revision/e6_content_audit.py"])
    run([PY, "scripts/revision/e7_review_efficacy.py"])
    print("[watch] DONE competitive encoder + rebuild")


if __name__ == "__main__":
    main()
