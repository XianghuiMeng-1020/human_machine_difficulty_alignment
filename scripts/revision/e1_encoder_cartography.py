#!/usr/bin/env python3
"""E1 helpers: competitive retrain launcher, train-vs-val cartography, length/truncation.

Full GPU retrain is launched via run_competitive_retrain.sh on a CUDA box.
This script:
  - writes the recommended TrainingArguments recipe
  - computes length/truncation diagnostics on official val
  - if train+val TD files exist, compares region labels (E1b)
  - optionally runs a tiny smoke train on MPS/CPU for pipeline check
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    REPO_ROOT,
    REVISION_ROOT,
    assign_datamap_region,
    cohen_kappa,
    compute_td_metrics,
    ensure_dir,
    save_table,
)


COMPETITIVE_RECIPE = {
    "models": ["allenai/longformer-base-4096", "google/bigbird-roberta-base"],
    "max_len": 2048,
    "epochs": 5,
    "lr": 2e-5,
    "batch_size": 2,
    "eval_batch_size": 4,
    "gradient_accumulation_steps": 16,
    "warmup_ratio": 0.06,
    "weight_decay": 0.01,
    "seed": 42,
    "target_val_accuracy": 0.70,
    "notes": (
        "Prior local run used lr=5e-4 / short effective optimization and reached ~33% "
        "(near chance). Competitive recipe follows common RACE finetuning ranges."
    ),
}


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--race_val_csv", default=str(REPO_ROOT / "race_prepared/race_mcq_val.csv"))
    ap.add_argument("--race_train_csv", default=str(REPO_ROOT / "race_prepared/race_mcq_train.csv"))
    ap.add_argument(
        "--val_td_csv",
        default=str(
            REPO_ROOT
            / "race_trainedmodels_5e-4_e5_256bs/models_longformer-base-4096/training_dynamics_val.csv"
        ),
    )
    ap.add_argument(
        "--train_td_csv",
        default=str(REVISION_ROOT / "artifacts/training_dynamics_train.csv"),
        help="Produced when retrain logs train-set dynamics (E1b setting A)",
    )
    ap.add_argument("--smoke_train", action="store_true", help="Tiny MPS/CPU smoke train")
    ap.add_argument("--smoke_n", type=int, default=64)
    ap.add_argument("--out_dir", default=str(REVISION_ROOT))
    return ap.parse_args()


def length_truncation_analysis(val: pd.DataFrame, tables: Path, figures: Path) -> None:
    from transformers import AutoTokenizer

    tok_name = REPO_ROOT / "race_trainedmodels_5e-4_e5_256bs/allenai_longformer-base-4096"
    if not tok_name.exists():
        tok_name = "allenai/longformer-base-4096"
    try:
        tokenizer = AutoTokenizer.from_pretrained(str(tok_name))
    except Exception as e:
        print(f"[WARN] tokenizer load failed ({e}); using char proxy only")
        tokenizer = None

    rows = []
    for _, row in val.iterrows():
        text = (
            "Read the following passage and answer the question.\n\n"
            f"Passage:\n{row['article']}\n\n"
            f"Question:\n{row['question']}\n\n"
            "Options:\n"
            f"A. {row['option_a']}\n"
            f"B. {row['option_b']}\n"
            f"C. {row['option_c']}\n"
            f"D. {row['option_d']}\n\n"
            "Please choose the best answer from A, B, C, or D."
        )
        if tokenizer is not None:
            n_tok = len(tokenizer(text, truncation=False, add_special_tokens=True)["input_ids"])
        else:
            n_tok = int(len(text) / 4)
        rows.append(
            {
                "question_id": row["question_id"],
                "designer_difficulty_str": row["designer_difficulty_str"],
                "n_tokens": n_tok,
                "truncated_at_2048": int(n_tok > 2048),
                "truncated_at_1600": int(n_tok > 1600),
            }
        )
    length_df = pd.DataFrame(rows)
    save_table(length_df, REVISION_ROOT / "artifacts/val_token_lengths.csv")

    summary = (
        length_df.groupby("designer_difficulty_str")
        .agg(
            n=("question_id", "count"),
            mean_tokens=("n_tokens", "mean"),
            median_tokens=("n_tokens", "median"),
            p90_tokens=("n_tokens", lambda s: float(np.quantile(s, 0.9))),
            trunc_rate_2048=("truncated_at_2048", "mean"),
            trunc_rate_1600=("truncated_at_1600", "mean"),
        )
        .reset_index()
    )
    save_table(summary, tables / "table_e1c_length_truncation_by_band.csv")

    try:
        import matplotlib.pyplot as plt

        ensure_dir(figures)
        plt.figure(figsize=(7, 4))
        for band, color in [("MIDDLE", "#4E62AB"), ("HIGH", "#D6404E")]:
            sub = length_df[length_df.designer_difficulty_str == band]
            plt.hist(sub["n_tokens"], bins=40, alpha=0.5, label=band, color=color)
        plt.axvline(2048, color="black", linestyle="--", label="2048")
        plt.xlabel("Tokens")
        plt.ylabel("Count")
        plt.legend()
        plt.tight_layout()
        plt.savefig(figures / "e1c_token_length_by_band.png", dpi=200)
        plt.close()
    except Exception as e:
        print(f"[WARN] plot failed: {e}")


def compare_train_val_cartography(val_td: Path, train_td: Path, tables: Path) -> None:
    if not val_td.is_file():
        print("[WARN] missing val TD; skip E1b")
        return
    val_dm, _ = assign_datamap_region(compute_td_metrics(pd.read_csv(val_td)))
    val_dm = val_dm.rename(columns={"datamap_region": "region_val"})
    if not train_td.is_file():
        print("[WARN] train TD not yet available; write placeholder for E1b")
        save_table(
            pd.DataFrame(
                [
                    {
                        "status": "pending_retrain",
                        "message": "Run competitive retrain with --log_train_dynamics to produce training_dynamics_train.csv",
                    }
                ]
            ),
            tables / "table_e1b_train_vs_val_cartography.csv",
        )
        return
    train_dm, _ = assign_datamap_region(compute_td_metrics(pd.read_csv(train_td)))
    train_dm = train_dm.rename(columns={"datamap_region": "region_train"})
    # Compare on overlapping question_ids (usually empty if train vs val disjoint);
    # also report marginal region distributions.
    merged = val_dm.merge(train_dm, on="question_id", how="inner")
    rows = [
        {
            "comparison": "marginal_val",
            **val_dm["region_val"].value_counts(normalize=True).to_dict(),
            "n": len(val_dm),
            "kappa_overlap": float("nan"),
        },
        {
            "comparison": "marginal_train",
            **train_dm["region_train"].value_counts(normalize=True).to_dict(),
            "n": len(train_dm),
            "kappa_overlap": float("nan"),
        },
    ]
    if len(merged):
        rows.append(
            {
                "comparison": "overlap_item_kappa",
                "n": len(merged),
                "kappa_overlap": cohen_kappa(merged["region_val"], merged["region_train"]),
            }
        )
    save_table(pd.DataFrame(rows), tables / "table_e1b_train_vs_val_cartography.csv")


def write_retrain_launcher(out_dir: Path) -> None:
    ensure_dir(out_dir / "protocols")
    recipe_path = out_dir / "protocols/e1_competitive_retrain_recipe.json"
    recipe_path.write_text(json.dumps(COMPETITIVE_RECIPE, indent=2), encoding="utf-8")

    sh = out_dir / "protocols/run_competitive_retrain.sh"
    sh.write_text(
        f"""#!/usr/bin/env bash
# E1a: competitive RACE encoder retrain (run on CUDA machine)
set -euo pipefail
cd "$(dirname "$0")/../.."
OUT=revision/artifacts/encoder_competitive
mkdir -p "$OUT"
python scripts/RACE_train_bert_models_trainer.py \\
  --data_dir race_prepared \\
  --out_dir "$OUT" \\
  --max_len {COMPETITIVE_RECIPE['max_len']} \\
  --epochs {COMPETITIVE_RECIPE['epochs']} \\
  --lr {COMPETITIVE_RECIPE['lr']} \\
  --batch_size {COMPETITIVE_RECIPE['batch_size']} \\
  --eval_batch_size {COMPETITIVE_RECIPE['eval_batch_size']} \\
  --seed {COMPETITIVE_RECIPE['seed']} \\
  --model_names allenai/longformer-base-4096 google/bigbird-roberta-base \\
  --log_train_dynamics
echo "Target val accuracy >= {COMPETITIVE_RECIPE['target_val_accuracy']}"
""",
        encoding="utf-8",
    )
    sh.chmod(0o755)
    print(f"[OK] wrote {sh}")


def smoke_train(args) -> None:
    """Minimal train to verify Trainer+TD pipeline on this machine."""
    import torch
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
    )

    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    from RACE_train_bert_models_trainer import (  # type: ignore
        RaceMCQDataset,
        TrainingDynamicsCallback,
        compute_metrics,
        set_seed,
    )

    set_seed(42)
    train_df = pd.read_csv(args.race_train_csv).sample(args.smoke_n, random_state=42)
    val_df = pd.read_csv(args.race_val_csv).sample(min(32, args.smoke_n), random_state=42)
    model_name = "distilbert-base-uncased"
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=4)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)
    train_ds = RaceMCQDataset(train_df, tok, max_len=256)
    val_ds = RaceMCQDataset(val_df, tok, max_len=256)
    out = ensure_dir(REVISION_ROOT / "artifacts/smoke_encoder")
    targs = TrainingArguments(
        output_dir=str(out / "hf"),
        num_train_epochs=1,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        learning_rate=2e-5,
        eval_strategy="epoch",
        save_strategy="no",
        report_to=[],
        seed=42,
    )
    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tok,
        compute_metrics=compute_metrics,
    )
    cb = TrainingDynamicsCallback(val_ds, val_ds.question_ids, out / "training_dynamics_val.csv")
    cb.set_trainer(trainer)
    trainer.add_callback(cb)
    trainer.train()
    pred = trainer.predict(val_ds)
    acc = float((pred.predictions.argmax(-1) == pred.label_ids).mean())
    (out / "smoke_summary.json").write_text(json.dumps({"val_acc": acc, "n_train": len(train_df)}))
    print(f"[OK] smoke train val_acc={acc:.3f}")


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    tables = ensure_dir(out_dir / "tables")
    figures = ensure_dir(out_dir / "figures")

    write_retrain_launcher(out_dir)
    (out_dir / "protocols/e1_competitive_retrain_recipe.json").write_text(
        json.dumps(COMPETITIVE_RECIPE, indent=2), encoding="utf-8"
    )

    val = pd.read_csv(args.race_val_csv)
    print("[INFO] E1c length/truncation analysis (may download tokenizer)...")
    length_truncation_analysis(val, tables, figures)

    compare_train_val_cartography(Path(args.val_td_csv), Path(args.train_td_csv), tables)

    # Legacy failure card (B3 diagnosis)
    legacy = {
        "checkpoint": str(args.val_td_csv),
        "approx_val_accuracy": 0.329,
        "status": "non_competitive_near_chance",
        "action": "Do not use as main manuscript encoder result; retrain with competitive recipe.",
    }
    (out_dir / "artifacts/e1_legacy_encoder_diagnosis.json").write_text(
        json.dumps(legacy, indent=2), encoding="utf-8"
    )

    if args.smoke_train:
        smoke_train(args)

    print("[OK] E1 scaffolding complete")


if __name__ == "__main__":
    main()
