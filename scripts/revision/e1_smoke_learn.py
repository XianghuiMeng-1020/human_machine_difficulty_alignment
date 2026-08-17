#!/usr/bin/env python3
"""Fast learning smoke-test: does the data+Trainer pipeline learn at all?

Runs ~300 update steps on a small subset; aborts early with clear signal.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))


def build_text(row):
    return (
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


class DS(Dataset):
    def __init__(self, df, tok, max_len, global_attn=False):
        self.df = df.reset_index(drop=True)
        self.tok = tok
        self.max_len = max_len
        self.global_attn = global_attn

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        enc = self.tok(
            build_text(row),
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
        )
        item = {
            "input_ids": torch.tensor(enc["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(enc["attention_mask"], dtype=torch.long),
            "labels": torch.tensor(int(row["label"]), dtype=torch.long),
        }
        if self.global_attn:
            gam = torch.zeros(self.max_len, dtype=torch.long)
            gam[0] = 1
            item["global_attention_mask"] = gam
        return item


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="roberta-base")
    ap.add_argument("--max_len", type=int, default=512)
    ap.add_argument("--n_train", type=int, default=4000)
    ap.add_argument("--n_val", type=int, default=800)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--bf16", action="store_true")
    args = ap.parse_args()

    data = ROOT / "race_prepared"
    train = pd.read_csv(data / "race_mcq_train.csv").sample(n=args.n_train, random_state=42)
    val = pd.read_csv(data / "race_mcq_val.csv").sample(n=args.n_val, random_state=42)
    print("label train", train.label.value_counts().to_dict())
    print("device", "cuda" if torch.cuda.is_available() else "cpu")

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=4, problem_type="single_label_classification"
    )
    use_global = "longformer" in args.model.lower()
    tr_ds = DS(train, tok, args.max_len, use_global)
    va_ds = DS(val, tok, args.max_len, use_global)

    targs = TrainingArguments(
        output_dir=str(ROOT / "revision/artifacts/smoke_learn"),
        max_steps=args.steps,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch,
        eval_strategy="steps",
        eval_steps=100,
        logging_steps=50,
        save_strategy="no",
        report_to=[],
        fp16=args.fp16,
        bf16=args.bf16,
        warmup_ratio=0.06,
        weight_decay=0.01,
        dataloader_num_workers=0,
        seed=42,
    )

    def metrics(p):
        preds = np.argmax(p.predictions, axis=-1)
        return {"accuracy": accuracy_score(p.label_ids, preds)}

    try:
        trainer = Trainer(
            model=model,
            args=targs,
            train_dataset=tr_ds,
            eval_dataset=va_ds,
            compute_metrics=metrics,
            processing_class=tok,
        )
    except TypeError:
        trainer = Trainer(
            model=model,
            args=targs,
            train_dataset=tr_ds,
            eval_dataset=va_ds,
            compute_metrics=metrics,
            tokenizer=tok,
        )

    out = trainer.train()
    ev = trainer.evaluate()
    print("SMOKE_RESULT", {"train_loss": out.training_loss, "eval": ev})
    acc = float(ev.get("eval_accuracy", 0))
    if acc < 0.40:
        print("SMOKE_FAIL accuracy still near chance:", acc)
        sys.exit(2)
    print("SMOKE_OK accuracy:", acc)
    sys.exit(0)


if __name__ == "__main__":
    main()
