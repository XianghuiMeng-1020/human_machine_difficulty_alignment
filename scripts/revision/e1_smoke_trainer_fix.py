#!/usr/bin/env python3
"""Isolate Trainer failure: dynamic pad + DataCollatorWithPadding."""
from __future__ import annotations

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
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

ROOT = Path(__file__).resolve().parents[2]


def build_text(row):
    return (
        f"Passage: {row['article']}\nQuestion: {row['question']}\n"
        f"A. {row['option_a']}\nB. {row['option_b']}\n"
        f"C. {row['option_c']}\nD. {row['option_d']}"
    )


class DS(Dataset):
    def __init__(self, df, tok, max_len):
        self.df = df.reset_index(drop=True)
        self.tok = tok
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        enc = self.tok(
            build_text(row),
            truncation=True,
            max_length=self.max_len,
        )
        enc["labels"] = int(row["label"])
        return enc


def main():
    train = pd.read_csv(ROOT / "race_prepared/race_mcq_train.csv").sample(4000, random_state=42)
    val = pd.read_csv(ROOT / "race_prepared/race_mcq_val.csv").sample(800, random_state=42)
    tok = AutoTokenizer.from_pretrained("roberta-base")
    model = AutoModelForSequenceClassification.from_pretrained(
        "roberta-base", num_labels=4, problem_type="single_label_classification"
    )
    tr = DS(train, tok, 512)
    va = DS(val, tok, 512)
    collator = DataCollatorWithPadding(tokenizer=tok)

    args = TrainingArguments(
        output_dir=str(ROOT / "revision/artifacts/smoke_trainer_fix"),
        max_steps=400,
        learning_rate=3e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        eval_strategy="steps",
        eval_steps=100,
        logging_steps=50,
        save_strategy="no",
        report_to=[],
        fp16=False,
        bf16=False,
        warmup_ratio=0.06,
        dataloader_num_workers=0,
        seed=42,
        remove_unused_columns=False,
    )

    def metrics(p):
        return {"accuracy": accuracy_score(p.label_ids, np.argmax(p.predictions, axis=-1))}

    try:
        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=tr,
            eval_dataset=va,
            data_collator=collator,
            compute_metrics=metrics,
            processing_class=tok,
        )
    except TypeError:
        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=tr,
            eval_dataset=va,
            data_collator=collator,
            compute_metrics=metrics,
            tokenizer=tok,
        )

    trainer.train()
    ev = trainer.evaluate()
    print("FIX_SMOKE", ev)
    acc = float(ev["eval_accuracy"])
    print("FIX_OK" if acc >= 0.40 else "FIX_FAIL", acc)
    sys.exit(0 if acc >= 0.40 else 2)


if __name__ == "__main__":
    main()
