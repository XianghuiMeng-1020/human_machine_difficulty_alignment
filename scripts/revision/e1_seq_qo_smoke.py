#!/usr/bin/env python3
"""SeqClass smoke with Question+Options before Passage (keeps labels learnable under 512 trunc)."""
from __future__ import annotations

import argparse

import pandas as pd
import torch
from torch.optim import AdamW
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def build_text(row) -> str:
    # Question+options first so truncation cuts the passage, not the labels.
    return (
        f"Question: {row['question']}\n"
        f"A. {row['option_a']}\n"
        f"B. {row['option_b']}\n"
        f"C. {row['option_c']}\n"
        f"D. {row['option_d']}\n\n"
        f"Passage:\n{row['article']}"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv", default="revision/artifacts/smoke_custom_data/race_mcq_train.csv")
    ap.add_argument("--val_csv", default="revision/artifacts/smoke_custom_data/race_mcq_val.csv")
    ap.add_argument("--model", default="roberta-base")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--max_len", type=int, default=512)
    args = ap.parse_args()

    tr = pd.read_csv(args.train_csv)
    va = pd.read_csv(args.val_csv)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=4).cuda()
    opt = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    def batch_iter(df, bs, shuffle):
        idx = torch.randperm(len(df)).tolist() if shuffle else list(range(len(df)))
        for i in range(0, len(df), bs):
            rows = df.iloc[idx[i : i + bs]]
            texts = [build_text(r) for _, r in rows.iterrows()]
            labs = torch.tensor([int(r.label) for _, r in rows.iterrows()], device="cuda")
            enc = tok(
                texts,
                padding=True,
                truncation=True,
                max_length=args.max_len,
                return_tensors="pt",
            )
            yield {k: v.cuda() for k, v in enc.items()}, labs

    @torch.no_grad()
    def acc(df):
        model.eval()
        c = t = 0
        for enc, labs in batch_iter(df, 16, False):
            pred = model(**enc).logits.argmax(-1)
            c += (pred == labs).sum().item()
            t += len(labs)
        model.train()
        return c / max(t, 1)

    print(f"[qo_smoke] n_train={len(tr)} n_val={len(va)} model={args.model}")
    va_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for enc, labs in batch_iter(tr, args.batch_size, True):
            out = model(**enc, labels=labs)
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            opt.zero_grad()
            losses.append(float(out.loss.detach()))
        tr_acc = acc(tr)
        va_acc = acc(va)
        print(
            f"epoch {epoch} loss={sum(losses)/len(losses):.4f} "
            f"train={tr_acc:.3f} val={va_acc:.3f}",
            flush=True,
        )
    if va_acc < 0.40:
        raise SystemExit(3)
    print("QO_SMOKE_OK", va_acc)


if __name__ == "__main__":
    main()
