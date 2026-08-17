#!/usr/bin/env python3
"""MultipleChoice smoke on 3000/600 — claim path for competitive RACE accuracy."""
from __future__ import annotations

import argparse

import pandas as pd
import torch
from torch.optim import AdamW
from transformers import AutoModelForMultipleChoice, AutoTokenizer


def mc_batch(tok, rows, max_len: int = 512):
    all_c, all_e, labs = [], [], []
    for _, r in rows.iterrows():
        ctx = f"{r.article} {r.question}"
        ends = [str(r.option_a), str(r.option_b), str(r.option_c), str(r.option_d)]
        all_c.extend([ctx] * 4)
        all_e.extend(ends)
        labs.append(int(r.label))
    enc = tok(
        all_c,
        all_e,
        padding=True,
        truncation="only_first",
        max_length=max_len,
        return_tensors="pt",
    )
    bs = len(rows)
    enc = {k: v.view(bs, 4, -1).cuda() for k, v in enc.items()}
    return enc, torch.tensor(labs, device="cuda")


@torch.no_grad()
def accuracy(model, tok, df, max_len: int = 512, bs: int = 8) -> float:
    model.eval()
    correct = total = 0
    for i in range(0, len(df), bs):
        enc, labs = mc_batch(tok, df.iloc[i : i + bs], max_len=max_len)
        pred = model(**enc).logits.argmax(-1)
        correct += (pred == labs).sum().item()
        total += len(labs)
    model.train()
    return correct / max(total, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv", default="revision/artifacts/smoke_custom_data/race_mcq_train.csv")
    ap.add_argument("--val_csv", default="revision/artifacts/smoke_custom_data/race_mcq_val.csv")
    ap.add_argument("--model", default="roberta-base")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--max_len", type=int, default=512)
    args = ap.parse_args()

    tr = pd.read_csv(args.train_csv)
    va = pd.read_csv(args.val_csv)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForMultipleChoice.from_pretrained(args.model).cuda()
    opt = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    print(f"[mc_smoke] n_train={len(tr)} n_val={len(va)} model={args.model}")
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        idx = torch.randperm(len(tr)).tolist()
        for i in range(0, len(tr), args.batch_size):
            enc, labs = mc_batch(tok, tr.iloc[idx[i : i + args.batch_size]], max_len=args.max_len)
            out = model(**enc, labels=labs)
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            opt.zero_grad()
            losses.append(float(out.loss.detach()))
        tr_acc = accuracy(model, tok, tr, max_len=args.max_len)
        va_acc = accuracy(model, tok, va, max_len=args.max_len)
        print(
            f"MC epoch {epoch} loss={sum(losses)/len(losses):.4f} "
            f"train={tr_acc:.3f} val={va_acc:.3f}",
            flush=True,
        )
    if va_acc < 0.40:
        raise SystemExit(3)
    print("MC_SMOKE_OK", va_acc)


if __name__ == "__main__":
    main()
