#!/usr/bin/env python3
"""Raw PT loop on same 3000/600 smoke split; report train+val acc."""
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

tr = pd.read_csv("revision/artifacts/smoke_custom_data/race_mcq_train.csv")
va = pd.read_csv("revision/artifacts/smoke_custom_data/race_mcq_val.csv")
tok = AutoTokenizer.from_pretrained("roberta-base")
model = AutoModelForSequenceClassification.from_pretrained("roberta-base", num_labels=4).cuda()
opt = torch.optim.AdamW(model.parameters(), lr=3e-5)


def batch_iter(df, bs=8, shuffle=True):
    idx = torch.randperm(len(df)).tolist() if shuffle else list(range(len(df)))
    for i in range(0, len(df), bs):
        sl = idx[i : i + bs]
        rows = df.iloc[sl]
        texts = [
            f"Passage: {r.article}\nQuestion: {r.question}\n"
            f"A. {r.option_a}\nB. {r.option_b}\nC. {r.option_c}\nD. {r.option_d}"
            for _, r in rows.iterrows()
        ]
        labs = [int(r.label) for _, r in rows.iterrows()]
        enc = tok(texts, padding=True, truncation=True, max_length=512, return_tensors="pt")
        yield {k: v.cuda() for k, v in enc.items()}, torch.tensor(labs, device="cuda")


@torch.no_grad()
def acc(df):
    model.eval()
    correct = 0
    total = 0
    for enc, labels in batch_iter(df, bs=16, shuffle=False):
        pred = model(**enc).logits.argmax(-1)
        correct += (pred == labels).sum().item()
        total += len(labels)
    model.train()
    return correct / total


for epoch in range(1, 3):
    model.train()
    losses = []
    for enc, labels in batch_iter(tr, bs=8, shuffle=True):
        out = model(**enc, labels=labels)
        out.loss.backward()
        opt.step()
        opt.zero_grad()
        losses.append(float(out.loss.detach()))
    print(
        f"epoch {epoch} loss={sum(losses)/len(losses):.4f} "
        f"train_acc={acc(tr):.3f} val_acc={acc(va):.3f}"
    )
