#!/usr/bin/env python3
"""Debug why MultipleChoice / full-epoch RACE fails to leave chance."""
from __future__ import annotations

import pandas as pd
import torch
from torch.optim import AdamW
from transformers import AutoModelForMultipleChoice, AutoTokenizer

tr = pd.read_csv("revision/artifacts/smoke_custom_data/race_mcq_train.csv")
tok = AutoTokenizer.from_pretrained("roberta-base")


def mc_batch(rows, max_len=512):
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


def acc_model(m, df):
    m.eval()
    c = t = 0
    with torch.no_grad():
        for i in range(0, len(df), 8):
            enc, labs = mc_batch(df.iloc[i : i + 8])
            pred = m(**enc).logits.argmax(-1)
            c += (pred == labs).sum().item()
            t += len(labs)
    m.train()
    return c / t


# 1) MC with-replacement on 256
model = AutoModelForMultipleChoice.from_pretrained("roberta-base").cuda().train()
opt = AdamW(model.parameters(), lr=3e-5)
rows = tr.sample(256, random_state=0)
losses = []
for step in range(150):
    batch = rows.sample(4, replace=True, random_state=step)
    enc, labs = mc_batch(batch)
    out = model(**enc, labels=labs)
    out.loss.backward()
    gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    opt.zero_grad()
    losses.append(float(out.loss.detach()))
    if step % 30 == 0:
        print(f"replace step {step} loss={losses[-1]:.4f} grad_norm={float(gn):.4f}", flush=True)

print(
    f"MC replace loss {losses[0]:.4f}->{losses[-1]:.4f} train256={acc_model(model, rows):.3f}",
    flush=True,
)

# 2) Fresh model: epochs on 512 unique
model2 = AutoModelForMultipleChoice.from_pretrained("roberta-base").cuda().train()
opt2 = AdamW(model2.parameters(), lr=5e-5)
subset = tr.sample(512, random_state=1).reset_index(drop=True)
for epoch in range(1, 5):
    losses2 = []
    norms = []
    idx = torch.randperm(len(subset)).tolist()
    for i in range(0, len(subset), 4):
        enc, labs = mc_batch(subset.iloc[idx[i : i + 4]])
        out = model2(**enc, labels=labs)
        out.loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model2.parameters(), 1.0)
        opt2.step()
        opt2.zero_grad()
        losses2.append(float(out.loss.detach()))
        norms.append(float(gn))
    print(
        f"unique512 epoch {epoch} loss={sum(losses2)/len(losses2):.4f} "
        f"grad_mean={sum(norms)/len(norms):.4f} train={acc_model(model2, subset):.3f}",
        flush=True,
    )

# 3) Sanity: random labels should still drive loss down (memorization)
model3 = AutoModelForMultipleChoice.from_pretrained("roberta-base").cuda().train()
opt3 = AdamW(model3.parameters(), lr=5e-5)
tiny = tr.sample(64, random_state=2).reset_index(drop=True)
fake = tiny.copy()
fake["label"] = list(range(64))
fake["label"] = fake["label"] % 4
for epoch in range(1, 8):
    losses3 = []
    idx = torch.randperm(len(fake)).tolist()
    for i in range(0, len(fake), 4):
        enc, labs = mc_batch(fake.iloc[idx[i : i + 4]])
        out = model3(**enc, labels=labs)
        out.loss.backward()
        opt3.step()
        opt3.zero_grad()
        losses3.append(float(out.loss.detach()))
    print(
        f"fake64 epoch {epoch} loss={sum(losses3)/len(losses3):.4f} "
        f"train={acc_model(model3, fake):.3f}",
        flush=True,
    )
