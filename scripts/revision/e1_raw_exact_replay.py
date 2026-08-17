#!/usr/bin/env python3
"""Exact copy of working raw loop; then one variant with max_length pad."""
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

train = pd.read_csv("race_prepared/race_mcq_train.csv")
tok = AutoTokenizer.from_pretrained("roberta-base")
rows = train.sample(256, random_state=0)


def run(pad_mode: str):
    model = AutoModelForSequenceClassification.from_pretrained("roberta-base", num_labels=4).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-5)
    model.train()
    losses = []
    for step in range(100):
        batch = rows.sample(8, replace=True, random_state=step)
        texts, labs = [], []
        for _, r in batch.iterrows():
            texts.append(
                f"Passage: {r.article}\nQuestion: {r.question}\n"
                f"A. {r.option_a}\nB. {r.option_b}\nC. {r.option_c}\nD. {r.option_d}"
            )
            labs.append(int(r.label))
        if pad_mode == "dynamic":
            enc = tok(texts, padding=True, truncation=True, max_length=512, return_tensors="pt")
        else:
            enc = tok(texts, padding="max_length", truncation=True, max_length=512, return_tensors="pt")
        enc = {k: v.cuda() for k, v in enc.items()}
        labels = torch.tensor(labs, device="cuda")
        out = model(**enc, labels=labels)
        out.loss.backward()
        opt.step()
        opt.zero_grad()
        losses.append(float(out.loss.detach()))
    print(pad_mode, "start", losses[0], "end", losses[-1], "delta", losses[0] - losses[-1])


run("dynamic")
run("max_length")
