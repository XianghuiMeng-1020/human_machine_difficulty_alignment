#!/usr/bin/env python3
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

train = pd.read_csv("race_prepared/race_mcq_train.csv")
letters = "ABCD"
bad = 0
for _, r in train.head(2000).iterrows():
    if letters[int(r.label)] != str(r.answer_letter).strip().upper():
        bad += 1
print("label_mismatch_in_2000", bad)
print(train[["question_id", "answer_letter", "label"]].head(3).to_string())
print("nulls", train[["article", "question", "option_a", "label"]].isna().sum().to_dict())

tok = AutoTokenizer.from_pretrained("roberta-base")
model = AutoModelForSequenceClassification.from_pretrained("roberta-base", num_labels=4)
model.cuda().train()
opt = torch.optim.AdamW(model.parameters(), lr=3e-5)
rows = train.sample(256, random_state=0)
losses = []
for step in range(100):
    batch = rows.sample(8, replace=True, random_state=step)
    texts = []
    labs = []
    for _, r in batch.iterrows():
        texts.append(
            f"Passage: {r.article}\nQuestion: {r.question}\n"
            f"A. {r.option_a}\nB. {r.option_b}\nC. {r.option_c}\nD. {r.option_d}"
        )
        labs.append(int(r.label))
    enc = tok(texts, padding=True, truncation=True, max_length=512, return_tensors="pt")
    enc = {k: v.cuda() for k, v in enc.items()}
    labels = torch.tensor(labs, device="cuda")
    out = model(**enc, labels=labels)
    out.loss.backward()
    opt.step()
    opt.zero_grad()
    losses.append(float(out.loss))
    if step % 20 == 0:
        with torch.no_grad():
            pred = out.logits.argmax(-1)
            acc = (pred == labels).float().mean().item()
        print(f"step {step} loss={out.loss.item():.4f} batch_acc={acc:.3f}")
print(
    "loss_start",
    losses[0],
    "loss_end",
    losses[-1],
    "delta",
    losses[0] - losses[-1],
)
if losses[-1] < losses[0] - 0.05:
    print("RAW_PT_OK")
else:
    print("RAW_PT_FAIL")
