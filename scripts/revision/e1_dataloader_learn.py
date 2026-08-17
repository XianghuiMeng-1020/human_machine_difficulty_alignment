#!/usr/bin/env python3
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

train = pd.read_csv("race_prepared/race_mcq_train.csv")
rows = train.sample(256, random_state=0).reset_index(drop=True)
tok = AutoTokenizer.from_pretrained("roberta-base")


class DS(Dataset):
    def __len__(self):
        return len(rows)

    def __getitem__(self, i):
        r = rows.iloc[i]
        text = (
            f"Passage: {r.article}\nQuestion: {r.question}\n"
            f"A. {r.option_a}\nB. {r.option_b}\nC. {r.option_c}\nD. {r.option_d}"
        )
        return {"text": text, "label": int(r.label)}


def collate(batch):
    texts = [b["text"] for b in batch]
    labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)
    enc = tok(texts, padding=True, truncation=True, max_length=512, return_tensors="pt")
    enc["labels"] = labels
    return enc


model = AutoModelForSequenceClassification.from_pretrained("roberta-base", num_labels=4).cuda()
opt = torch.optim.AdamW(model.parameters(), lr=3e-5)
loader = DataLoader(DS(), batch_size=8, shuffle=True, collate_fn=collate)
losses = []
model.train()
step = 0
while step < 100:
    for batch in loader:
        labels = batch.pop("labels").cuda()
        batch = {k: v.cuda() for k, v in batch.items()}
        out = model(**batch, labels=labels)
        out.loss.backward()
        opt.step()
        opt.zero_grad()
        losses.append(float(out.loss.detach()))
        if step % 20 == 0:
            print("step", step, "loss", losses[-1])
        step += 1
        if step >= 100:
            break
print("delta", losses[0] - losses[-1])
print("DL_OK" if losses[-1] < losses[0] - 0.05 else "DL_FAIL")
