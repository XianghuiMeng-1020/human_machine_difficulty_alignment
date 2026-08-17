#!/usr/bin/env python3
"""Mirror the successful raw PT smoke using Dataset/DataLoader path."""
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

train = pd.read_csv("race_prepared/race_mcq_train.csv")
rows = train.sample(256, random_state=0).reset_index(drop=True)
tok = AutoTokenizer.from_pretrained("roberta-base")
model = AutoModelForSequenceClassification.from_pretrained("roberta-base", num_labels=4).cuda()
opt = torch.optim.AdamW(model.parameters(), lr=3e-5)


class DS(Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        text = (
            f"Passage: {r.article}\nQuestion: {r.question}\n"
            f"A. {r.option_a}\nB. {r.option_b}\nC. {r.option_c}\nD. {r.option_d}"
        )
        enc = tok(text, truncation=True, max_length=512)
        enc["labels"] = int(r.label)
        return enc


def collate(batch):
    return tok.pad(batch, return_tensors="pt")


loader = DataLoader(DS(rows), batch_size=8, shuffle=True, collate_fn=collate)
losses = []
model.train()
step = 0
while step < 100:
    for batch in loader:
        batch = {k: v.cuda() for k, v in batch.items()}
        out = model(**batch)
        out.loss.backward()
        opt.step()
        opt.zero_grad()
        losses.append(float(out.loss.detach()))
        if step % 20 == 0:
            print("step", step, "loss", losses[-1])
        step += 1
        if step >= 100:
            break
print("delta", losses[0] - losses[-1], "end", losses[-1])
print("MIRROR_OK" if losses[-1] < losses[0] - 0.05 else "MIRROR_FAIL")
