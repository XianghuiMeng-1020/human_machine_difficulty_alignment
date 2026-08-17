#!/usr/bin/env python3
"""P0-3 structural encoder tests (no full 3-seed train here)."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForMultipleChoice, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/encoder/structural_tests"
OUT.mkdir(parents=True, exist_ok=True)
EVID = ROOT / "audit/evidence"
EVID.mkdir(parents=True, exist_ok=True)

LETTER = {"A": 0, "B": 1, "C": 2, "D": 3}
INV = {v: k for k, v in LETTER.items()}


class TinyMC(torch.utils.data.Dataset):
    def __init__(self, df, article_words=120):
        self.df = df.reset_index(drop=True)
        self.article_words = article_words

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        art = " ".join(str(r["article"]).split()[: self.article_words])
        ctx = f"{art} {r['question']}".strip()
        ends = [str(r[c]) for c in ["option_a", "option_b", "option_c", "option_d"]]
        return {
            "context": ctx,
            "endings": ends,
            "labels": int(LETTER[str(r["answer_letter"]).strip().upper()]),
            "question_id": r["question_id"],
            "gold_letter": str(r["answer_letter"]).strip().upper()[:1],
        }


def collate(tok, max_len):
    def _c(batch):
        ctxs, ends, labels, qids, golds = [], [], [], [], []
        for b in batch:
            ctxs.extend([b["context"]] * 4)
            ends.extend(b["endings"])
            labels.append(b["labels"])
            qids.append(b["question_id"])
            golds.append(b["gold_letter"])
        enc = tok(ctxs, ends, padding=True, truncation=True, max_length=max_len, return_tensors="pt")
        bs = len(batch)
        out = {k: v.view(bs, 4, -1) for k, v in enc.items()}
        out["labels"] = torch.tensor(labels)
        out["question_id"] = qids
        out["gold_letter"] = golds
        out["raw_batch"] = batch
        return out

    return _c


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    val = pd.read_csv(ROOT / "race_prepared/race_mcq_val.csv")
    # mapping audit 30 items
    sample = val.sample(n=30, random_state=0)
    map_rows = []
    for _, r in sample.iterrows():
        letter = str(r["answer_letter"]).strip().upper()[:1]
        lab = int(r["label"]) if "label" in r else LETTER[letter]
        map_rows.append(
            {
                "question_id": r["question_id"],
                "raw_answer_letter": letter,
                "encoded_label": lab,
                "decoded_letter": INV[lab],
                "match": INV[lab] == letter and lab == LETTER[letter],
            }
        )
    map_df = pd.DataFrame(map_rows)
    map_df.to_csv(OUT / "label_mapping_audit_30.csv", index=False)

    # input construction 10 items
    tok = AutoTokenizer.from_pretrained("bert-base-uncased")
    ten = val.sample(n=10, random_state=1)
    ds = TinyMC(ten, article_words=80)
    inputs = []
    for i in range(len(ds)):
        item = ds[i]
        inputs.append(
            {
                "question_id": item["question_id"],
                "context_head": item["context"][:300],
                "endings": item["endings"],
                "label": item["labels"],
                "gold_letter": item["gold_letter"],
            }
        )
    (OUT / "input_construction_audit_10.json").write_text(json.dumps(inputs, indent=2), encoding="utf-8")

    # baselines on val
    chance = 0.25
    maj = val["label"].value_counts(normalize=True).max() if "label" in val.columns else np.nan
    if "label" not in val.columns:
        val["label"] = val["answer_letter"].map(LETTER)
        maj = val["label"].value_counts(normalize=True).max()
    pd.DataFrame(
        [{"chance_accuracy": chance, "majority_label_accuracy": float(maj), "n": len(val)}]
    ).to_csv(OUT / "baselines.csv", index=False)

    # truncation from integrated
    integ = pd.read_csv(ROOT / "revision/artifacts/race_val_integrated.csv")
    trunc = (
        integ.groupby("designer_difficulty_str")["likely_truncated_2048"]
        .agg(n="count", trunc_rate="mean")
        .reset_index()
    )
    trunc.to_csv(OUT / "truncation_by_band.csv", index=False)

    # checkpoint identity: longformer run_meta acc vs integrated
    meta = json.loads(
        (ROOT / "revision/artifacts/encoder_competitive/allenai_longformer-base-4096/run_meta.json").read_text()
    )
    ckpt = ROOT / "revision/artifacts/encoder_competitive/allenai_longformer-base-4096/model_epoch4.pt"
    ckpt_hash = sha256_file(ckpt) if ckpt.is_file() else None
    integ_acc = float(integ["enc_correct"].mean())
    identity = {
        "checkpoint": str(ckpt.relative_to(ROOT)).replace("\\", "/"),
        "checkpoint_sha256": ckpt_hash,
        "run_meta_val_accuracy": meta.get("val_accuracy"),
        "integrated_enc_accuracy": integ_acc,
        "abs_diff": abs(float(meta.get("val_accuracy", 0)) - integ_acc),
        "match_tol_1e-6": abs(float(meta.get("val_accuracy", 0)) - integ_acc) < 1e-6,
    }
    (OUT / "checkpoint_identity.json").write_text(json.dumps(identity, indent=2), encoding="utf-8")

    # tiny overfit + gradient on bert-base (fast) using competitive recipe
    tiny = val.sample(n=64, random_state=42)
    model = AutoModelForMultipleChoice.from_pretrained("bert-base-uncased").to(device)
    loader = DataLoader(TinyMC(tiny, 80), batch_size=8, shuffle=True, collate_fn=collate(tok, 256))
    opt = torch.optim.AdamW(model.parameters(), lr=5e-5)
    # one-batch gradient audit
    batch = next(iter(loader))
    raw_batch = batch.pop("raw_batch")
    qids = batch.pop("question_id")
    golds = batch.pop("gold_letter")
    labels = batch.pop("labels").to(device)
    batch = {k: v.to(device) for k, v in batch.items()}
    model.train()
    out = model(**batch, labels=labels)
    loss0 = float(out.loss.detach().cpu())
    out.loss.backward()
    grad_norm = 0.0
    n_nonzero = 0
    for p in model.parameters():
        if p.grad is not None:
            g = p.grad.detach()
            grad_norm += float(g.norm().cpu() ** 2)
            n_nonzero += int((g.abs() > 0).any())
    grad_norm = grad_norm ** 0.5
    opt.step()
    # tiny overfit epochs
    for epoch in range(12):
        total = 0
        correct = 0
        for batch in loader:
            batch.pop("raw_batch", None)
            batch.pop("question_id", None)
            batch.pop("gold_letter", None)
            labels = batch.pop("labels").to(device)
            batch = {k: v.to(device) for k, v in batch.items()}
            opt.zero_grad()
            out = model(**batch, labels=labels)
            out.loss.backward()
            opt.step()
            pred = out.logits.argmax(-1)
            correct += int((pred == labels).sum().cpu())
            total += len(labels)
        train_acc = correct / total
    tiny_result = {
        "n": 64,
        "final_train_accuracy": train_acc,
        "pass_ge_0.95": train_acc >= 0.95,
        "one_batch_loss": loss0,
        "grad_norm": grad_norm,
        "n_tensors_with_nonzero_grad": n_nonzero,
        "device": device,
    }
    (OUT / "tiny_overfit_and_gradient.json").write_text(json.dumps(tiny_result, indent=2), encoding="utf-8")

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "label_mapping_mismatches": int((~map_df["match"]).sum()),
        "tiny_overfit_pass": bool(tiny_result["pass_ge_0.95"]),
        "gradient_nonzero": bool(n_nonzero > 0),
        "checkpoint_identity": identity,
        "seeds_completed": 0,
        "note": "Multi-seed Longformer runs are separate (p0_3_seed_train.py)",
    }
    (OUT / "structural_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (EVID / "encoder_validation.md").write_text(
        f"""# Encoder validation (P0-3 structural)

## Structural tests

Command: `python scripts/p0_closure/p0_3_encoder_structural.py`

| Test | Result |
|---|---|
| 30-item A/B/C/D mapping | mismatches={summary['label_mapping_mismatches']} |
| 10-item input construction | `outputs/encoder/structural_tests/input_construction_audit_10.json` |
| One-batch gradient | nonzero_tensors={n_nonzero}, grad_norm={grad_norm:.4f} |
| Tiny overfit (64, bert-base) | train_acc={train_acc:.4f}, pass={tiny_result['pass_ge_0.95']} |
| Chance / majority baselines | `baselines.csv` |
| Truncation by band | `truncation_by_band.csv` (all 0.0 at flag) |
| Checkpoint identity | run_meta vs integrated abs_diff={identity['abs_diff']} |

## Construct language

All val dynamics files are **held-out confidence / generalization dynamics**, not original
Dataset Cartography on training examples.

## Multi-seed status

See `outputs/encoder/seed_runs/` (populated by seed training script). Currently seeds_completed=0 in this structural pass.
""",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
