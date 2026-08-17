#!/usr/bin/env python3
"""Custom PyTorch trainer for RACE encoders.

Transformers==5.9 Trainer stayed at chance (~27%) in multiple smokes while a
raw AdamW loop learns. This path bypasses Trainer for claim-preserving accuracy.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

ROOT = Path(__file__).resolve().parents[2]


def build_text(row) -> str:
    # Question+options BEFORE passage. With right-truncation, cutting the
    # article keeps the label-bearing tokens; passage-first collapses to chance
    # because CLS/pooled reps of long shared context bury the options.
    return (
        "Read the passage and choose the best answer from A, B, C, or D.\n\n"
        f"Question:\n{row['question']}\n\n"
        "Options:\n"
        f"A. {row['option_a']}\n"
        f"B. {row['option_b']}\n"
        f"C. {row['option_c']}\n"
        f"D. {row['option_d']}\n\n"
        f"Passage:\n{row['article']}"
    )


class RaceDS(Dataset):
    def __init__(self, df: pd.DataFrame, tok, max_len: int, use_global: bool):
        self.df = df.reset_index(drop=True)
        self.tok = tok
        self.max_len = max_len
        self.use_global = use_global

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        return {
            "text": build_text(row),
            "labels": int(row["label"]),
            "question_id": row["question_id"],
        }


def make_collate(tok, max_len: int, use_global: bool):
    def collate(batch):
        texts = [b["text"] for b in batch]
        qids = [b["question_id"] for b in batch]
        labels = torch.tensor([b["labels"] for b in batch], dtype=torch.long)
        enc = tok(
            texts,
            padding=True,
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
        )
        if use_global:
            gam = torch.zeros_like(enc["input_ids"])
            gam[:, 0] = 1
            enc["global_attention_mask"] = gam
        enc["labels"] = labels
        enc["question_id"] = qids
        return enc

    return collate


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_logits, all_labels, all_qids = [], [], []
    for batch in loader:
        qids = batch.pop("question_id")
        batch = {k: v.to(device) for k, v in batch.items()}
        logits = model(**{k: v for k, v in batch.items() if k != "labels"}).logits
        all_logits.append(logits.cpu())
        all_labels.append(batch["labels"].cpu())
        all_qids.extend(qids)
    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels)
    preds = logits.argmax(-1)
    acc = (preds == labels).float().mean().item()
    probs = torch.softmax(logits, dim=-1)
    prob_correct = probs[torch.arange(len(labels)), labels]
    return {
        "accuracy": acc,
        "question_id": all_qids,
        "gold_label": labels.numpy(),
        "pred_label": preds.numpy(),
        "prob_correct": prob_correct.numpy(),
        "probs": probs.numpy(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", default="allenai/longformer-base-4096")
    ap.add_argument("--data_dir", default=str(ROOT / "race_prepared"))
    ap.add_argument("--out_dir", default=str(ROOT / "revision/artifacts/encoder_competitive"))
    ap.add_argument("--max_len", type=int, default=2048)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--eval_batch_size", type=int, default=4)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fp16", action="store_true", default=False, help="Enable fp16 (broken on this stack; keep off)")
    ap.add_argument("--train_td_max", type=int, default=5000)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = Path(args.data_dir)
    train_df = pd.read_csv(data / "race_mcq_train.csv")
    val_df = pd.read_csv(data / "race_mcq_val.csv")

    model_name = args.model_name
    use_global = "longformer" in model_name.lower()
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=4, problem_type="single_label_classification"
    ).to(device)

    collate = make_collate(tok, args.max_len, use_global)
    train_ds = RaceDS(train_df, tok, args.max_len, use_global)
    val_ds = RaceDS(val_df, tok, args.max_len, use_global)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate, num_workers=0
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.eval_batch_size, shuffle=False, collate_fn=collate, num_workers=0
    )

    # train-TD subsample
    td_df = train_df.sample(n=min(args.train_td_max, len(train_df)), random_state=args.seed)
    td_loader = DataLoader(
        RaceDS(td_df, tok, args.max_len, use_global),
        batch_size=args.eval_batch_size,
        shuffle=False,
        collate_fn=collate,
        num_workers=0,
    )

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = (len(train_loader) // args.grad_accum) * args.epochs
    # Constant LR — linear schedule + tiny early LR correlated with majority-class collapse
    sched = None
    use_amp = bool(args.fp16 and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp) if use_amp else None

    out_root = Path(args.out_dir) / model_name.replace("/", "_")
    out_root.mkdir(parents=True, exist_ok=True)
    td_records = []

    print(
        f"[custom] model={model_name} device={device} n_train={len(train_df)} "
        f"n_val={len(val_df)} steps≈{total_steps} global_attn={use_global} amp={use_amp}"
    )

    global_step = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        nseen = 0
        opt.zero_grad(set_to_none=True)
        pbar = tqdm(train_loader, desc=f"epoch {epoch}")
        for i, batch in enumerate(pbar, 1):
            batch.pop("question_id")
            labels = batch.pop("labels").to(device)
            batch = {k: v.to(device) for k, v in batch.items()}
            if use_amp:
                with torch.amp.autocast("cuda", enabled=True):
                    out = model(**batch, labels=labels)
                    loss = out.loss / args.grad_accum
                scaler.scale(loss).backward()
            else:
                out = model(**batch, labels=labels)
                loss = out.loss / args.grad_accum
                loss.backward()
            running += float(out.loss.detach()) * labels.size(0)
            nseen += labels.size(0)
            if i % args.grad_accum == 0:
                if use_amp:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(opt)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()
                opt.zero_grad(set_to_none=True)
                if sched is not None:
                    sched.step()
                global_step += 1
                if global_step % 50 == 0:
                    pbar.set_postfix(loss=running / max(nseen, 1), step=global_step)

        # epoch-end eval + dynamics
        val_res = evaluate(model, val_loader, device)
        print(f"[custom] epoch={epoch} val_acc={val_res['accuracy']:.4f}")
        if val_res["accuracy"] < 0.35 and epoch >= 1:
            print("[WARN] still weak after epoch", epoch)

        # val TD
        for qid, pc, gold, pred in zip(
            val_res["question_id"],
            val_res["prob_correct"],
            val_res["gold_label"],
            val_res["pred_label"],
        ):
            td_records.append(
                {
                    "question_id": qid,
                    "epoch": float(epoch),
                    "prob_correct": float(pc),
                    "is_correct": int(pred == gold),
                }
            )
        # train TD subsample
        td_res = evaluate(model, td_loader, device)
        train_td_rows = []
        for qid, pc, gold, pred in zip(
            td_res["question_id"], td_res["prob_correct"], td_res["gold_label"], td_res["pred_label"]
        ):
            train_td_rows.append(
                {
                    "question_id": qid,
                    "epoch": float(epoch),
                    "prob_correct": float(pc),
                    "is_correct": int(pred == gold),
                }
            )
        pd.DataFrame(train_td_rows).to_csv(out_root / "training_dynamics_train_epoch.csv", index=False)
        # early stop success path: keep going to args.epochs but save best
        torch.save(model.state_dict(), out_root / f"model_epoch{epoch}.pt")
        with open(out_root / "epoch_metrics.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps({"epoch": epoch, "val_accuracy": val_res["accuracy"]}) + "\n")

    # final exports matching e0 expectations
    final = evaluate(model, val_loader, device)
    print(f"[custom] FINAL val accuracy = {final['accuracy']:.4f}")
    pd.DataFrame(
        {
            "question_id": final["question_id"],
            "gold_label": final["gold_label"],
            "pred_label": final["pred_label"],
            "prob_correct": final["prob_correct"],
        }
    ).to_csv(out_root / "val_predictions.csv", index=False)
    pd.DataFrame(td_records).to_csv(out_root / "training_dynamics_val.csv", index=False)
    # also write aggregated train TD if multi-epoch collected above only last; reload from records pattern
    model.save_pretrained(out_root / "hf_model")
    tok.save_pretrained(out_root / "hf_model")
    meta = {
        "model_name": model_name,
        "val_accuracy": final["accuracy"],
        "epochs": args.epochs,
        "max_len": args.max_len,
        "lr": args.lr,
        "trainer": "custom_pytorch",
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    (out_root / "run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("[OK]", meta)
    if final["accuracy"] < 0.55:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
