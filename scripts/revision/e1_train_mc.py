#!/usr/bin/env python3
"""Competitive RACE encoder via MultipleChoice (claim-preserving accuracy path).

Root cause of prior ~27% runs: Passage-first SequenceClassification + long shared
context collapses choice-discriminative signal (pooled reps cosine ~1.0). This
script uses AutoModelForMultipleChoice with article head-words + question, which
BERT/RoBERTa can optimize. Longformer MC can use larger article_words / max_len.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForMultipleChoice, AutoTokenizer, get_linear_schedule_with_warmup

ROOT = Path(__file__).resolve().parents[2]


class RaceMC(Dataset):
    def __init__(self, df: pd.DataFrame, article_words: int):
        self.df = df.reset_index(drop=True)
        self.article_words = article_words

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        art = " ".join(str(r["article"]).split()[: self.article_words])
        ctx = f"{art} {r['question']}".strip()
        ends = [str(r["option_a"]), str(r["option_b"]), str(r["option_c"]), str(r["option_d"])]
        return {
            "context": ctx,
            "endings": ends,
            "labels": int(r["label"]),
            "question_id": r["question_id"],
        }


def make_collate(tok, max_len: int):
    def collate(batch):
        all_c, all_e, labels, qids = [], [], [], []
        for b in batch:
            all_c.extend([b["context"]] * 4)
            all_e.extend(b["endings"])
            labels.append(b["labels"])
            qids.append(b["question_id"])
        enc = tok(
            all_c,
            all_e,
            padding=True,
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
        )
        bs = len(batch)
        out = {k: v.view(bs, 4, -1) for k, v in enc.items()}
        out["labels"] = torch.tensor(labels, dtype=torch.long)
        out["question_id"] = qids
        return out

    return collate


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_logits, all_labels, all_qids = [], [], []
    for batch in loader:
        qids = batch.pop("question_id")
        labels = batch.pop("labels").to(device)
        batch = {k: v.to(device) for k, v in batch.items()}
        logits = model(**batch).logits
        all_logits.append(logits.cpu())
        all_labels.append(labels.cpu())
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
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", default="bert-base-uncased")
    ap.add_argument("--data_dir", default=str(ROOT / "race_prepared"))
    ap.add_argument("--out_dir", default=str(ROOT / "revision/artifacts/encoder_competitive"))
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--article_words", type=int, default=80)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--eval_batch_size", type=int, default=8)
    ap.add_argument("--grad_accum", type=int, default=4)
    ap.add_argument("--warmup_ratio", type=float, default=0.06)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train_td_max", type=int, default=5000)
    ap.add_argument("--max_train", type=int, default=0, help="If >0, subsample train for smoke")
    ap.add_argument("--gate_epoch1", type=float, default=0.35)
    ap.add_argument(
        "--flat_out",
        action="store_true",
        help="Write artifacts directly to --out_dir (do not nest model_name subdirectory)",
    )
    ap.add_argument("--amp", action="store_true", help="Use CUDA autocast + GradScaler")
    ap.add_argument(
        "--cuda_memory_fraction",
        type=float,
        default=0.0,
        help="If >0, torch.cuda.set_per_process_memory_fraction (e.g. 0.40 leaves ~60%% GPU for other jobs)",
    )
    ap.add_argument(
        "--grad_checkpoint",
        action="store_true",
        help="Enable gradient checkpointing to lower VRAM (safer with other GPU jobs)",
    )
    ap.add_argument("--num_workers", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda" and args.cuda_memory_fraction > 0:
        torch.cuda.set_per_process_memory_fraction(min(args.cuda_memory_fraction, 0.95))
        torch.cuda.empty_cache()
        print(
            f"[mc] cuda_memory_fraction={args.cuda_memory_fraction} "
            f"(hard cap; other projects can use remaining VRAM)",
            flush=True,
        )
    data = Path(args.data_dir)
    train_df = pd.read_csv(data / "race_mcq_train.csv")
    val_df = pd.read_csv(data / "race_mcq_val.csv")
    if args.max_train > 0:
        train_df = train_df.sample(n=min(args.max_train, len(train_df)), random_state=args.seed)

    tok = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForMultipleChoice.from_pretrained(args.model_name)
    if args.grad_checkpoint and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        if hasattr(model, "config"):
            model.config.use_cache = False
        print("[mc] gradient_checkpointing=ON", flush=True)
    model = model.to(device)
    collate = make_collate(tok, args.max_len)
    nw = max(0, int(args.num_workers))
    train_loader = DataLoader(
        RaceMC(train_df, args.article_words),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate,
        num_workers=nw,
        pin_memory=device.type == "cuda",
        persistent_workers=nw > 0,
    )
    val_loader = DataLoader(
        RaceMC(val_df, args.article_words),
        batch_size=args.eval_batch_size,
        shuffle=False,
        collate_fn=collate,
        num_workers=nw,
        pin_memory=device.type == "cuda",
        persistent_workers=nw > 0,
    )
    td_df = train_df.sample(n=min(args.train_td_max, len(train_df)), random_state=args.seed)
    td_loader = DataLoader(
        RaceMC(td_df, args.article_words),
        batch_size=args.eval_batch_size,
        shuffle=False,
        collate_fn=collate,
        num_workers=min(nw, 2) if nw else 0,
        pin_memory=device.type == "cuda",
        persistent_workers=(min(nw, 2) > 0),
    )

    opt = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = max(1, (len(train_loader) // args.grad_accum) * args.epochs)
    warmup = int(total_steps * args.warmup_ratio)
    sched = get_linear_schedule_with_warmup(opt, warmup, total_steps)

    out_root = Path(args.out_dir)
    if not args.flat_out:
        out_root = out_root / args.model_name.replace("/", "_")
    out_root.mkdir(parents=True, exist_ok=True)
    td_records = []
    best_acc = -1.0
    scaler = torch.cuda.amp.GradScaler(enabled=bool(args.amp and device.type == "cuda"))
    print(
        f"[mc] model={args.model_name} device={device} n_train={len(train_df)} "
        f"n_val={len(val_df)} article_words={args.article_words} max_len={args.max_len} "
        f"steps≈{total_steps} amp={args.amp} flat_out={args.flat_out} seed={args.seed}",
        flush=True,
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
            with torch.cuda.amp.autocast(enabled=bool(args.amp and device.type == "cuda")):
                out = model(**batch, labels=labels)
                loss = out.loss / args.grad_accum
            scaler.scale(loss).backward()
            running += float(out.loss.detach()) * labels.size(0)
            nseen += labels.size(0)
            if i % args.grad_accum == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                sched.step()
                opt.zero_grad(set_to_none=True)
                global_step += 1
                if global_step % 50 == 0:
                    pbar.set_postfix(loss=running / max(nseen, 1), step=global_step)

        val_res = evaluate(model, val_loader, device)
        print(f"[mc] epoch={epoch} val_acc={val_res['accuracy']:.4f}", flush=True)
        if epoch == 1 and val_res["accuracy"] < args.gate_epoch1:
            print(
                f"[GATE] epoch1 val {val_res['accuracy']:.4f} < {args.gate_epoch1} — abort",
                flush=True,
            )
            raise SystemExit(4)

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
        td_res = evaluate(model, td_loader, device)
        pd.DataFrame(
            [
                {
                    "question_id": qid,
                    "epoch": float(epoch),
                    "prob_correct": float(pc),
                    "is_correct": int(pred == gold),
                }
                for qid, pc, gold, pred in zip(
                    td_res["question_id"],
                    td_res["prob_correct"],
                    td_res["gold_label"],
                    td_res["pred_label"],
                )
            ]
        ).to_csv(out_root / "training_dynamics_train_epoch.csv", index=False)

        torch.save(model.state_dict(), out_root / f"model_epoch{epoch}.pt")
        with open(out_root / "epoch_metrics.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps({"epoch": epoch, "val_accuracy": val_res["accuracy"]}) + "\n")
        if val_res["accuracy"] > best_acc:
            best_acc = val_res["accuracy"]
            model.save_pretrained(out_root / "hf_model")
            tok.save_pretrained(out_root / "hf_model")
            pd.DataFrame(
                {
                    "question_id": val_res["question_id"],
                    "gold_label": val_res["gold_label"],
                    "pred_label": val_res["pred_label"],
                    "prob_correct": val_res["prob_correct"],
                }
            ).to_csv(out_root / "val_predictions.csv", index=False)

    final = evaluate(model, val_loader, device)
    pd.DataFrame(td_records).to_csv(out_root / "training_dynamics_val.csv", index=False)
    # ensure preds exist even if last != best
    if not (out_root / "val_predictions.csv").is_file():
        pd.DataFrame(
            {
                "question_id": final["question_id"],
                "gold_label": final["gold_label"],
                "pred_label": final["pred_label"],
                "prob_correct": final["prob_correct"],
            }
        ).to_csv(out_root / "val_predictions.csv", index=False)
    meta = {
        "model_name": args.model_name,
        "seed": args.seed,
        "val_accuracy": final["accuracy"],
        "best_val_accuracy": best_acc,
        "epochs": args.epochs,
        "max_len": args.max_len,
        "article_words": args.article_words,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "trainer": "custom_pytorch_multiple_choice",
        "construct": "held-out confidence/generalization dynamics (not original Dataset Cartography training dynamics)",
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    (out_root / "run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("[OK]", meta, flush=True)
    if best_acc < 0.55:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
