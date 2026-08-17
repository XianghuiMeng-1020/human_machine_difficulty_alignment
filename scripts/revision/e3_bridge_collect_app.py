#!/usr/bin/env python3
"""Local collector for Bridge-RACE student responses (E3a).

Usage:
  python scripts/revision/e3_bridge_collect_app.py
  open http://127.0.0.1:7861
"""
from __future__ import annotations

import csv
from pathlib import Path

import gradio as gr
import pandas as pd

from common import REVISION_ROOT

ITEMS = REVISION_ROOT / "bridge/bridge_race_items.csv"
OUT = REVISION_ROOT / "bridge/bridge_race_responses.csv"


def load_items():
    return pd.read_csv(ITEMS)


def next_item(participant_id: str, idx: int):
    df = load_items()
    if idx >= len(df):
        return idx, "DONE — thank you.", "", gr.update(choices=[], interactive=False)
    row = df.iloc[idx]
    qid = row["question_id"]
    text = (
        f"### Q {idx+1}/{len(df)} — `{qid}`\n\n"
        f"{row.get('article','')}\n\n"
        f"**{row.get('question','')}**\n\n"
        f"A. {row.get('option_a','')}\n\n"
        f"B. {row.get('option_b','')}\n\n"
        f"C. {row.get('option_c','')}\n\n"
        f"D. {row.get('option_d','')}"
    )
    return idx, text, participant_id, gr.update(choices=["A", "B", "C", "D"], value=None, interactive=True)


def submit(participant_id, idx, choice):
    df = load_items()
    if idx >= len(df) or not choice:
        return next_item(participant_id or "p1", idx)
    row = df.iloc[idx]
    gold = str(row.get("answer_letter", "")).strip().upper()
    is_correct = int(str(choice).upper() == gold)
    out_exists = OUT.is_file()
    with OUT.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["question_id", "participant_id", "chosen_letter", "is_correct"],
        )
        if not out_exists:
            w.writeheader()
        w.writerow(
            {
                "question_id": row["question_id"],
                "participant_id": participant_id or "p1",
                "chosen_letter": choice,
                "is_correct": is_correct,
            }
        )
    return next_item(participant_id or "p1", idx + 1)


def build():
    with gr.Blocks(title="Bridge-RACE Collection") as demo:
        gr.Markdown("# Bridge-RACE\nAnswer each question (A–D).")
        pid = gr.Textbox(label="Participant ID", value="p1")
        idx = gr.State(0)
        md = gr.Markdown()
        choice = gr.Radio(choices=["A", "B", "C", "D"], label="Your answer")
        btn = gr.Button("Submit", variant="primary")
        demo.load(next_item, inputs=[pid, idx], outputs=[idx, md, pid, choice])
        btn.click(submit, inputs=[pid, idx, choice], outputs=[idx, md, pid, choice])
    return demo


if __name__ == "__main__":
    build().launch(server_name="127.0.0.1", server_port=7861, share=False)
