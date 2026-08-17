#!/usr/bin/env python3
"""Minimal local rating UI for E6 content audit (30+30 blind items).

Usage:
  python scripts/revision/e6_rating_app.py
  open http://127.0.0.1:7860
"""
from __future__ import annotations

import csv
from pathlib import Path

import gradio as gr
import pandas as pd

from common import REVISION_ROOT

ITEMS = REVISION_ROOT / "audit/e6_blind_items_for_raters.csv"
OUT = REVISION_ROOT / "audit/e6_ratings.csv"
RUBRIC = REVISION_ROOT / "audit/e6_coding_rubric.json"


def load_items():
    df = pd.read_csv(ITEMS)
    return df


def item_key(row) -> str:
    return str(row.get("question_id") or row.get("item_id"))


def next_item(rater_id: str, idx: int):
    df = load_items()
    if idx >= len(df):
        return idx, "DONE — all items rated for this session index.", "", "", "", "", gr.update(interactive=False)
    row = df.iloc[idx]
    text = (
        f"### Item {idx+1}/{len(df)} — `{item_key(row)}`\n\n"
        f"**Passage**\n\n{row.get('passage', row.get('article', ''))}\n\n"
        f"**Question**\n\n{row.get('question','')}\n\n"
        f"**Options**\n\n"
        f"A. {row.get('option_a','')}\n\n"
        f"B. {row.get('option_b','')}\n\n"
        f"C. {row.get('option_c','')}\n\n"
        f"D. {row.get('option_d','')}\n\n"
        f"**Keyed answer:** {row.get('answer_letter', row.get('gold_letter',''))}"
    )
    return idx, text, rater_id, False, False, False, gr.update(interactive=True)


def submit(rater_id, idx, ambiguous_key, flawed_distractors, evidence_not_locatable, notes):
    df = load_items()
    if idx >= len(df):
        return idx, "Already done.", rater_id, False, False, False, gr.update(interactive=False)
    row = df.iloc[idx]
    out_exists = OUT.is_file()
    with OUT.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "item_id",
                "rater_id",
                "ambiguous_key",
                "flawed_distractors",
                "evidence_not_locatable",
                "notes",
            ],
        )
        if not out_exists:
            w.writeheader()
        w.writerow(
            {
                "item_id": item_key(row),
                "rater_id": rater_id or "r1",
                "ambiguous_key": int(bool(ambiguous_key)),
                "flawed_distractors": int(bool(flawed_distractors)),
                "evidence_not_locatable": int(bool(evidence_not_locatable)),
                "notes": notes or "",
            }
        )
    return next_item(rater_id or "r1", idx + 1)


def build():
    with gr.Blocks(title="E6 Content Audit") as demo:
        gr.Markdown("# E6 Blind Content Audit\nMark item flaws. Arm (high/low disagreement) is hidden.")
        rater = gr.Textbox(label="Rater ID", value="r1")
        idx = gr.State(0)
        md = gr.Markdown()
        ambiguous = gr.Checkbox(label="Ambiguous key")
        flawed = gr.Checkbox(label="Flawed distractors")
        evidence = gr.Checkbox(label="Evidence not locatable in passage")
        notes = gr.Textbox(label="Notes")
        btn = gr.Button("Submit & Next", variant="primary")
        demo.load(next_item, inputs=[rater, idx], outputs=[idx, md, rater, ambiguous, flawed, evidence, btn])
        btn.click(
            submit,
            inputs=[rater, idx, ambiguous, flawed, evidence, notes],
            outputs=[idx, md, rater, ambiguous, flawed, evidence, btn],
        )
    return demo


if __name__ == "__main__":
    build().launch(server_name="127.0.0.1", server_port=7860, share=False)
