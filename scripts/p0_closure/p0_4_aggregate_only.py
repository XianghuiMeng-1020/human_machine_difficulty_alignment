#!/usr/bin/env python3
"""Recompute LLM metrics from frozen response jsonl without API calls."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("p04", ROOT / "scripts/p0_closure/p0_4_llm_frozen_rerun.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
items = pd.read_csv(ROOT / "revision/artifacts/race_val_integrated.csv")
mod.aggregate_and_metrics(items)
print("[OK] metrics recomputed")
