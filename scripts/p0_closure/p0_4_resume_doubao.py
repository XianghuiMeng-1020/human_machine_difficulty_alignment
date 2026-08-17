#!/usr/bin/env python3
"""Resume only Doubao frozen backend (DeepSeek/GPT already complete). CPU/API only."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("p04", ROOT / "scripts/p0_closure/p0_4_llm_frozen_rerun.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

mod.load_dotenv()
protocol = yaml.safe_load((ROOT / "configs/llm_protocol.yaml").read_text(encoding="utf-8"))
items = pd.read_csv(ROOT / "revision/artifacts/race_val_integrated.csv")
backend = next(b for b in protocol["backends"] if b["provider"] == "volcengine_ark")
# modest workers to avoid rate-limit / machine load while other projects run
meta, err = mod.run_backend(backend, items, workers=12, retry_unparsed=True)
print("doubao", meta is not None, err)
mod.aggregate_and_metrics(items)
print("[OK] doubao resume + metrics")
