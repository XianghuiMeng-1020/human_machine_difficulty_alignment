#!/usr/bin/env python3
"""Run all revision experiment pipelines that can execute without GPU/API/humans."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PY = sys.executable
SCRIPTS = Path(__file__).resolve().parent


def run(script: str, extra: list[str] | None = None):
    cmd = [PY, str(SCRIPTS / script)] + (extra or [])
    print("\n" + "=" * 72)
    print("RUNNING:", " ".join(cmd))
    print("=" * 72)
    subprocess.check_call(cmd, cwd=str(ROOT))


def main():
    # Order matches plan dependencies
    run("e2_llm_vote_aggregate.py")
    run("e0_build_integrated_table.py")
    run("e1_encoder_cartography.py")  # length analysis; skip smoke by default
    run("e5_eedi_reliability.py")
    run("e3_human_machine_bridge.py")
    run("e4_agreement_sensitivity.py")
    run("e6_content_audit.py")
    run("e7_review_efficacy.py")
    run("e8_open_ended_pilot.py")
    print("\n[OK] Full local revision pipeline finished. See revision/")


if __name__ == "__main__":
    main()
