#!/usr/bin/env python3
"""Continuous experiment driver: monitor E1, chain BigBird, rebuild, report.

Emits AGENT_LOOP_WAKE_fode_rev lines for Cursor notify_on_output.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PY = sys.executable
LOG = ROOT / "revision/logs/monitor_drive.log"
STATUS = ROOT / "revision/STATUS.md"
LF_PRED = (
    ROOT
    / "revision/artifacts/encoder_competitive/allenai_longformer-base-4096/val_predictions.csv"
)
LF_TD = (
    ROOT
    / "revision/artifacts/encoder_competitive/allenai_longformer-base-4096/training_dynamics_val.csv"
)
BB_PRED = (
    ROOT
    / "revision/artifacts/encoder_competitive/google_bigbird-roberta-base/val_predictions.csv"
)
E1_LOG = ROOT / "revision/logs/e1_longformer.log"
BB_LOG = ROOT / "revision/logs/e1_bigbird.log"


def log(msg: str) -> None:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    line = f"[{datetime.now().isoformat(timespec='seconds')}] {msg}"
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def wake(prompt: str, extra: dict | None = None) -> None:
    payload = {"prompt": prompt, "ts": datetime.now(timezone.utc).isoformat()}
    if extra:
        payload.update(extra)
    print(f"AGENT_LOOP_WAKE_fode_rev {json.dumps(payload, ensure_ascii=False)}", flush=True)


def read_text_auto(path: Path) -> str:
    raw = path.read_bytes()
    if raw.startswith(b"\xff\xfe") or raw.startswith(b"\xfe\xff"):
        return raw.decode("utf-16", errors="replace")
    if raw.startswith(b"\xef\xbb\xbf"):
        return raw.decode("utf-8-sig", errors="replace")
    return raw.decode("utf-8", errors="replace")


def parse_e1_progress() -> dict:
    if not E1_LOG.is_file():
        return {"state": "no_log"}
    text = read_text_auto(E1_LOG)
    if "FINAL val accuracy" in text:
        # last occurrence
        acc = None
        for line in text.splitlines()[::-1]:
            if "FINAL val accuracy" in line:
                try:
                    acc = float(line.split("=")[-1].strip())
                except Exception:
                    pass
                break
        return {"state": "done", "val_acc": acc}
    # Progress bars may be CR-overwritten / mojibake; parse step/total robustly.
    import re

    matches = list(re.finditer(r"(\d+)\s*/\s*(27460|(\d{4,}))", text))
    # Prefer known Longformer total 27460 when present
    step_matches = list(re.finditer(r"(\d+)/27460", text))
    if step_matches:
        cur = int(step_matches[-1].group(1))
        total = 27460
        pct = int(100 * cur / total)
        return {"state": "running", "pct": pct, "step": cur, "total": total}
    pct_matches = list(re.finditer(r"(\d+)%", text))
    if pct_matches:
        return {"state": "running", "pct": int(pct_matches[-1].group(1))}
    if matches:
        cur = int(matches[-1].group(1))
        total = int(matches[-1].group(2))
        return {
            "state": "running",
            "pct": int(100 * cur / max(total, 1)),
            "step": cur,
            "total": total,
        }
    return {"state": "starting"}


def run(cmd: list[str], log_path: Path | None = None) -> int:
    log("RUN " + " ".join(cmd))
    if log_path:
        with log_path.open("a", encoding="utf-8") as f:
            p = subprocess.run(cmd, cwd=str(ROOT), stdout=f, stderr=subprocess.STDOUT)
            return p.returncode
    p = subprocess.run(cmd, cwd=str(ROOT))
    return p.returncode


def train_bigbird() -> None:
    if BB_PRED.is_file():
        log("BigBird already done, skip train")
        return
    # Faster recipe: more batch (32GB headroom), same epochs
    rc = run(
        [
            PY,
            "-u",
            "scripts/RACE_train_bert_models_trainer.py",
            "--data_dir",
            "race_prepared",
            "--out_dir",
            "revision/artifacts/encoder_competitive",
            "--max_len",
            "2048",
            "--epochs",
            "5",
            "--lr",
            "2e-05",
            "--batch_size",
            "4",
            "--eval_batch_size",
            "8",
            "--seed",
            "42",
            "--model_names",
            "google/bigbird-roberta-base",
            "--log_train_dynamics",
            "--train_td_max_examples",
            "5000",
        ],
        BB_LOG,
    )
    if rc != 0:
        raise RuntimeError(f"BigBird train failed rc={rc}")


def rebuild() -> None:
    run(
        [
            PY,
            "-u",
            "scripts/revision/e0_build_integrated_table.py",
            "--bert_pred_csv",
            str(LF_PRED),
            "--bert_td_csv",
            str(LF_TD),
        ]
    )
    for script in [
        "e1_encoder_cartography.py",
        "e4_agreement_sensitivity.py",
        "e6_content_audit.py",
        "e7_review_efficacy.py",
        "e3_human_machine_bridge.py",
    ]:
        run([PY, "-u", f"scripts/revision/{script}"])


def try_e2_if_keys() -> str:
    has_gpt = bool(os.environ.get("BYTEDANCE_GPT_AK") or os.environ.get("GPT_AK"))
    has_ark = bool(os.environ.get("ARK_API_KEY") or os.environ.get("DOUBAO_API_KEY"))
    has_ds = bool(os.environ.get("DEEPSEEK_API_KEY"))
    # also load .env if present
    env_path = ROOT / ".env"
    if env_path.is_file():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            if not line.strip() or line.strip().startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
        has_gpt = bool(os.environ.get("BYTEDANCE_GPT_AK") or os.environ.get("GPT_AK"))
        has_ark = bool(os.environ.get("ARK_API_KEY") or os.environ.get("DOUBAO_API_KEY"))
        has_ds = bool(os.environ.get("DEEPSEEK_API_KEY"))
    started = []
    if has_gpt:
        run([PY, "-u", "scripts/revision/e2_fill_missing_gpt.py"])
        started.append("gpt")
    if has_ark:
        run([PY, "-u", "scripts/revision/e2_fill_missing_doubao.py"])
        started.append("doubao")
    if has_ds:
        run([PY, "-u", "scripts/revision/e2_run_deepseek_backend.py"])
        started.append("deepseek")
    return ",".join(started) if started else "blocked_no_keys"


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "tick"
    log(f"monitor mode={mode}")

    if mode == "watch_e1":
        # block until Longformer artifacts exist, then chain
        while not (LF_PRED.is_file() and LF_TD.is_file()):
            prog = parse_e1_progress()
            log(f"waiting Longformer: {prog}")
            time.sleep(120)
        wake(
            "E1 Longformer finished — verify accuracy, ensure BigBird+rebuild running, update STATUS",
            {"event": "e1_longformer_done", "progress": parse_e1_progress()},
        )
        log("Longformer artifacts ready — training BigBird")
        train_bigbird()
        wake(
            "E1 BigBird finished — rebuild tables and report final encoder metrics",
            {"event": "e1_bigbird_done"},
        )
        rebuild()
        e2 = try_e2_if_keys()
        wake(
            "Encoder chain complete; check E2 status and human UIs; update STATUS.md",
            {"event": "encoder_chain_done", "e2": e2},
        )
        log("encoder chain DONE")
        return

    # tick / heartbeat
    prog = parse_e1_progress()
    e2 = "keys_absent"
    env_path = ROOT / ".env"
    if env_path.is_file():
        e2 = "env_present_try"
    ports = []
    try:
        import socket

        for port in (7860, 7861):
            s = socket.socket()
            s.settimeout(0.3)
            ok = s.connect_ex(("127.0.0.1", port)) == 0
            s.close()
            ports.append(f"{port}:{'up' if ok else 'down'}")
    except Exception:
        ports = ["unknown"]

    summary = {
        "e1": prog,
        "lf_pred": LF_PRED.is_file(),
        "bb_pred": BB_PRED.is_file(),
        "e2": e2,
        "ui": ports,
    }
    log(json.dumps(summary, ensure_ascii=False))
    wake(
        "FODE revision heartbeat: check E1 progress, keep chain alive, try E2 if keys appeared, keep UIs up",
        summary,
    )

    # If Longformer done but BigBird missing and no train log growing, start chain
    if LF_PRED.is_file() and LF_TD.is_file() and not BB_PRED.is_file():
        # avoid double-start: check if bigbird log recently written
        if not BB_LOG.is_file() or (time.time() - BB_LOG.stat().st_mtime > 600):
            if not BB_LOG.is_file() or "FINAL val accuracy" in read_text_auto(BB_LOG):
                log("tick detected Longformer done without BigBird — launching chain")
                # spawn detached chain
                subprocess.Popen(
                    [PY, "-u", str(Path(__file__)), "watch_e1"],
                    cwd=str(ROOT),
                    stdout=open(ROOT / "revision/logs/monitor_chain.out", "a"),
                    stderr=subprocess.STDOUT,
                )


if __name__ == "__main__":
    main()
