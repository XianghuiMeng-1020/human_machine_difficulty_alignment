#!/usr/bin/env python3
"""Zero-idle post-E1 orchestrator — no claim downgrade path.

Parallel plan the instant Longformer artifacts appear:
  A) rebuild analyses on Longformer (CPU)          [immediate]
  B) BigBird competitive train (GPU)               [immediate, parallel]
  C) E2 API fills as soon as .env keys exist       [immediate/parallel]
  D) after any E2 land → merge votes → rebuild E4  [immediate]
  E) keep human UIs alive; refresh E6 sample       [immediate]
  F) after BigBird → secondary encoder tables      [immediate]

Emits AGENT_LOOP_WAKE_fode_rev for Cursor monitoring.
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
LOG = ROOT / "revision/logs/zero_idle.log"
LOCK = ROOT / "revision/artifacts/.zero_idle.lock"

# Prefer Longformer if present; else any competitive MC/encoder run that has preds+TD.
_ENCODER_CANDIDATES = [
    ROOT / "revision/artifacts/encoder_competitive/allenai_longformer-base-4096",
    ROOT / "revision/artifacts/encoder_competitive/google_bigbird-roberta-base",
    ROOT / "revision/artifacts/encoder_competitive/bert-base-uncased",
    ROOT / "revision/artifacts/encoder_competitive/roberta-base",
]


def _resolve_encoder_dir() -> Path | None:
    for d in _ENCODER_CANDIDATES:
        if (d / "val_predictions.csv").is_file() and (d / "training_dynamics_val.csv").is_file():
            return d
    return None


_enc = _resolve_encoder_dir()
LF_PRED = (_enc / "val_predictions.csv") if _enc else _ENCODER_CANDIDATES[0] / "val_predictions.csv"
LF_TD = (_enc / "training_dynamics_val.csv") if _enc else _ENCODER_CANDIDATES[0] / "training_dynamics_val.csv"
LF_TRAIN_TD = (
    (_enc / "training_dynamics_train.csv")
    if _enc
    else _ENCODER_CANDIDATES[0] / "training_dynamics_train.csv"
)
BB_DIR = ROOT / "revision/artifacts/encoder_competitive/google_bigbird-roberta-base"
BB_PRED = BB_DIR / "val_predictions.csv"
BB_TD = BB_DIR / "training_dynamics_val.csv"
BB_META = BB_DIR / "run_meta.json"
BB_LOG = ROOT / "revision/logs/e1_bigbird.log"
E1_LOG = ROOT / "revision/logs/e1_longformer.log"
ENV = ROOT / ".env"
FLAG_DIR = ROOT / "revision/artifacts/flags"


def log(msg: str) -> None:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    line = f"[{datetime.now().isoformat(timespec='seconds')}] {msg}"
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def wake(prompt: str, **extra) -> None:
    payload = {"prompt": prompt, "ts": datetime.now(timezone.utc).isoformat(), **extra}
    print(f"AGENT_LOOP_WAKE_fode_rev {json.dumps(payload, ensure_ascii=False)}", flush=True)


def flag(name: str) -> Path:
    FLAG_DIR.mkdir(parents=True, exist_ok=True)
    return FLAG_DIR / name


def marked(name: str) -> bool:
    return flag(name).is_file()


def nlines(p: Path) -> int:
    if not p.is_file():
        return 0
    return sum(1 for _ in p.open(encoding="utf-8", errors="ignore"))


def mark(name: str, text: str = "1") -> None:
    flag(name).write_text(text, encoding="utf-8")


def load_env() -> None:
    if not ENV.is_file():
        return
    for line in ENV.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#") or "=" not in s:
            continue
        k, v = s.split("=", 1)
        os.environ[k.strip()] = v.strip().strip('"').strip("'")


def spawn(name: str, cmd: list[str], log_path: Path) -> subprocess.Popen:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log(f"SPAWN {name}: {' '.join(cmd)}")
    fout = open(log_path, "a", encoding="utf-8")
    return subprocess.Popen(cmd, cwd=str(ROOT), stdout=fout, stderr=subprocess.STDOUT)


def run_blocking(cmd: list[str], log_path: Path | None = None) -> int:
    log("RUN " + " ".join(cmd))
    if log_path:
        with log_path.open("a", encoding="utf-8") as f:
            return subprocess.run(cmd, cwd=str(ROOT), stdout=f, stderr=subprocess.STDOUT).returncode
    return subprocess.run(cmd, cwd=str(ROOT)).returncode


def lf_ready() -> bool:
    d = _resolve_encoder_dir()
    if d is None:
        return False
    # refresh module-level paths used by rebuild
    global LF_PRED, LF_TD, LF_TRAIN_TD
    LF_PRED = d / "val_predictions.csv"
    LF_TD = d / "training_dynamics_val.csv"
    LF_TRAIN_TD = d / "training_dynamics_train.csv"
    return True


def parse_lf_acc() -> float | None:
    if not E1_LOG.is_file():
        return None
    raw = E1_LOG.read_bytes()
    text = raw.decode("utf-16", errors="replace") if raw[:2] in (b"\xff\xfe", b"\xfe\xff") else raw.decode(
        "utf-8", errors="replace"
    )
    acc = None
    for line in text.splitlines():
        if "FINAL val accuracy" in line:
            try:
                acc = float(line.split("=")[-1].strip())
            except Exception:
                pass
    return acc


def rebuild_longformer_stack() -> None:
    """Immediate post-LF analyses — claim-preserving numbers for manuscript."""
    if marked("rebuild_lf_done"):
        log("rebuild_lf already done")
        return
    run_blocking(
        [
            PY,
            "-u",
            "scripts/revision/e0_build_integrated_table.py",
            "--bert_pred_csv",
            str(LF_PRED),
            "--bert_td_csv",
            str(LF_TD),
        ],
        ROOT / "revision/logs/rebuild_e0.log",
    )
    # Point E1b train dynamics if present
    if LF_TRAIN_TD.is_file():
        run_blocking(
            [
                PY,
                "-u",
                "scripts/revision/e1_encoder_cartography.py",
                "--val_td_csv",
                str(LF_TD),
                "--train_td_csv",
                str(LF_TRAIN_TD),
            ],
            ROOT / "revision/logs/rebuild_e1.log",
        )
    else:
        run_blocking(
            [PY, "-u", "scripts/revision/e1_encoder_cartography.py", "--val_td_csv", str(LF_TD)],
            ROOT / "revision/logs/rebuild_e1.log",
        )
    for script in [
        "e4_agreement_sensitivity.py",
        "e6_content_audit.py",
        "e7_review_efficacy.py",
        "e3_human_machine_bridge.py",
        "e5_eedi_reliability.py",
    ]:
        run_blocking([PY, "-u", f"scripts/revision/{script}"], ROOT / f"revision/logs/rebuild_{script}.log")
    mark("rebuild_lf_done")
    wake(
        "Longformer rebuild stack DONE — manuscript encoder tables ready; keep BigBird/E2/human running",
        event="rebuild_lf_done",
        val_acc=parse_lf_acc(),
    )


def bb_done() -> bool:
    """True only when full MC run finished (mid-epoch val_predictions alone is not enough)."""
    return BB_PRED.is_file() and BB_TD.is_file() and BB_META.is_file()


def start_bigbird() -> subprocess.Popen | None:
    if bb_done() or marked("bigbird_spawned"):
        log("BigBird skip spawn")
        return None
    # MultipleChoice path — old SeqClass Trainer collapses to ~27% majority.
    p = spawn(
        "bigbird",
        [
            PY,
            "-u",
            "scripts/revision/e1_train_mc.py",
            "--model_name",
            "google/bigbird-roberta-base",
            "--data_dir",
            "race_prepared",
            "--out_dir",
            "revision/artifacts/encoder_competitive",
            "--max_len",
            "512",
            "--article_words",
            "200",
            "--epochs",
            "4",
            "--lr",
            "2e-05",
            "--batch_size",
            "2",
            "--eval_batch_size",
            "4",
            "--grad_accum",
            "8",
            "--gate_epoch1",
            "0.32",
        ],
        BB_LOG,
    )
    mark("bigbird_spawned", str(p.pid))
    wake("BigBird training STARTED in parallel with rebuild/E2", event="bigbird_started", pid=p.pid)
    return p


def ensure_uis() -> None:
    import socket

    def up(port: int) -> bool:
        s = socket.socket()
        s.settimeout(0.3)
        ok = s.connect_ex(("127.0.0.1", port)) == 0
        s.close()
        return ok

    if not up(7860):
        spawn("e6_ui", [PY, "-u", "scripts/revision/e6_rating_app.py"], ROOT / "revision/logs/e6_app.out")
    if not up(7861):
        spawn(
            "e3_ui",
            [PY, "-u", "scripts/revision/e3_bridge_collect_app.py"],
            ROOT / "revision/logs/e3_app.out",
        )


def start_e2_parallel() -> list[str]:
    """Launch each available backend fill as a separate process — no waiting between backends."""
    load_env()
    started = []
    mapping = [
        (
            "gpt",
            bool(
                os.environ.get("BYTEDANCE_GPT_AK")
                or os.environ.get("GPT_AK")
                or os.environ.get("GPT_API_KEY")
                or os.environ.get("OPENAI_API_KEY")
            ),
            [PY, "-u", "scripts/revision/e2_fill_missing_gpt.py", "--max_workers", "16"],
            ROOT / "revision/logs/e2_gpt_fill.log",
            "e2_gpt_spawned",
        ),
        (
            "doubao",
            bool(os.environ.get("ARK_API_KEY") or os.environ.get("DOUBAO_API_KEY"))
            and not marked("e2_doubao_quota_blocked"),
            [PY, "-u", "scripts/revision/e2_fill_missing_doubao.py", "--max_workers", "16"],
            ROOT / "revision/logs/e2_doubao_fill.log",
            "e2_doubao_spawned",
        ),
        (
            "deepseek",
            bool(os.environ.get("DEEPSEEK_API_KEY")),
            [PY, "-u", "scripts/revision/e2_run_deepseek_backend.py", "--max_workers", "16"],
            ROOT / "revision/logs/e2_deepseek.log",
            "e2_deepseek_spawned",
        ),
    ]
    for name, ok, cmd, logp, fl in mapping:
        if not ok or marked(fl):
            continue
        spawn(name, cmd, logp)
        mark(fl)
        started.append(name)
    if started:
        wake("E2 fills STARTED (no-downgrade backends)", event="e2_started", backends=started)
    return started


def maybe_merge_e2_and_rebuild() -> None:
    """When fill outputs grow, merge + re-aggregate votes + refresh E4/E7."""
    gpt_fill = ROOT / "LLM_out/gpt4o_1124/race_llm_prompts_val_gpt_high_fill.jsonl"
    doubao_fill = ROOT / "LLM_out/doubao_1.8/race_llm_prompts_val_doubao_high_fill.jsonl"
    ds = ROOT / "revision/artifacts/llm_deepseek_val.jsonl"

    g, d, s = nlines(gpt_fill), nlines(doubao_fill), nlines(ds)
    # Trigger merge when counts actually change (not every poll while stuck on Doubao=0)
    sig = f"{g}:{d}:{s}"
    prev = flag("e2_merge_sig").read_text(encoding="utf-8") if marked("e2_merge_sig") else ""
    if g + d + s == 0:
        return
    if sig == prev:
        return
    # Parse previous counts; only rebuild on +500 growth or a backend newly finishing.
    # NOTE: do NOT treat "g already >=3451" as finished_band — that spam-rebuilds on every Doubao tick.
    old_g = old_d = old_s = 0
    if prev.count(":") == 2:
        parts = prev.split(":")
        if all(p.isdigit() for p in parts):
            old_g, old_d, old_s = (int(parts[0]), int(parts[1]), int(parts[2]))
    prev_total = old_g + old_d + old_s
    grew = (g + d + s) - prev_total >= 500
    crossed = (g >= 3451 > old_g) or (d >= 3451 > old_d) or (s >= 4887 > old_s)
    if prev and not (grew or crossed):
        mark("e2_merge_sig", sig)
        return
    if g:
        run_blocking([PY, "-u", "scripts/revision/e2_merge_gpt_fills.py"], ROOT / "revision/logs/e2_merge.log")
    if d:
        run_blocking(
            [PY, "-u", "scripts/revision/e2_merge_doubao_fills.py"],
            ROOT / "revision/logs/e2_merge_doubao.log",
        )
    gpt_merged = ROOT / "LLM_out/gpt4o_1124/race_llm_prompts_val_gpt_merged.jsonl"
    doubao_merged = ROOT / "LLM_out/doubao_1.8/race_llm_prompts_val_doubao_merged.jsonl"
    vote_cmd = [PY, "-u", "scripts/revision/e2_llm_vote_aggregate.py"]
    if gpt_merged.is_file():
        vote_cmd += ["--gpt_jsonl", str(gpt_merged)]
    if doubao_merged.is_file():
        vote_cmd += ["--doubao_jsonl", str(doubao_merged)]
    run_blocking(vote_cmd, ROOT / "revision/logs/e2_vote.log")
    if lf_ready():
        run_blocking(
            [
                PY,
                "-u",
                "scripts/revision/e0_build_integrated_table.py",
                "--bert_pred_csv",
                str(LF_PRED),
                "--bert_td_csv",
                str(LF_TD),
            ],
            ROOT / "revision/logs/rebuild_e0.log",
        )
        for script in ["e4_agreement_sensitivity.py", "e6_content_audit.py", "e7_review_efficacy.py"]:
            run_blocking([PY, "-u", f"scripts/revision/{script}"], ROOT / f"revision/logs/rebuild_{script}.log")
    mark("e2_merge_sig", sig)
    wake("E2 merge+rebuild pulse", event="e2_merge_pulse", gpt=g, doubao=d, deepseek=s)


def post_bigbird() -> None:
    if not bb_done() or marked("bigbird_post_done"):
        return
    # Secondary comparison — use BigBird TD once the full run has finished
    run_blocking(
        [
            PY,
            "-u",
            "scripts/revision/e1_encoder_cartography.py",
            "--val_td_csv",
            str(BB_TD),
        ],
        ROOT / "revision/logs/rebuild_e1_bb.log",
    )
    mark("bigbird_post_done")
    wake("BigBird finished — secondary encoder comparison updated", event="bigbird_done")


def write_status(extra: dict) -> None:
    path = ROOT / "revision/STATUS.md"
    lines = [
        "# FODE-D-26-00032 Revision Experiment Status",
        "",
        f"Auto: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "## Zero-idle orchestrator",
        "",
        "```json",
        json.dumps(extra, indent=2, ensure_ascii=False),
        "```",
        "",
        "## Claim-preserving remaining",
        "",
        "- E2 full HIGH GPT/Doubao + DeepSeek 4887 (needs `.env`)",
        "- E3 Bridge humans http://127.0.0.1:7861",
        "- E6 audit ratings http://127.0.0.1:7860 (≥2 raters)",
        "- Do NOT substitute local LLMs for GPT/Doubao/DeepSeek in RQ3",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def blast_once_lf_ready() -> None:
    """Fire everything that must start the second LF lands — no serial waits."""
    if marked("blast_started"):
        return
    mark("blast_started")
    acc = parse_lf_acc()
    wake(
        "ZERO-IDLE BLAST: Longformer ready — starting rebuild+BigBird+E2+UI in parallel (no claim downgrade)",
        event="zero_idle_blast",
        val_acc=acc,
    )
    ensure_uis()
    bb = start_bigbird()
    # rebuild on CPU while GPU goes to BigBird
    rebuild_longformer_stack()
    start_e2_parallel()
    maybe_merge_e2_and_rebuild()
    write_status(
        {
            "val_acc": acc,
            "bigbird_pid": bb.pid if bb else None,
            "rebuild_lf": marked("rebuild_lf_done"),
            "e2": start_e2_parallel(),  # idempotent via flags
        }
    )


def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else "run"
    log(f"zero_idle mode={mode}")
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    LOCK.write_text(str(os.getpid()), encoding="utf-8")

    # Phase 0: while waiting for LF, still hunt keys / keep UIs (no GPU steal)
    last_wake = 0.0
    while not lf_ready():
        ensure_uis()
        started = start_e2_parallel()
        if started:
            maybe_merge_e2_and_rebuild()
        log(f"pre-LF wait: env={ENV.is_file()} e2_started={started} lf={lf_ready()}")
        # Cursor wake at most every 10 minutes (local log stays every 15s; BLAST still instant)
        now = time.time()
        if now - last_wake >= 1800:
            wake(
                "Waiting competitive encoder (BERT-MC/Longformer); hunting E2 keys; will BLAST on finish",
                event="pre_encoder_wait",
            )
            last_wake = now
        time.sleep(15)

    # Phase 1: BLAST
    blast_once_lf_ready()

    # Phase 2: keep driving until BigBird + E2 milestones + no more work
    last_block_wake = 0.0
    last_complete_wake = 0.0
    while True:
        ensure_uis()
        start_e2_parallel()
        maybe_merge_e2_and_rebuild()
        post_bigbird()
        load_env()
        state = {
            "lf": lf_ready(),
            "bb": bb_done(),
            "rebuild_lf": marked("rebuild_lf_done"),
            "e2_flags": {
                "gpt": marked("e2_gpt_spawned"),
                "doubao": marked("e2_doubao_spawned"),
                "deepseek": marked("e2_deepseek_spawned"),
            },
            "has_env": ENV.is_file(),
        }
        write_status(state)
        # Exit only when BigBird done AND (env absent for long OR all three e2 spawned)
        if bb_done() and marked("rebuild_lf_done") and marked("bigbird_post_done"):
            if ENV.is_file():
                # keep merging until fills plateau
                maybe_merge_e2_and_rebuild()
                now = time.time()
                # Only claim "complete" when Doubao HIGH actually has outputs (not just spawned).
                doubao_n = nlines(
                    ROOT / "LLM_out/doubao_1.8/race_llm_prompts_val_doubao_high_fill.jsonl"
                ) + nlines(ROOT / "LLM_out/doubao_1.8/race_llm_prompts_val_doubao_merged.jsonl")
                e2_filled = (
                    marked("e2_gpt_spawned")
                    and marked("e2_deepseek_spawned")
                    and doubao_n >= 1000
                )
                if e2_filled and now - last_complete_wake >= 3600:
                    wake(
                        "Machine-side zero-idle chain complete — humans still needed for E3/E6; E2 fills running",
                        event="machine_chain_complete",
                        **state,
                        doubao_n=doubao_n,
                    )
                    last_complete_wake = now
                elif (not e2_filled) and now - last_complete_wake >= 3600:
                    wake(
                        f"Waiting Doubao HIGH fill (n={doubao_n}); GPT/DeepSeek ready — close Ark safe-mode or send new ep",
                        event="blocked_doubao",
                        **state,
                        doubao_n=doubao_n,
                    )
                    last_complete_wake = now
                time.sleep(120)
                continue
            else:
                now = time.time()
                if now - last_block_wake >= 3600:
                    wake(
                        "Encoder+rebuild done; BLOCKED on .env for E2 — put keys in .env to resume fills instantly",
                        event="blocked_e2_keys",
                        **state,
                    )
                    last_block_wake = now
                time.sleep(120)
                continue
        time.sleep(30)


if __name__ == "__main__":
    main()
