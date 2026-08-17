#!/usr/bin/env python3
"""Independent RACE raw-split audit. Does NOT import project count constants."""
from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EVID = Path(__file__).resolve().parents[1] / "evidence"
EVID.mkdir(parents=True, exist_ok=True)

RAW = {
    ("train", "MIDDLE"): ROOT / "data/RACE/train_mid.jsonl",
    ("train", "HIGH"): ROOT / "data/RACE/train_high.jsonl",
    ("dev", "MIDDLE"): ROOT / "data/RACE/dev_mid.jsonl",
    ("dev", "HIGH"): ROOT / "data/RACE/dev_high.jsonl",
    ("test", "MIDDLE"): ROOT / "data/RACE/test_mid.jsonl",
    ("test", "HIGH"): ROOT / "data/RACE/test_high.jsonl",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def enumerate_file(path: Path, split: str, band: str):
    rows = []
    issues = []
    if not path.is_file():
        issues.append(f"MISSING {path}")
        return rows, issues
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception as e:
                issues.append(f"{path.name}:{line_no} JSON {e}")
                continue
            base_id = str(obj.get("id", ""))
            article = obj.get("article", "")
            questions = obj.get("questions", []) or []
            options = obj.get("options", []) or []
            answers = obj.get("answers", []) or []
            if not (len(questions) == len(options) == len(answers)):
                issues.append(f"{base_id} len mismatch q/o/a")
                continue
            for i, q in enumerate(questions):
                opts = options[i]
                ans = str(answers[i]).strip().upper()
                qid = f"{base_id}_q{i}"
                ok_opts = isinstance(opts, (list, tuple)) and len(opts) == 4
                ok_ans = ans in {"A", "B", "C", "D"}
                if not ok_opts:
                    issues.append(f"{qid} opts!=4")
                if not ok_ans:
                    issues.append(f"{qid} bad ans {ans}")
                rows.append(
                    {
                        "question_id": qid,
                        "split": split,
                        "grade_band": band,
                        "source_file": path.name,
                        "passage_file": base_id,
                        "question_index": i,
                        "gold_option": ans if ok_ans else "",
                        "n_options": len(opts) if isinstance(opts, (list, tuple)) else -1,
                        "has_article": bool(str(article).strip()),
                        "has_question": bool(str(q).strip()),
                        "valid": bool(ok_opts and ok_ans and str(article).strip() and str(q).strip()),
                    }
                )
    return rows, issues


def main():
    log_lines = []
    all_rows = []
    all_issues = []
    file_hashes = {}
    for (split, band), path in RAW.items():
        rows, issues = enumerate_file(path, split, band)
        all_rows.extend(rows)
        all_issues.extend(issues)
        if path.is_file():
            file_hashes[str(path.relative_to(ROOT))] = {
                "sha256": sha256(path),
                "bytes": path.stat().st_size,
                "mtime": path.stat().st_mtime,
                "n_questions": len(rows),
            }
        log_lines.append(f"{split}/{band}: path={path} exists={path.is_file()} n={len(rows)} issues={len(issues)}")

    # uniqueness
    ids = [r["question_id"] for r in all_rows]
    dup = [k for k, v in Counter(ids).items() if v > 1]
    log_lines.append(f"total_questions={len(all_rows)} unique_ids={len(set(ids))} duplicate_ids={len(dup)}")
    if dup[:10]:
        log_lines.append(f"dup_examples={dup[:10]}")

    counts = defaultdict(lambda: defaultdict(int))
    valid_counts = defaultdict(lambda: defaultdict(int))
    for r in all_rows:
        counts[r["split"]][r["grade_band"]] += 1
        if r["valid"]:
            valid_counts[r["split"]][r["grade_band"]] += 1

    split_json = {
        "raw_files": file_hashes,
        "counts_all_rows": {s: dict(b) for s, b in counts.items()},
        "counts_valid_only": {s: dict(b) for s, b in valid_counts.items()},
        "totals_by_split": {s: sum(b.values()) for s, b in counts.items()},
        "dev_total": sum(counts["dev"].values()),
        "dev_middle": counts["dev"]["MIDDLE"],
        "dev_high": counts["dev"]["HIGH"],
        "n_duplicate_ids": len(dup),
        "n_issues": len(all_issues),
        "issues_head": all_issues[:50],
        "official_dev_claim_target": {"n": 4887, "middle": 1436, "high": 3451},
        "matches_official_dev_claim": (
            counts["dev"]["MIDDLE"] == 1436
            and counts["dev"]["HIGH"] == 3451
            and sum(counts["dev"].values()) == 4887
        ),
    }

    # write manifest csv
    import csv

    man = EVID / "race_raw_manifest.csv"
    with man.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()) if all_rows else ["question_id"])
        w.writeheader()
        w.writerows(all_rows)

    (EVID / "race_split_counts.json").write_text(json.dumps(split_json, indent=2), encoding="utf-8")
    (EVID / "race_split_audit.log").write_text("\n".join(log_lines) + "\n", encoding="utf-8")

    # compare to prepared/integrated if present
    comparisons = {}
    for label, p in [
        ("race_mcq_val", ROOT / "race_prepared/race_mcq_val.csv"),
        ("integrated", ROOT / "revision/artifacts/race_val_integrated.csv"),
        ("manifest", ROOT / "revision/tables/manifest.json"),
    ]:
        if not p.is_file():
            comparisons[label] = {"exists": False}
            continue
        if p.suffix == ".json":
            obj = json.loads(p.read_text(encoding="utf-8"))
            comparisons[label] = {
                "exists": True,
                "n_val": obj.get("n_val"),
                "n_middle": obj.get("n_middle"),
                "n_high": obj.get("n_high"),
                "sha256": sha256(p),
            }
        else:
            # count without pandas dependency if needed
            import pandas as pd

            df = pd.read_csv(p, usecols=lambda c: c in {"question_id", "split", "designer_difficulty_str", "difficulty_str"})
            band_col = "designer_difficulty_str" if "designer_difficulty_str" in df.columns else None
            if band_col is None and "difficulty_str" in df.columns:
                band_col = "difficulty_str"
            vc = df[band_col].value_counts().to_dict() if band_col else {}
            comparisons[label] = {
                "exists": True,
                "n": int(len(df)),
                "band_counts": {str(k): int(v) for k, v in vc.items()},
                "n_unique_qid": int(df["question_id"].nunique()) if "question_id" in df.columns else None,
                "sha256": sha256(p),
                "matches_raw_dev": (
                    len(df) == split_json["dev_total"]
                    and vc.get("MIDDLE", vc.get("mid", -1)) == split_json["dev_middle"]
                    and vc.get("HIGH", vc.get("high", -1)) == split_json["dev_high"]
                ),
            }

    (EVID / "race_split_comparisons.json").write_text(json.dumps(comparisons, indent=2), encoding="utf-8")
    print(json.dumps({"dev": split_json["totals_by_split"].get("dev"), **{k: split_json[k] for k in ["dev_middle", "dev_high", "matches_official_dev_claim"]}}, indent=2))
    print("WROTE", man)
    return 0 if split_json["matches_official_dev_claim"] else 2


if __name__ == "__main__":
    sys.exit(main())
