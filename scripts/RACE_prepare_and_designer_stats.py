import json
from pathlib import Path
import argparse

import pandas as pd
import matplotlib.pyplot as plt


def parse_args():
    ap = argparse.ArgumentParser(description="准备 RACE 数据 + 命题人视角统计")

    ap.add_argument("--train_mid", type=str, default='data/RACE/train_mid.jsonl', 
                    help="train_mid.jsonl 文件路径")
    ap.add_argument("--train_high", type=str, default='data/RACE/train_high.jsonl', 
                    help="train_high.jsonl 文件路径")
    ap.add_argument("--val_mid", type=str, default='data/RACE/dev_mid.jsonl',
                    help="val_mid.jsonl 文件路径")
    ap.add_argument("--val_high", type=str, default='data/RACE/dev_mid.jsonl',
                    help="val_high.jsonl 文件路径")
    ap.add_argument("--test_mid", type=str, default="data/RACE/test_mid.jsonl",
                    help="test_mid.jsonl 文件路径（可选，没有就留空）")
    ap.add_argument("--test_high", type=str, default="data/RACE/test_high.jsonl",
                    help="test_high.jsonl 文件路径（可选，没有就留空）")
    ap.add_argument("--out_dir", type=str, default="race_prepared",
                    help="输出目录")

    return ap.parse_args()


def read_jsonl(path: Path):
    data = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    return data


def explode_split(jsonl_path: Path, split: str, difficulty_str: str):
    records = []
    if not jsonl_path.is_file():
        print(f"[WARN] 文件不存在，跳过: {jsonl_path}")
        return records

    print(f"  -> 读取 {split} ({difficulty_str}) : {jsonl_path}")
    raw_list = read_jsonl(jsonl_path)

    for item in raw_list:
        article = item.get("article", "")
        base_id = item.get("id", "")
        questions = item.get("questions", [])
        options = item.get("options", [])
        answers = item.get("answers", [])

        if not (len(questions) == len(options) == len(answers)):
            print(f"[WARN] 样本 {base_id} 问题数/options/answers 不一致，跳过该样本")
            continue

        for i, q in enumerate(questions):
            opts = options[i]
            if len(opts) != 4:
                print(f"[WARN] 样本 {base_id} 问题 {i} 选项数 != 4，跳过该题")
                continue
            ans = str(answers[i]).strip().upper()
            if ans not in ["A", "B", "C", "D"]:
                print(f"[WARN] 样本 {base_id} 问题 {i} 答案不是 A-D：{ans}，跳过该题")
                continue
            label = ord(ans) - ord("A")

            qid = f"{base_id}_q{i}"

            # 构造自然语言文本（BERT/LLM 共用）
            prompt = (
                "Read the following passage and answer the question.\n\n"
                f"Passage:\n{article}\n\n"
                f"Question:\n{q}\n\n"
                "Options:\n"
                f"A. {opts[0]}\n"
                f"B. {opts[1]}\n"
                f"C. {opts[2]}\n"
                f"D. {opts[3]}\n\n"
                "Please choose the best answer from A, B, C, or D."
            )

            records.append({
                "question_id": qid,
                "base_id": base_id,
                "split": split,
                "designer_difficulty_str": difficulty_str,   # "MIDDLE"/"HIGH"
                "designer_difficulty": 0 if difficulty_str == "MIDDLE" else 1,
                "article": article,
                "question": q,
                "option_a": opts[0],
                "option_b": opts[1],
                "option_c": opts[2],
                "option_d": opts[3],
                "answer_letter": ans,
                "label": label,
                "prompt": prompt,
            })

    return records


def save_llm_prompts(df: pd.DataFrame, out_path: Path):
    print(f"[INFO] 导出 LLM prompts -> {out_path}")
    with out_path.open("w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            rec = {
                "id": row["question_id"],
                "prompt": row["prompt"],
                "gold_answer": row["answer_letter"],
                "designer_difficulty": row["designer_difficulty_str"],
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def plot_designer_stats(all_df: pd.DataFrame, out_dir: Path):
    print("\n===== 命题人视角难度统计（Designer difficulty） =====")
    counts = all_df["designer_difficulty_str"].value_counts().reindex(["MIDDLE", "HIGH"])
    counts = counts.fillna(0).astype(int)
    total = counts.sum()
    for k, v in counts.items():
        ratio = v / total * 100 if total > 0 else 0.0
        print(f"{k:>6}: {v:6d} ({ratio:6.2f}%)")

    print("\n按 split 分组：")
    group = (
        all_df.groupby(["split", "designer_difficulty_str"])["question_id"]
        .count()
        .unstack(fill_value=0)
        .reindex(columns=["MIDDLE", "HIGH"])
    )
    print(group)

    plt.figure(figsize=(5, 4))
    plt.bar(counts.index, counts.values)
    plt.xlabel("Designer difficulty")
    plt.ylabel("Number of questions")
    plt.title("RACE question counts by designer difficulty")
    for i, v in enumerate(counts.values):
        plt.text(i, v, str(v), ha="center", va="bottom")
    out_path = out_dir / "race_designer_difficulty_bar.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[OK] 已保存命题人视角柱状图: {out_path.resolve()}")


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_records = []
    all_records += explode_split(Path(args.train_mid), "train", "MIDDLE")
    all_records += explode_split(Path(args.train_high), "train", "HIGH")
    all_records += explode_split(Path(args.val_mid), "val", "MIDDLE")
    all_records += explode_split(Path(args.val_high), "val", "HIGH")
    if args.test_mid:
        all_records += explode_split(Path(args.test_mid), "test", "MIDDLE")
    if args.test_high:
        all_records += explode_split(Path(args.test_high), "test", "HIGH")

    if not all_records:
        print("[ERROR] 没有成功读取任何题目，请检查 jsonl 路径。")
        return

    df = pd.DataFrame(all_records)
    print(f"\n[INFO] 展开后的总题目数: {len(df)}")

    all_csv = out_dir / "race_mcq_all.csv"
    df.to_csv(all_csv, index=False)
    print(f"[OK] 已保存全部题目到: {all_csv.resolve()}")

    for split in ["train", "val", "test"]:
        sub = df[df["split"] == split].reset_index(drop=True)
        if len(sub) == 0:
            continue
        out_csv = out_dir / f"race_mcq_{split}.csv"
        sub.to_csv(out_csv, index=False)
        print(f"[OK] 已保存 {split} 集到: {out_csv.resolve()}")

    plot_designer_stats(df, out_dir)

    val_df = df[df["split"] == "val"].reset_index(drop=True)
    if len(val_df) > 0:
        save_llm_prompts(val_df, out_dir / "race_llm_prompts_val.jsonl")
    test_df = df[df["split"] == "test"].reset_index(drop=True)
    if len(test_df) > 0:
        save_llm_prompts(test_df, out_dir / "race_llm_prompts_test.jsonl")

    print("\n全部完成。")


if __name__ == "__main__":
    main()