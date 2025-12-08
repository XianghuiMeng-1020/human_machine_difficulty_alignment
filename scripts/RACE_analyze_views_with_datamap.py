import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap


def parse_args():
    ap = argparse.ArgumentParser(description="RACE 三视角 + DataMap 对齐分析")

    ap.add_argument(
        "--race_val_csv",
        type=str,
        required=True,
        help="脚本一产出的 RACE val CSV，例如 race_prepared/race_mcq_val.csv",
    )
    ap.add_argument(
        "--bert_pred_csv",
        type=str,
        required=True,
        help="脚本二产出的 BERT 验证集预测结果 val_predictions.csv",
    )
    ap.add_argument(
        "--bert_td_csv",
        type=str,
        required=True,
        help="BERT training dynamics CSV，例如 training_dynamics_val.csv，"
             "需要包含 question_id, epoch, prob_correct, is_correct 列",
    )
    ap.add_argument(
        "--llm_res_jsonl",
        type=str,
        required=True,
        help="LLM 结果 JSONL，每行至少包含 question_id 和 LLM 预测选项（见下）",
    )
    ap.add_argument(
        "--out_dir",
        type=str,
        default="race_analysis_with_datamap",
        help="分析结果输出目录",
    )

    return ap.parse_args()

LABEL2LETTER = {0: "A", 1: "B", 2: "C", 3: "D"}
LETTER2LABEL = {v: k for k, v in LABEL2LETTER.items()}


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def load_race_val(race_val_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(race_val_csv)
    required_cols = ["question_id", "label", "answer_letter", "designer_difficulty_str"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"[ERROR] {race_val_csv} 缺少列: {col}")
    return df


def load_bert_val_pred(bert_pred_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(bert_pred_csv)
    required_cols = ["question_id", "gold_label", "pred_label", "prob_correct"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"[ERROR] {bert_pred_csv} 缺少列: {col}")
    return df


def safe_parse_llm_label(x):
    if x is None:
        return None
    if isinstance(x, str):
        x_strip = x.strip().upper()
        if x_strip in LETTER2LABEL:
            return LETTER2LABEL[x_strip]
        try:
            v = int(x_strip)
            if v in [0, 1, 2, 3]:
                return v
        except Exception:
            return None
    if isinstance(x, (int, np.integer)):
        if x in [0, 1, 2, 3]:
            return int(x)
    return None


def load_llm_results(llm_res_jsonl: Path) -> pd.DataFrame:
    records = []
    with llm_res_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)

            qid = obj.get("question_id")
            if qid is None:
                continue

            label_raw = None
            for key in ["llm_label", "pred_label", "answer"]:
                if key in obj:
                    label_raw = obj[key]
                    break

            label = safe_parse_llm_label(label_raw)
            llm_correct = obj.get("llm_correct", None)
            if isinstance(llm_correct, bool):
                llm_correct = int(llm_correct)
            elif isinstance(llm_correct, (int, np.integer)):
                llm_correct = int(llm_correct)
            else:
                llm_correct = None

            records.append(
                {
                    "question_id": qid,
                    "llm_pred_label": label,
                    "llm_correct": llm_correct,
                }
            )

    df = pd.DataFrame(records)
    if df.empty:
        raise ValueError(f"[ERROR] {llm_res_jsonl} 读取后为空，请检查 JSONL 格式和字段名。")

    df = df.drop_duplicates(subset=["question_id"], keep="first").reset_index(drop=True)
    return df


def load_bert_training_dynamics(bert_td_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(bert_td_csv)
    required_cols = ["question_id", "epoch", "prob_correct", "is_correct"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"[ERROR] {bert_td_csv} 缺少列: {col}")
    return df


def compute_datamap_metrics(td_df: pd.DataFrame) -> pd.DataFrame:
    g = td_df.groupby("question_id")
    agg = g.agg(
        mean_prob=("prob_correct", "mean"),
        std_prob=("prob_correct", "std"),
        frac_correct=("is_correct", "mean"),
        num_epochs=("epoch", "nunique"),
    ).reset_index()

    agg["std_prob"] = agg["std_prob"].fillna(0.0)

    mu_low = np.quantile(agg["mean_prob"], 0.33)
    mu_high = np.quantile(agg["mean_prob"], 0.66)
    sigma_low = np.quantile(agg["std_prob"], 0.33)
    sigma_high = np.quantile(agg["std_prob"], 0.66)

    def assign_region(row):
        mu = row["mean_prob"]
        sigma = row["std_prob"]
        frac = row["frac_correct"]

        region = "middle"
        # 优先 ambiguous
        if sigma >= sigma_high:
            region = "ambiguous"
        # 再看 hard
        if mu <= mu_low and frac < 0.5:
            region = "hard"
        # 再看 easy
        if mu >= mu_high and sigma <= sigma_low:
            region = "easy"
        return region

    agg["datamap_region"] = agg.apply(assign_region, axis=1)

    return agg

def plot_designer_vs_views(df: pd.DataFrame, out_png: Path):
    difficulties = sorted(df["designer_difficulty_str"].unique().tolist())
    metrics = []

    for diff in difficulties:
        sub = df[df["designer_difficulty_str"] == diff]
        n = len(sub)
        if n == 0:
            continue
        designer_acc = sub["is_gold"].mean()
        bert_acc = (sub["bert_correct"] == 1).mean()
        llm_acc = (sub["llm_correct"] == 1).mean()
        metrics.append((diff, designer_acc, bert_acc, llm_acc))

    if not metrics:
        return

    labels = [m[0] for m in metrics]
    designer_acc = [m[1] for m in metrics]
    bert_acc = [m[2] for m in metrics]
    llm_acc = [m[3] for m in metrics]

    x = np.arange(len(labels))
    width = 0.25

    plt.figure(figsize=(8, 4))
    plt.bar(x - width, bert_acc, width=width, label="BERT acc")
    plt.bar(x, llm_acc, width=width, label="LLM acc")
    plt.bar(x + width, designer_acc, width=width, label="Gold (1.0)")

    plt.xticks(x, labels)
    plt.ylim(0, 1.05)
    plt.ylabel("Accuracy")
    plt.title("Accuracy by Designer Difficulty (BERT / LLM)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()


def plot_datamap_scatter(df_dm: pd.DataFrame, df_merged: pd.DataFrame, out_png1: Path, out_png2: Path):
    merged = df_dm.merge(
        df_merged[["question_id", "designer_difficulty_str", "llm_correct"]],
        on="question_id",
        how="left",
        suffixes=("", "_views"),
    )

    plt.figure(figsize=(6, 5))
    diffs = sorted(merged["designer_difficulty_str"].dropna().unique().tolist())
    cmap = get_cmap("tab10")
    for i, d in enumerate(diffs):
        sub = merged[merged["designer_difficulty_str"] == d]
        plt.scatter(
            sub["mean_prob"], sub["std_prob"],
            s=10, alpha=0.7,
            label=d,
            color=cmap(i)
        )
    plt.xlabel("Mean prob_correct across epochs")
    plt.ylabel("Std prob_correct across epochs (variability)")
    plt.title("Data Map: Mean vs Variability colored by Designer difficulty")
    plt.legend(markerscale=2)
    plt.tight_layout()
    plt.savefig(out_png1, dpi=300)
    plt.close()

    plt.figure(figsize=(6, 5))
    correct = merged[merged["llm_correct"] == 1]
    wrong = merged[merged["llm_correct"] == 0]

    plt.scatter(
        wrong["mean_prob"], wrong["std_prob"],
        s=10, alpha=0.7, label="LLM wrong", color="red"
    )
    plt.scatter(
        correct["mean_prob"], correct["std_prob"],
        s=10, alpha=0.7, label="LLM correct", color="green"
    )
    plt.xlabel("Mean prob_correct across epochs")
    plt.ylabel("Std prob_correct across epochs (variability)")
    plt.title("Data Map: Mean vs Variability colored by LLM correctness")
    plt.legend(markerscale=2)
    plt.tight_layout()
    plt.savefig(out_png2, dpi=300)
    plt.close()


def plot_datamap_region_by_designer(df_dm: pd.DataFrame, df_merged: pd.DataFrame, out_png: Path):
    merged = df_dm.merge(
        df_merged[["question_id", "designer_difficulty_str"]],
        on="question_id",
        how="left",
    )

    region_order = ["easy", "ambiguous", "hard", "middle"]
    difficulties = sorted(merged["designer_difficulty_str"].dropna().unique().tolist())

    data = {d: {r: 0 for r in region_order} for d in difficulties}

    for d in difficulties:
        sub = merged[merged["designer_difficulty_str"] == d]
        total = len(sub)
        if total == 0:
            continue
        counts = sub["datamap_region"].value_counts().to_dict()
        for r in region_order:
            data[d][r] = counts.get(r, 0) / total

    bottom = np.zeros(len(difficulties))
    x = np.arange(len(difficulties))

    plt.figure(figsize=(7, 4))
    colors = {
        "easy": "#4daf4a",
        "ambiguous": "#377eb8",
        "hard": "#e41a1c",
        "middle": "#999999",
    }

    for r in region_order:
        heights = np.array([data[d][r] for d in difficulties])
        plt.bar(x, heights, bottom=bottom, label=r, color=colors.get(r, None))
        bottom += heights

    plt.xticks(x, difficulties)
    plt.ylim(0, 1.01)
    plt.ylabel("Proportion")
    plt.title("DataMap regions distribution by Designer difficulty")
    plt.legend(title="DataMap region")
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    race_val_csv = Path(args.race_val_csv)
    bert_pred_csv = Path(args.bert_pred_csv)
    bert_td_csv = Path(args.bert_td_csv)
    llm_res_jsonl = Path(args.llm_res_jsonl)

    print(f"[INFO] 读取 RACE val: {race_val_csv}")
    df_val = load_race_val(race_val_csv)

    print(f"[INFO] 读取 BERT 验证预测: {bert_pred_csv}")
    df_bert = load_bert_val_pred(bert_pred_csv)

    print(f"[INFO] 读取 LLM 结果: {llm_res_jsonl}")
    df_llm = load_llm_results(llm_res_jsonl)

    print(f"[INFO] 读取 BERT Training Dynamics: {bert_td_csv}")
    df_td = load_bert_training_dynamics(bert_td_csv)

    df = df_val.merge(
        df_bert[["question_id", "pred_label", "prob_correct"]],
        on="question_id",
        how="left",
    )
    df = df.rename(columns={
        "pred_label": "bert_pred_label",
        "prob_correct": "bert_prob_correct",
    })

    df = df.merge(
        df_llm,
        on="question_id",
        how="left",
    )

    df["is_gold"] = 1 

    df["bert_correct"] = (df["bert_pred_label"] == df["label"]).astype(int)

    mask_missing_llm_correct = df["llm_correct"].isna()
    df.loc[mask_missing_llm_correct, "llm_correct"] = (
        df.loc[mask_missing_llm_correct, "llm_pred_label"] == df.loc[mask_missing_llm_correct, "label"]
    ).astype(int)
    df["llm_correct"] = df["llm_correct"].fillna(0).astype(int)

    print("[INFO] 基于 training dynamics 计算 Data Map 指标 …")
    df_dm = compute_datamap_metrics(df_td)

    df_all = df.merge(df_dm, on="question_id", how="left")

    out_csv = out_dir / "race_val_all_views_with_datamap.csv"
    df_all.to_csv(out_csv, index=False)
    print(f"[OK] 已保存综合视角表: {out_csv.resolve()}")

    print("\n[STATS] Designer 难度分布：")
    print(df_all["designer_difficulty_str"].value_counts(dropna=False))

    print("\n[STATS] DataMap 区域分布（整体）：")
    print(df_all["datamap_region"].value_counts(dropna=False))

    print("\n[STATS] 各视角整体正确率：")
    print(f"  BERT overall acc: {df_all['bert_correct'].mean():.4f}")
    print(f"  LLM  overall acc: {df_all['llm_correct'].mean():.4f}")

    png1 = out_dir / "designer_vs_views_bar.png"
    print(f"[INFO] 画图：{png1.name}")
    plot_designer_vs_views(df_all, png1)

    png2 = out_dir / "datamap_scatter_designer.png"
    png3 = out_dir / "datamap_scatter_llm_correct.png"
    print(f"[INFO] 画图：{png2.name}, {png3.name}")
    plot_datamap_scatter(df_dm, df_all, png2, png3)

    png4 = out_dir / "datamap_region_by_designer.png"
    print(f"[INFO] 画图：{png4.name}")
    plot_datamap_region_by_designer(df_dm, df_all, png4)

    print("\n[OK] 全部 DataMap + 三视角对齐分析完成。你可以查看：")
    print(f"  1) 总表: {out_csv}")
    print(f"  2) 图: {png1}, {png2}, {png3}, {png4}")


if __name__ == "__main__":
    main()