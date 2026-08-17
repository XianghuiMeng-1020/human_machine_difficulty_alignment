import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def parse_args():
    ap = argparse.ArgumentParser(
        description="Eedi 人类难度统计（正确率阈值法）+ 饼状图 + 柱状图"
    )

    ap.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="一个或多个 Eedi 作答日志 CSV 文件路径，例如：train_task_1_2.csv train_task_3_4.csv",
    )
    ap.add_argument(
        "--out_dir",
        default=".",
        help="输出目录（默认当前目录），会写入统计结果 CSV、饼状图和柱状图",
    )
    ap.add_argument(
        "--min_attempts",
        type=int,
        default=5,
        help="每道题最少需要多少次作答才纳入统计（默认 5）",
    )
    ap.add_argument(
        "--easy_thr",
        type=float,
        default=0.8,
        help="简单题的正确率下限阈值（默认 0.8，表示 >=0.8 视为简单题）",
    )
    ap.add_argument(
        "--hard_thr",
        type=float,
        default=0.4,
        help="难题的正确率上限阈值（默认 0.4，表示 <=0.4 视为难题）",
    )

    return ap.parse_args()


def load_and_check_multiple(paths) -> pd.DataFrame:
    dfs = []
    needed = ["QuestionId", "UserId", "IsCorrect"]

    for p in paths:
        path = Path(p)
        print(f"  -> 读取文件: {path.resolve()}")
        df = pd.read_csv(path)

        missing = [c for c in needed if c not in df.columns]
        if missing:
            raise ValueError(f"文件 {path} 缺少必要列: {missing}")

        # 统一列名
        df = df.rename(
            columns={
                "QuestionId": "question_id",
                "UserId": "student_id",
                "IsCorrect": "is_correct",
            }
        )
        df["is_correct"] = df["is_correct"].astype(int)

        dfs.append(df)

    if not dfs:
        raise ValueError("未读取到任何有效的输入文件。")

    df_all = pd.concat(dfs, ignore_index=True)
    print(f"\n合并后总作答记录数: {len(df_all)}")
    return df_all


def compute_question_difficulty(
    df: pd.DataFrame, min_attempts: int = 5
) -> pd.DataFrame:
    agg = (
        df.groupby("question_id")["is_correct"]
        .agg(n_attempts="count", n_correct="sum", mean_correct="mean")
        .reset_index()
    )

    before = len(agg)
    agg = agg[agg["n_attempts"] >= min_attempts].copy()
    after = len(agg)
    print(f"题目总数={before}，过滤作答次数 < {min_attempts} 后，剩余题目数={after}")

    return agg


def bucket_by_rate(agg: pd.DataFrame, easy_thr: float, hard_thr: float) -> pd.DataFrame:
    if not (0.0 < hard_thr < easy_thr < 1.0):
        raise ValueError(
            f"阈值设置不合理：需要满足 0 < hard_thr < easy_thr < 1, "
            f"当前 hard_thr={hard_thr}, easy_thr={easy_thr}"
        )

    print(f"\n使用正确率阈值划分难度：easy_thr={easy_thr}, hard_thr={hard_thr}")
    print("  - mean_correct >= easy_thr -> Human-Easy")
    print("  - mean_correct <= hard_thr -> Human-Hard")
    print("  - 其他                     -> Human-Mid")

    def bucket(p):
        if p >= easy_thr:
            return "Human-Easy"
        elif p <= hard_thr:
            return "Human-Hard"
        else:
            return "Human-Mid"

    agg["human_bucket"] = agg["mean_correct"].apply(bucket)

    # 顺便加一个“难度得分” = 1 - 正确率，方便后续分析
    agg["human_diff_score"] = 1.0 - agg["mean_correct"]

    return agg


def summarize_and_plot_pie(agg: pd.DataFrame, out_dir: Path):
    counts = agg["human_bucket"].value_counts().reindex(
        ["Human-Easy", "Human-Mid", "Human-Hard"]
    )
    counts = counts.fillna(0).astype(int)

    total = counts.sum()
    print("\n===== Human difficulty 题目数量统计（合并 tasks 后的整体） =====")
    for k, v in counts.items():
        ratio = v / total * 100 if total > 0 else 0.0
        print(f"{k:>11}: {v:6d}  ({ratio:6.2f}%)")

    # 保存统计结果 CSV
    summary_path = out_dir / "eedi_human_difficulty_summary.csv"
    counts_df = counts.reset_index()
    counts_df.columns = ["human_bucket", "n_items"]
    counts_df["ratio"] = counts_df["n_items"] / total
    counts_df.to_csv(summary_path, index=False)
    print(f"\n[OK] 已保存难-中-易比例统计到: {summary_path.resolve()}")

    # 画饼状图
    labels = counts.index.tolist()
    sizes = counts.values.tolist()

    plt.figure(figsize=(6, 6))
    plt.pie(
        sizes,
        labels=labels,
        autopct="%1.1f%%",
        startangle=90,
        wedgeprops={"edgecolor": "white"},
    )
    plt.title("Eedi Human Difficulty Distribution (Easy / Mid / Hard, by success rate)")
    plt.axis("equal")  # 让饼图是圆的

    pie_path = out_dir / "eedi_human_difficulty_pie.png"
    plt.savefig(pie_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[OK] 已保存饼状图到: {pie_path.resolve()}")


def plot_bar_correct_counts(agg: pd.DataFrame, out_dir: Path):
    agg_sorted = agg.sort_values("mean_correct").reset_index(drop=True)
    x = range(len(agg_sorted))
    heights = agg_sorted["n_correct"].values
    buckets = agg_sorted["human_bucket"].values

    # 定义颜色映射
    color_map = {
        "Human-Hard": "#d73027",  # 红
        "Human-Mid": "#fc8d59",   # 橙
        "Human-Easy": "#1a9850",  # 绿
    }
    colors = [color_map.get(b, "#999999") for b in buckets]

    plt.figure(figsize=(14, 4))
    plt.bar(x, heights, color=colors, width=1.0)

    plt.xlabel("Question index (sorted by success rate, hardest → easiest)")
    plt.ylabel("Number of correct answers per question")
    plt.title("Per-question correct counts (color by Human difficulty bucket)")

    # 画一个简单的图例
    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor=color_map["Human-Hard"], label="Human-Hard"),
        Patch(facecolor=color_map["Human-Mid"], label="Human-Mid"),
        Patch(facecolor=color_map["Human-Easy"], label="Human-Easy"),
    ]
    plt.legend(handles=legend_handles, loc="upper left")

    bar_path = out_dir / "eedi_question_correct_counts_bar.png"
    plt.tight_layout()
    plt.savefig(bar_path, dpi=300)
    plt.close()
    print(f"[OK] 已保存柱状图到: {bar_path.resolve()}")


def main():
    args = parse_args()

    input_paths = args.inputs
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("开始读取多个 train_task 文件并合并：")
    df = load_and_check_multiple(input_paths)

    print("\n计算每题正确率...")
    agg = compute_question_difficulty(df, min_attempts=args.min_attempts)

    print("\n根据正确率阈值划分 Human-Easy / Human-Mid / Human-Hard...")
    agg = bucket_by_rate(agg, easy_thr=args.easy_thr, hard_thr=args.hard_thr)

    # 保存题级难度明细
    detail_path = out_dir / "eedi_question_human_difficulty.csv"
    agg.to_csv(detail_path, index=False)
    print(f"[OK] 已保存题级人类难度明细到: {detail_path.resolve()}")

    print("\n汇总难度比例并绘制饼状图...")
    summarize_and_plot_pie(agg, out_dir)

    print("\n绘制各题答对人数柱状图...")
    plot_bar_correct_counts(agg, out_dir)

    print("\n全部完成。")


if __name__ == "__main__":
    main()