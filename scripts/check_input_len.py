import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from transformers import AutoTokenizer

try:
    from tqdm import tqdm
except Exception:  
    def tqdm(x, **kwargs):
        return x


def parse_args():
    ap = argparse.ArgumentParser(description="检查训练数据的实际 token 长度分布")

    ap.add_argument(
        "--data_csv",
        type=str,
        required=True,
        help="训练数据的 CSV 路径，例如 race_prepared/race_mcq_train.csv",
    )
    ap.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="HuggingFace 模型名或本地模型目录，例如 roberta-base / roberta-large / microsoft/deberta-v3-base",
    )
    ap.add_argument(
        "--max_len",
        type=int,
        default=256,
        help="训练时使用的 max_length（用来统计截断比例）",
    )
    ap.add_argument(
        "--sample_n",
        type=int,
        default=0,
        help="如果 >0，则只随机抽 sample_n 条样本检查，节省时间",
    )

    return ap.parse_args()


def build_text_from_row(row: pd.Series) -> str:
    article = row["article"]
    question = row["question"]
    a = row["option_a"]
    b = row["option_b"]
    c = row["option_c"]
    d = row["option_d"]

    text = (
        "Read the following passage and answer the question.\n\n"
        f"Passage:\n{article}\n\n"
        f"Question:\n{question}\n\n"
        "Options:\n"
        f"A. {a}\n"
        f"B. {b}\n"
        f"C. {c}\n"
        f"D. {d}\n\n"
        "Please choose the best answer from A, B, C, or D."
    )
    return text


def main():
    args = parse_args()

    data_path = Path(args.data_csv)
    if not data_path.is_file():
        raise FileNotFoundError(f"找不到数据文件: {data_path}")

    print(f"[INFO] 读取数据: {data_path}")
    df = pd.read_csv(data_path)

    required_cols = ["article", "question", "option_a", "option_b", "option_c", "option_d"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"CSV 中缺少列: {col}，请确认和训练数据格式一致。")

    if args.sample_n > 0 and args.sample_n < len(df):
        print(f"[INFO] 从 {len(df)} 条样本中随机抽样 {args.sample_n} 条进行检查")
        df = df.sample(n=args.sample_n, random_state=42).reset_index(drop=True)
    else:
        print(f"[INFO] 使用全部样本进行检查，共 {len(df)} 条")

    print(f"[INFO] 加载 tokenizer: {args.model_name}")
    if "deberta" in args.model_name.lower():
        tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=False)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    lengths = []
    qids = df["question_id"].tolist() if "question_id" in df.columns else [None] * len(df)

    print("[INFO] 开始逐样本计算 token 长度（不截断）...")
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        text = build_text_from_row(row)
        enc = tokenizer(
            text,
            add_special_tokens=True,
            truncation=False,
        )
        length = len(enc["input_ids"])
        lengths.append(length)

    lengths = np.array(lengths, dtype=np.int32)

    print("\n===== Token 长度统计结果 =====")
    print(f"样本数: {len(lengths)}")
    print(f"最小长度: {lengths.min()}")
    print(f"最大长度: {lengths.max()}")
    print(f"平均长度: {lengths.mean():.2f}")
    print(f"中位数: {np.median(lengths):.2f}")
    for q in [0.9, 0.95, 0.99]:
        print(f"{int(q*100)} 分位数: {np.quantile(lengths, q):.2f}")

    if args.max_len is not None and args.max_len > 0:
        over = (lengths > args.max_len)
        n_over = int(over.sum())
        ratio = n_over / len(lengths) * 100
        print(f"\n以 max_len = {args.max_len} 为截断阈值：")
        print(f"  超过 max_len 的样本数: {n_over} ({ratio:.2f}%)")
    else:
        n_over = 0

    top_k = 10
    print(f"\n最长的前 {top_k} 条样本：")
    sorted_idx = np.argsort(-lengths) 
    for i in range(min(top_k, len(lengths))):
        idx = sorted_idx[i]
        qid = qids[idx]
        print(f"  rank {i+1}: idx={idx}, question_id={qid}, length={lengths[idx]}")

    print("\n检查完成。你可以根据以上统计决定：")
    print("1）max_len=256 是否过小（如果 95% 长度 < 256，截断影响不大）；")
    print("2）是否需要改成 384/512，或对超长样本做特殊处理。")


if __name__ == "__main__":
    main()