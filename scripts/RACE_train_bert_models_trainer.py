import argparse
from pathlib import Path
import random
import numpy as np
import pandas as pd
import torch

from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    TrainerCallback,
    BigBirdTokenizer,  
    DebertaV2Tokenizer
)
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

def parse_args():
    ap = argparse.ArgumentParser(description="使用 Trainer 训练多种 BERT 系列模型并比较性能")

    ap.add_argument("--data_dir", type=str, default="race_prepared",
                    help="包含 race_mcq_train/val/test.csv 的目录")
    ap.add_argument("--out_dir", type=str, default="race_bert_models_trainer",
                    help="模型和结果输出目录")
    ap.add_argument("--max_len", type=int, default=1600,
                    help="最大序列长度")
    ap.add_argument("--batch_size", type=int, default=8,
                    help="训练 batch size（per_device_train_batch_size）")
    ap.add_argument("--eval_batch_size", type=int, default=16,
                    help="评估 batch size（per_device_eval_batch_size）")
    ap.add_argument("--epochs", type=float, default=3.0,
                    help="训练轮数 num_train_epochs")
    ap.add_argument("--lr", type=float, default=2e-5,
                    help="学习率 learning_rate")
    ap.add_argument("--seed", type=int, default=42,
                    help="随机种子")

    ap.add_argument(
        "--model_names",
        nargs="+",
        default=[
            # "models/roberta-base",
            # "models/roberta-large",
            # "models/deberta-v3-base",
            # "models/deberta-v3-large",
            'models/bigbird-roberta-base',
            'models/longformer-base-4096'
        ],
        help="要对比的模型名称列表（HuggingFace hub 模型名）",
    )

    return ap.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_text(row):
    text = (
        "Read the following passage and answer the question.\n\n"
        f"Passage:\n{row['article']}\n\n"
        f"Question:\n{row['question']}\n\n"
        "Options:\n"
        f"A. {row['option_a']}\n"
        f"B. {row['option_b']}\n"
        f"C. {row['option_c']}\n"
        f"D. {row['option_d']}\n\n"
        "Please choose the best answer from A, B, C, or D."
    )
    return text

class TrainingDynamicsCallback(TrainerCallback):
    def __init__(self, val_dataset, question_ids, output_csv: Path):
        super().__init__()
        self.val_dataset = val_dataset
        self.question_ids = list(question_ids)
        self.output_csv = Path(output_csv)
        self.records = []
        self.trainer = None 

    def set_trainer(self, trainer: Trainer):
        self.trainer = trainer

    def on_evaluate(self, args, state, control, **kwargs):
        if self.trainer is None or self.val_dataset is None:
            return
        epoch = state.epoch
        if epoch is None:
            epoch = 0.0
        pred_output = self.trainer.predict(self.val_dataset)
        logits = pred_output.predictions
        labels = pred_output.label_ids

        probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()
        pred_labels = probs.argmax(axis=-1)
        prob_correct = probs[np.arange(len(labels)), labels]
        is_correct = (pred_labels == labels).astype(int)

        if len(self.question_ids) != len(labels):
            print(
                f"[WARN][TD] question_ids 数量 {len(self.question_ids)} "
                f"与 val 样本数 {len(labels)} 不一致，跳过本次记录。"
            )
            return

        for qid, pc, ic in zip(self.question_ids, prob_correct, is_correct):
            self.records.append(
                {
                    "question_id": qid,
                    "epoch": float(epoch),
                    "prob_correct": float(pc),
                    "is_correct": int(ic),
                }
            )

    def on_train_end(self, args, state, control, **kwargs):
        if not self.records:
            print("[INFO][TD] 没有记录到任何 training dynamics，跳过写出。")
            return

        df = pd.DataFrame(self.records)
        self.output_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(self.output_csv, index=False)
        print(f"[OK][TD] 已保存 training dynamics 到: {self.output_csv.resolve()}")

class RaceMCQDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, max_len: int):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        text = build_text(row)
        label = int(row["label"])

        enc = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
        )
        item = {
            "input_ids": torch.tensor(enc["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(enc["attention_mask"], dtype=torch.long),
            "labels": torch.tensor(label, dtype=torch.long),
        }
        return item

    @property
    def question_ids(self):
        return self.df["question_id"].tolist()

    @property
    def gold_labels(self):
        return self.df["label"].astype(int).tolist()
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc}


def train_and_eval_one_model(
    model_name: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    args,
    device,
    out_root: Path,
):
    print(f"\n========== 训练模型: {model_name} ==========")

    model_dir = out_root / model_name.replace("/", "_")
    model_dir.mkdir(parents=True, exist_ok=True)

    model_name_lower = model_name.lower()

    try:
        if "bigbird" in model_name_lower or "big_bird" in model_name_lower:
            print(f"[INFO] Detected BigBird model `{model_name}`, using BigBirdTokenizer (slow).")
            tokenizer = BigBirdTokenizer.from_pretrained(model_name)
        elif "deberta" in model_name_lower:
            print(f"[INFO] Detected DeBERTa model `{model_name}`, using DebertaV2Tokenizer (slow).")
            tokenizer = DebertaV2Tokenizer.from_pretrained(model_name)
        else:
            print(f"[INFO] Using AutoTokenizer for `{model_name}` (fast -> slow fallback).")
            try:
                tokenizer = AutoTokenizer.from_pretrained(model_name)
            except AttributeError as e:
                print(f"[WARN] AutoTokenizer.fast for `{model_name}` 出错: {e}")
                print("[WARN] 回退到 use_fast=False 再试一次 …")
                tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    except Exception as e:
        print(f"[ERROR] 加载 tokenizer `{model_name}` 时出错: {e}")
        print("[INFO] 尝试 AutoTokenizer(..., use_fast=False) 兜底")
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=4,
    )

    model.to(device)

    train_dataset = RaceMCQDataset(train_df, tokenizer, args.max_len)
    val_dataset = RaceMCQDataset(val_df, tokenizer, args.max_len)
    test_dataset = RaceMCQDataset(test_df, tokenizer, args.max_len) if test_df is not None and len(test_df) > 0 else None

    training_args = TrainingArguments(
        output_dir=str(model_dir / "hf_output"),
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        eval_strategy="epoch",
        save_strategy="no",
        logging_strategy="steps",        
        logging_steps=10,
        load_best_model_at_end=False,
        report_to=["tensorboard"],
        logging_dir=str(model_dir / "tb_logs"),             
        seed=args.seed,
        fp16=torch.cuda.is_available(), 
        # fp16=False,     
        weight_decay=0.01,                 
        warmup_ratio=0.1,  
        dataloader_num_workers=64,   
        dataloader_pin_memory=True, 
        gradient_accumulation_steps=4                
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    td_csv = model_dir / "training_dynamics_val.csv"
    td_callback = TrainingDynamicsCallback(
        val_dataset=val_dataset,
        question_ids=val_dataset.question_ids,  
        output_csv=td_csv,
    )
    td_callback.set_trainer(trainer)
    trainer.add_callback(td_callback)

    trainer.train()
    print(f"[{model_name}] 在 val 上评估并导出预测结果...")
    val_pred_output = trainer.predict(val_dataset)
    val_logits = val_pred_output.predictions
    val_labels = val_pred_output.label_ids
    val_probs = torch.softmax(torch.tensor(val_logits), dim=-1).numpy()
    val_pred_labels = val_probs.argmax(axis=-1)

    val_qids = val_dataset.question_ids
    val_gold = val_dataset.gold_labels

    val_prob_correct = val_probs[np.arange(len(val_labels)), val_labels]

    val_acc = accuracy_score(val_labels, val_pred_labels)
    print(f"[{model_name}] FINAL val accuracy = {val_acc:.4f}")

    val_pred_df = pd.DataFrame({
        "question_id": val_qids,
        "gold_label": val_gold,
        "pred_label": val_pred_labels,
        "prob_correct": val_prob_correct,
    })
    val_pred_csv = model_dir / "val_predictions.csv"
    val_pred_df.to_csv(val_pred_csv, index=False)
    print(f"[{model_name}] 已保存 val_predictions.csv -> {val_pred_csv.resolve()}")

    if test_dataset is not None:
        print(f"[{model_name}] 在 test 上评估并导出预测结果...")
        test_pred_output = trainer.predict(test_dataset)
        test_logits = test_pred_output.predictions
        test_labels = test_pred_output.label_ids
        test_probs = torch.softmax(torch.tensor(test_logits), dim=-1).numpy()
        test_pred_labels = test_probs.argmax(axis=-1)

        test_qids = test_dataset.question_ids
        test_gold = test_dataset.gold_labels
        test_prob_correct = test_probs[np.arange(len(test_labels)), test_labels]

        test_acc = accuracy_score(test_labels, test_pred_labels)
        print(f"[{model_name}] FINAL test accuracy = {test_acc:.4f}")

        test_pred_df = pd.DataFrame({
            "question_id": test_qids,
            "gold_label": test_gold,
            "pred_label": test_pred_labels,
            "prob_correct": test_prob_correct,
        })
        test_pred_csv = model_dir / "test_predictions.csv"
        test_pred_df.to_csv(test_pred_csv, index=False)
        print(f"[{model_name}] 已保存 test_predictions.csv -> {test_pred_csv.resolve()}")
    else:
        test_acc = float("nan")
        print(f"[{model_name}] 未提供 test 集，跳过 test 评估。")

    try:
        model.save_pretrained(str(model_dir / "model"))
        tokenizer.save_pretrained(str(model_dir / "model"))
    except Exception as e:
        print(f"[WARN] 保存模型失败: {e}")

    return val_acc, test_acc

def main():
    args = parse_args()
    set_seed(args.seed)

    data_dir = Path(args.data_dir)
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    train_csv = data_dir / "race_mcq_train.csv"
    val_csv = data_dir / "race_mcq_val.csv"
    test_csv = data_dir / "race_mcq_test.csv"

    if not train_csv.is_file() or not val_csv.is_file():
        print("[ERROR] 需要先运行 race_prepare_and_designer_stats.py 生成 race_mcq_train/val.csv")
        return

    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)
    if test_csv.is_file():
        test_df = pd.read_csv(test_csv)
        print(f"[INFO] 发现 test 集: {test_csv}")
    else:
        test_df = None
        print("[WARN] 未发现 test 集，将只在 train/val 上训练与验证。")
        
    # 针对m系列mac适配
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"[INFO] 使用设备: {device}")

    summary_records = []

    for model_name in args.model_names:
        val_acc, test_acc = train_and_eval_one_model(
            model_name=model_name,
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            args=args,
            device=device,
            out_root=out_root,
        )
        summary_records.append({
            "model_name": model_name,
            "val_accuracy": val_acc,
            "test_accuracy": test_acc,
        })

    summary_df = pd.DataFrame(summary_records)
    summary_csv = out_root / "models_accuracy_summary.csv"
    summary_df.to_csv(summary_csv, index=False)
    print(f"\n[OK] 模型准确率汇总已保存到: {summary_csv.resolve()}")
    print(summary_df)

    plt.figure(figsize=(8, 4))
    x = np.arange(len(summary_df))
    val_accs = summary_df["val_accuracy"].values
    plt.bar(x, val_accs, width=0.6)
    plt.xticks(x, summary_df["model_name"], rotation=30, ha="right")
    plt.ylabel("Validation accuracy")
    plt.title("BERT-family models on RACE (validation accuracy, Trainer)")
    for i, v in enumerate(val_accs):
        plt.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    acc_png = out_root / "models_val_accuracy_bar.png"
    plt.tight_layout()
    plt.savefig(acc_png, dpi=300)
    plt.close()
    print(f"[OK] 已保存模型验证集准确率柱状图: {acc_png.resolve()}")

    print("\n全部完成。接下来：")
    print("1）从 models_accuracy_summary.csv 里选一个 val_accuracy 最高的模型；")
    print("2）在 race_analyze_views.py 里，把 --bert_pred_dir 换成这个模型目录；")
    print("3）继续做 Designer / BERT / LLM 三视角对齐分析。")


if __name__ == "__main__":
    main()