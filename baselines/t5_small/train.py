import argparse
import json
import os
import random
from datetime import datetime
from typing import Any, Dict, List

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer,
    Trainer,
    TrainingArguments,
)
from rouge_score import rouge_scorer
import sacrebleu


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


class Seq2SeqDataset(Dataset):
    def __init__(
        self,
        examples: List[Dict[str, Any]],
        tokenizer: T5Tokenizer,
        max_source_length: int = 256,
        max_target_length: int = 256,
    ):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        ex = self.examples[index]
        source = ex["intent"]
        target = ex["snippet"]

        source_enc = self.tokenizer(
            source,
            max_length=self.max_source_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        target_enc = self.tokenizer(
            target,
            max_length=self.max_target_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        labels = target_enc["input_ids"].squeeze(0)
        labels[labels == self.tokenizer.pad_token_id] = -100  

        return {
            "input_ids": source_enc["input_ids"].squeeze(0),
            "attention_mask": source_enc["attention_mask"].squeeze(0),
            "labels": labels,
        }


def build_compute_metrics(tokenizer: T5Tokenizer):
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred

       
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)

        
        labels = np.where(labels == -100, tokenizer.pad_token_id, labels)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        rouge_totals = {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
        for pred, gold in zip(decoded_preds, decoded_labels):
            scores = scorer.score(gold, pred)
            rouge_totals["rouge1"] += scores["rouge1"].fmeasure
            rouge_totals["rouge2"] += scores["rouge2"].fmeasure
            rouge_totals["rougeL"] += scores["rougeL"].fmeasure

        n = max(1, len(decoded_preds))
        rouge_avg = {k: v / n for k, v in rouge_totals.items()}

        bleu = sacrebleu.corpus_bleu(decoded_preds, [decoded_labels]).score

        return {
            "rouge1": rouge_avg["rouge1"],
            "rouge2": rouge_avg["rouge2"],
            "rougeL": rouge_avg["rougeL"],
            "bleu": bleu,
        }

    return compute_metrics


def save_json(path: str, obj: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_path", type=str, default="data/splits/train.jsonl")
    ap.add_argument("--valid_path", type=str, default="data/splits/valid.jsonl")
    ap.add_argument("--test_path", type=str, default="data/splits/test.jsonl")

    ap.add_argument("--model_name", type=str, default="t5-small")
    ap.add_argument("--output_dir", type=str, default="outputs/t5_small")
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--max_source_length", type=int, default=256)
    ap.add_argument("--max_target_length", type=int, default=256)

    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--train_batch_size", type=int, default=4)
    ap.add_argument("--eval_batch_size", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--weight_decay", type=float, default=0.01)

    
    ap.add_argument("--num_beams", type=int, default=4)
    ap.add_argument("--max_new_tokens", type=int, default=128)

    args = ap.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    train_rows = read_jsonl(args.train_path)
    valid_rows = read_jsonl(args.valid_path)
    test_rows = read_jsonl(args.test_path)

    print(f"Loaded splits: train={len(train_rows)}, valid={len(valid_rows)}, test={len(test_rows)}")

    tokenizer = T5Tokenizer.from_pretrained(args.model_name)
    model = T5ForConditionalGeneration.from_pretrained(args.model_name)

    train_dataset = Seq2SeqDataset(
        train_rows,
        tokenizer,
        max_source_length=args.max_source_length,
        max_target_length=args.max_target_length,
    )
    valid_dataset = Seq2SeqDataset(
        valid_rows,
        tokenizer,
        max_source_length=args.max_source_length,
        max_target_length=args.max_target_length,
    )
    test_dataset = Seq2SeqDataset(
        test_rows,
        tokenizer,
        max_source_length=args.max_source_length,
        max_target_length=args.max_target_length,
    )

    run_dir = os.path.join(args.output_dir, "run")
    model_dir = os.path.join(args.output_dir, "model")
    metrics_path = os.path.join(args.output_dir, "metrics.json")

    training_args = TrainingArguments(
        output_dir=run_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        learning_rate=args.lr,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        num_train_epochs=args.epochs,
        weight_decay=args.weight_decay,
        logging_dir=os.path.join(args.output_dir, "logs"),
        logging_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        predict_with_generate=True,
        generation_num_beams=args.num_beams,
        generation_max_new_tokens=args.max_new_tokens,
        report_to=[],
        seed=args.seed,
        data_seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        tokenizer=tokenizer,
        compute_metrics=build_compute_metrics(tokenizer),
    )

    trainer.train()

    
    test_metrics = trainer.evaluate(eval_dataset=test_dataset, metric_key_prefix="test")
    

    os.makedirs(model_dir, exist_ok=True)
    trainer.save_model(model_dir)
    tokenizer.save_pretrained(model_dir)

    summary = {
        "model": args.model_name,
        "baseline": "t5_small",
        "created_at": datetime.utcnow().isoformat() + "Z",
        "seed": args.seed,
        "paths": {
            "train": args.train_path,
            "valid": args.valid_path,
            "test": args.test_path,
            "model_dir": model_dir,
        },
        "generation": {
            "num_beams": args.num_beams,
            "max_new_tokens": args.max_new_tokens,
        },
        "metrics": {
            "rouge1": float(test_metrics.get("test_rouge1", 0.0)),
            "rouge2": float(test_metrics.get("test_rouge2", 0.0)),
            "rougeL": float(test_metrics.get("test_rougeL", 0.0)),
            "bleu": float(test_metrics.get("test_bleu", 0.0)),
        },
        "raw_eval": {k: (float(v) if isinstance(v, (int, float, np.number)) else v) for k, v in test_metrics.items()},
    }

    save_json(metrics_path, summary)
    print(f"Saved model to: {model_dir}")
    print(f"Saved metrics to: {metrics_path}")
    print("Test metrics:", summary["metrics"])


if __name__ == "__main__":
    main()
