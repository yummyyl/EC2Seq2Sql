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
    GPT2LMHeadModel,
    GPT2Tokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from rouge_score import rouge_scorer
import sacrebleu


SEP = "\n###SNIPPET###\n"


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
            if line:
                rows.append(json.loads(line))
    return rows


class GPT2TrainDataset(Dataset):
    def __init__(self, examples: List[Dict[str, Any]], tokenizer: GPT2Tokenizer, max_length: int):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        ex = self.examples[idx]
        text = ex["intent"] + SEP + ex["snippet"]
        enc = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].squeeze(0)
        attention_mask = enc["attention_mask"].squeeze(0)
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def compute_text_metrics(preds: List[str], golds: List[str]) -> Dict[str, float]:
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    rsum = {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
    for p, g in zip(preds, golds):
        s = scorer.score(g, p)
        for k in rsum:
            rsum[k] += s[k].fmeasure
    n = max(1, len(preds))
    rouge = {k: v / n for k, v in rsum.items()}
    bleu = sacrebleu.corpus_bleu(preds, [golds]).score
    return {"rouge1": rouge["rouge1"], "rouge2": rouge["rouge2"], "rougeL": rouge["rougeL"], "bleu": bleu}


def save_json(path: str, obj: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


@torch.no_grad()
def generate_snippet(model: GPT2LMHeadModel, tokenizer: GPT2Tokenizer, intent: str, max_new_tokens: int, num_beams: int) -> str:
    prompt = intent + SEP
    enc = tokenizer(prompt, return_tensors="pt")
    enc = {k: v.to(model.device) for k, v in enc.items()}

    out = model.generate(
        **enc,
        max_new_tokens=max_new_tokens,
        num_beams=num_beams,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    text = tokenizer.decode(out[0], skip_special_tokens=True)
    # extract part after SEP
    if SEP in text:
        return text.split(SEP, 1)[1].strip()
    return text.strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_path", type=str, default="data/splits/train.jsonl")
    ap.add_argument("--valid_path", type=str, default="data/splits/valid.jsonl")
    ap.add_argument("--test_path", type=str, default="data/splits/test.jsonl")

    ap.add_argument("--model_name", type=str, default="gpt2")
    ap.add_argument("--output_dir", type=str, default="outputs/gpt2")
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--max_length", type=int, default=512)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--train_batch_size", type=int, default=4)
    ap.add_argument("--eval_batch_size", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--weight_decay", type=float, default=0.0)

    ap.add_argument("--num_beams", type=int, default=1)
    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--max_test_examples", type=int, default=0, help="0 = no limit")
    args = ap.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    train_rows = read_jsonl(args.train_path)
    valid_rows = read_jsonl(args.valid_path)
    test_rows = read_jsonl(args.test_path)
    print(f"Loaded splits: train={len(train_rows)}, valid={len(valid_rows)}, test={len(test_rows)}")

    tokenizer = GPT2Tokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = GPT2LMHeadModel.from_pretrained(args.model_name)

    train_ds = GPT2TrainDataset(train_rows, tokenizer, args.max_length)
    valid_ds = GPT2TrainDataset(valid_rows, tokenizer, args.max_length)

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    run_dir = os.path.join(args.output_dir, "run")
    model_dir = os.path.join(args.output_dir, "model")
    metrics_path = os.path.join(args.output_dir, "metrics.json")

    targs = TrainingArguments(
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
        report_to=[],
        seed=args.seed,
        data_seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=valid_ds,
        tokenizer=tokenizer,
        data_collator=collator,
    )

    trainer.train()

    os.makedirs(model_dir, exist_ok=True)
    trainer.save_model(model_dir)
    tokenizer.save_pretrained(model_dir)


    model.eval()
    model.to(device)

    if args.max_test_examples and args.max_test_examples > 0:
        test_rows = test_rows[: args.max_test_examples]

    preds, golds = [], []
    for ex in test_rows:
        preds.append(generate_snippet(model, tokenizer, ex["intent"], args.max_new_tokens, args.num_beams))
        golds.append(ex["snippet"])

    m = compute_text_metrics(preds, golds)

    summary = {
        "baseline": "gpt2",
        "model": args.model_name,
        "created_at": datetime.utcnow().isoformat() + "Z",
        "seed": args.seed,
        "paths": {"train": args.train_path, "valid": args.valid_path, "test": args.test_path},
        "generation": {"num_beams": args.num_beams, "max_new_tokens": args.max_new_tokens},
        "metrics": m,
    }
    save_json(metrics_path, summary)
    print("Test metrics:", m)
    print("Saved:", metrics_path)


if __name__ == "__main__":
    main()
