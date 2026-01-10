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
    EncoderDecoderModel,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
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
            if line:
                rows.append(json.loads(line))
    return rows


class EncDecDataset(Dataset):
    def __init__(
        self,
        examples: List[Dict[str, Any]],
        enc_tok,
        dec_tok,
        max_source_length: int,
        max_target_length: int,
    ):
        self.examples = examples
        self.enc_tok = enc_tok
        self.dec_tok = dec_tok
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        ex = self.examples[idx]
        src = self.enc_tok(
            ex["intent"],
            max_length=self.max_source_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        tgt = self.dec_tok(
            ex["snippet"],
            max_length=self.max_target_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        labels = tgt["input_ids"].squeeze(0)
        labels[labels == self.dec_tok.pad_token_id] = -100
        return {
            "input_ids": src["input_ids"].squeeze(0),
            "attention_mask": src["attention_mask"].squeeze(0),
            "labels": labels,
        }


def build_compute_metrics(dec_tok):
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

    def compute_metrics(eval_pred):
        preds, labels = eval_pred
        decoded_preds = dec_tok.batch_decode(preds, skip_special_tokens=True)
        labels = np.where(labels == -100, dec_tok.pad_token_id, labels)
        decoded_labels = dec_tok.batch_decode(labels, skip_special_tokens=True)

        rsum = {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
        for p, g in zip(decoded_preds, decoded_labels):
            s = scorer.score(g, p)
            for k in rsum:
                rsum[k] += s[k].fmeasure
        n = max(1, len(decoded_preds))
        rouge = {k: v / n for k, v in rsum.items()}
        bleu = sacrebleu.corpus_bleu(decoded_preds, [decoded_labels]).score
        return {"rouge1": rouge["rouge1"], "rouge2": rouge["rouge2"], "rougeL": rouge["rougeL"], "bleu": bleu}

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

    ap.add_argument("--encoder_name", type=str, default="dmis-lab/biobert-base-cased-v1.1")
    ap.add_argument("--decoder_name", type=str, default="bert-base-uncased")
    ap.add_argument("--output_dir", type=str, default="outputs/biobert")
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--max_source_length", type=int, default=256)
    ap.add_argument("--max_target_length", type=int, default=256)

    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--train_batch_size", type=int, default=4)
    ap.add_argument("--eval_batch_size", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=3)
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

    enc_tok = AutoTokenizer.from_pretrained(args.encoder_name)
    dec_tok = AutoTokenizer.from_pretrained(args.decoder_name)

    model = EncoderDecoderModel.from_encoder_decoder_pretrained(args.encoder_name, args.decoder_name)

   
    model.config.decoder_start_token_id = dec_tok.cls_token_id
    model.config.eos_token_id = dec_tok.sep_token_id
    model.config.pad_token_id = dec_tok.pad_token_id
    model.config.vocab_size = model.config.decoder.vocab_size

    train_ds = EncDecDataset(train_rows, enc_tok, dec_tok, args.max_source_length, args.max_target_length)
    valid_ds = EncDecDataset(valid_rows, enc_tok, dec_tok, args.max_source_length, args.max_target_length)
    test_ds = EncDecDataset(test_rows, enc_tok, dec_tok, args.max_source_length, args.max_target_length)

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
        predict_with_generate=True,
        generation_num_beams=args.num_beams,
        generation_max_new_tokens=args.max_new_tokens,
        report_to=[],
        seed=args.seed,
        data_seed=args.seed,
    )

    collator = DataCollatorForSeq2Seq(tokenizer=dec_tok, model=model)

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=valid_ds,
        tokenizer=dec_tok, 
        data_collator=collator,
        compute_metrics=build_compute_metrics(dec_tok),
    )

    trainer.train()
    test_metrics = trainer.evaluate(eval_dataset=test_ds, metric_key_prefix="test")

    os.makedirs(model_dir, exist_ok=True)
    trainer.save_model(model_dir)
    enc_tok.save_pretrained(os.path.join(model_dir, "encoder_tokenizer"))
    dec_tok.save_pretrained(os.path.join(model_dir, "decoder_tokenizer"))

    summary = {
        "baseline": "biobert",
        "model": {"encoder": args.encoder_name, "decoder": args.decoder_name},
        "created_at": datetime.utcnow().isoformat() + "Z",
        "seed": args.seed,
        "paths": {"train": args.train_path, "valid": args.valid_path, "test": args.test_path},
        "metrics": {
            "rouge1": float(test_metrics.get("test_rouge1", 0.0)),
            "rouge2": float(test_metrics.get("test_rouge2", 0.0)),
            "rougeL": float(test_metrics.get("test_rougeL", 0.0)),
            "bleu": float(test_metrics.get("test_bleu", 0.0)),
        },
    }
    save_json(metrics_path, summary)
    print("Test metrics:", summary["metrics"])
    print("Saved:", metrics_path)


if __name__ == "__main__":
    main()
