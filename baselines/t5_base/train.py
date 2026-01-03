import json
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration,
    Trainer,
    TrainingArguments,
)
from rouge_score import rouge_scorer
import sacrebleu


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class Seq2SeqDataset(Dataset):
    def __init__(
        self,
        inputs: List[str],
        targets: List[str],
        tokenizer: T5Tokenizer,
        max_source_length: int = 256,
        max_target_length: int = 256,
    ):
        self.inputs = inputs
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        source = self.inputs[index]
        target = self.targets[index]

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
        # Ignore padding tokens in loss
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

        # predictions are generated token ids when predict_with_generate=True
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)

        # labels contain -100; replace with pad token id for decoding
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


def main() -> None:
    # Config via environment variables (safe for public repos)
    data_path = os.getenv("DATA_PATH", "data.json")
    model_name = os.getenv("MODEL_NAME", "t5-base")
    output_dir = os.getenv("OUTPUT_DIR", "./best_t5_base_seq2seq_model")

    test_size = float(os.getenv("TEST_SIZE", "0.1"))
    seed = int(os.getenv("SEED", "42"))

    max_source_length = int(os.getenv("MAX_SOURCE_LENGTH", "256"))
    max_target_length = int(os.getenv("MAX_TARGET_LENGTH", "256"))

    lr = float(os.getenv("LR", "5e-5"))
    train_bs = int(os.getenv("TRAIN_BATCH_SIZE", "4"))
    eval_bs = int(os.getenv("EVAL_BATCH_SIZE", "4"))
    epochs = int(os.getenv("EPOCHS", "3"))
    weight_decay = float(os.getenv("WEIGHT_DECAY", "0.01"))

    # Generation settings used during evaluation
    num_beams = int(os.getenv("NUM_BEAMS", "4"))
    max_new_tokens = int(os.getenv("MAX_NEW_TOKENS", "128"))

    set_seed(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    intents = [x["intent"] for x in data]
    snippets = [x["snippet"] for x in data]

    train_intents, val_intents, train_snippets, val_snippets = train_test_split(
        intents,
        snippets,
        test_size=test_size,
        random_state=seed,
        shuffle=True,
    )

    tokenizer = T5Tokenizer.from_pretrained(model_name)

    train_dataset = Seq2SeqDataset(
        train_intents,
        train_snippets,
        tokenizer,
        max_source_length=max_source_length,
        max_target_length=max_target_length,
    )
    val_dataset = Seq2SeqDataset(
        val_intents,
        val_snippets,
        tokenizer,
        max_source_length=max_source_length,
        max_target_length=max_target_length,
    )

    model = T5ForConditionalGeneration.from_pretrained(model_name)

    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        learning_rate=lr,
        per_device_train_batch_size=train_bs,
        per_device_eval_batch_size=eval_bs,
        num_train_epochs=epochs,
        weight_decay=weight_decay,
        logging_dir="./logs",
        logging_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        predict_with_generate=True,
        generation_num_beams=num_beams,
        generation_max_new_tokens=max_new_tokens,
        report_to=[],
        seed=seed,
        data_seed=seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        compute_metrics=build_compute_metrics(tokenizer),
    )

    trainer.train()

    os.makedirs(output_dir, exist_ok=True)
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Saved model and tokenizer to: {output_dir}")


if __name__ == "__main__":
    main()
