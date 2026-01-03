import json
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from transformers import TapasTokenizer, TapasForQuestionAnswering
from rouge_score import rouge_scorer
import sacrebleu


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass
class Config:
    data_path: str = "data.json"
    model_name: str = "google/tapas-base-finetuned-wtq"
    output_dir: str = "./tapas_model"

    seed: int = 42
    test_size: float = 0.1

    # Candidate table size per example: 1 gold + (k-1) negatives
    num_candidates: int = 32

    max_length: int = 512
    batch_size: int = 2
    lr: float = 3e-5
    weight_decay: float = 0.0
    epochs: int = 3

    # Generation-like metrics on the selected snippet string (optional but useful)
    enable_string_metrics: bool = True


class TapasCandidateDataset(Dataset):
    """
    TAPAS is a table QA model. We turn the task into table cell selection:
    - table: a single-column table with candidate snippets (strings)
    - query: intent
    - label: coordinates of the correct snippet cell (row index, column index)
    """

    def __init__(
        self,
        items: List[Dict[str, str]],
        tokenizer: TapasTokenizer,
        all_snippets: List[str],
        num_candidates: int = 32,
        max_length: int = 512,
        seed: int = 42,
    ):
        self.items = items
        self.tokenizer = tokenizer
        self.all_snippets = all_snippets
        self.num_candidates = num_candidates
        self.max_length = max_length
        self.rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self.items)

    def _sample_candidates(self, gold: str) -> Tuple[List[str], int]:
        # Ensure gold is included and sample negatives
        negatives = [s for s in self.all_snippets if s != gold]
        k = max(2, self.num_candidates)  # at least 2 candidates
        k = min(k, len(self.all_snippets))

        sampled_negs = self.rng.sample(negatives, k - 1) if len(negatives) >= (k - 1) else negatives
        candidates = [gold] + sampled_negs
        self.rng.shuffle(candidates)

        gold_row = candidates.index(gold)
        return candidates, gold_row

    def __getitem__(self, idx: int) -> Dict:
        intent = self.items[idx]["intent"]
        gold_snippet = self.items[idx]["snippet"]

        candidates, gold_row = self._sample_candidates(gold_snippet)

        # A single-column table for TAPAS
        table = pd.DataFrame({"snippet": candidates})

        # Tokenize table + query
        enc = self.tokenizer(
            table=table,
            queries=intent,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        # TAPAS expects answer coordinates: List[List[Tuple[row, col]]]
        # Here col=0 because we have a single column: "snippet"
        answer_coordinates = [[(gold_row, 0)]]
        answer_text = [gold_snippet]

        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "token_type_ids": enc["token_type_ids"].squeeze(0),
            "answer_coordinates": answer_coordinates,
            "answer_text": answer_text,
            "candidates": candidates,
            "gold_row": gold_row,
            "gold_snippet": gold_snippet,
        }


def collate_fn(batch: List[Dict]) -> Dict:
    # Stack tensors
    input_ids = torch.stack([b["input_ids"] for b in batch], dim=0)
    attention_mask = torch.stack([b["attention_mask"] for b in batch], dim=0)
    token_type_ids = torch.stack([b["token_type_ids"] for b in batch], dim=0)

    # Keep python objects as lists
    answer_coordinates = [b["answer_coordinates"][0] for b in batch]  # unwrap one level
    answer_text = [b["answer_text"][0] for b in batch]
    candidates = [b["candidates"] for b in batch]
    gold_rows = [b["gold_row"] for b in batch]
    gold_snippets = [b["gold_snippet"] for b in batch]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "token_type_ids": token_type_ids,
        "answer_coordinates": answer_coordinates,  # List[List[Tuple[int,int]]]
        "answer_text": answer_text,                # List[str]
        "candidates": candidates,                  # List[List[str]]
        "gold_rows": gold_rows,                    # List[int]
        "gold_snippets": gold_snippets,            # List[str]
    }


@torch.no_grad()
def evaluate(
    model: TapasForQuestionAnswering,
    tokenizer: TapasTokenizer,
    dataloader: DataLoader,
    device: torch.device,
    enable_string_metrics: bool = True,
) -> Dict[str, float]:
    model.eval()

    total = 0
    correct = 0

    # Optional string-based metrics on selected snippet
    if enable_string_metrics:
        rouge = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
        rouge_totals = {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
        preds_texts = []
        gold_texts = []

    for batch in tqdm(dataloader, desc="Evaluating", leave=False):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        token_type_ids = batch["token_type_ids"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        # Convert logits to predicted answer coordinates
        # tokenizer.convert_logits_to_predictions expects CPU tensors
        pred_coords, _ = tokenizer.convert_logits_to_predictions(
            {"input_ids": input_ids.cpu(), "token_type_ids": token_type_ids.cpu(), "attention_mask": attention_mask.cpu()},
            outputs.logits.cpu(),
            outputs.logits_aggregation.cpu(),
        )

        # pred_coords: List[List[Tuple[row, col]]] for each example
        for i, coords in enumerate(pred_coords):
            total += 1

            if coords is None or len(coords) == 0:
                pred_row = -1
            else:
                pred_row = coords[0][0]  # first predicted cell (row index)

            gold_row = batch["gold_rows"][i]
            if pred_row == gold_row:
                correct += 1

            if enable_string_metrics:
                cands = batch["candidates"][i]
                pred_snippet = cands[pred_row] if 0 <= pred_row < len(cands) else ""
                gold_snippet = batch["gold_snippets"][i]

                preds_texts.append(pred_snippet)
                gold_texts.append(gold_snippet)

                rs = rouge.score(gold_snippet, pred_snippet)
                rouge_totals["rouge1"] += rs["rouge1"].fmeasure
                rouge_totals["rouge2"] += rs["rouge2"].fmeasure
                rouge_totals["rougeL"] += rs["rougeL"].fmeasure

    acc = correct / max(1, total)
    metrics = {"top1_acc": acc}

    if enable_string_metrics:
        n = max(1, len(preds_texts))
        metrics.update(
            {
                "rouge1": rouge_totals["rouge1"] / n,
                "rouge2": rouge_totals["rouge2"] / n,
                "rougeL": rouge_totals["rougeL"] / n,
                "bleu": sacrebleu.corpus_bleu(preds_texts, [gold_texts]).score,
                "exact_match": float(np.mean([p == g for p, g in zip(preds_texts, gold_texts)])),
            }
        )

    return metrics


def main() -> None:
    cfg = Config(
        data_path=os.getenv("DATA_PATH", "data.json"),
        model_name=os.getenv("MODEL_NAME", "google/tapas-base-finetuned-wtq"),
        output_dir=os.getenv("OUTPUT_DIR", "./tapas_model"),
        seed=int(os.getenv("SEED", "42")),
        test_size=float(os.getenv("TEST_SIZE", "0.1")),
        num_candidates=int(os.getenv("NUM_CANDIDATES", "32")),
        max_length=int(os.getenv("MAX_LENGTH", "512")),
        batch_size=int(os.getenv("BATCH_SIZE", "2")),
        lr=float(os.getenv("LR", "3e-5")),
        weight_decay=float(os.getenv("WEIGHT_DECAY", "0.0")),
        epochs=int(os.getenv("EPOCHS", "3")),
        enable_string_metrics=os.getenv("ENABLE_STRING_METRICS", "1") == "1",
    )

    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    with open(cfg.data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Shuffle deterministically, then split
    rng = random.Random(cfg.seed)
    rng.shuffle(data)

    n_total = len(data)
    n_val = int(n_total * cfg.test_size)
    val_items = data[:n_val]
    train_items = data[n_val:]

    all_snippets = list({x["snippet"] for x in data})
    tokenizer = TapasTokenizer.from_pretrained(cfg.model_name)
    model = TapasForQuestionAnswering.from_pretrained(cfg.model_name).to(device)

    train_ds = TapasCandidateDataset(
        items=train_items,
        tokenizer=tokenizer,
        all_snippets=all_snippets,
        num_candidates=cfg.num_candidates,
        max_length=cfg.max_length,
        seed=cfg.seed,
    )
    val_ds = TapasCandidateDataset(
        items=val_items,
        tokenizer=tokenizer,
        all_snippets=all_snippets,
        num_candidates=cfg.num_candidates,
        max_length=cfg.max_length,
        seed=cfg.seed + 1,
    )

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, collate_fn=collate_fn)

    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    for epoch in range(cfg.epochs):
        model.train()
        total_loss = 0.0

        for batch in tqdm(train_loader, desc=f"Training epoch {epoch+1}/{cfg.epochs}"):
            optimizer.zero_grad()

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)

            # Note: answer_coordinates is List[List[Tuple[int,int]]]
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                answer_coordinates=batch["answer_coordinates"],
                answer_text=batch["answer_text"],
            )

            loss = outputs.loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / max(1, len(train_loader))
        metrics = evaluate(model, tokenizer, val_loader, device, enable_string_metrics=cfg.enable_string_metrics)

        print(
            f"Epoch {epoch+1}/{cfg.epochs} | loss={avg_loss:.4f} | "
            f"top1_acc={metrics['top1_acc']:.4f}"
            + (
                f" | EM={metrics.get('exact_match', 0.0):.4f}"
                f" | ROUGE-1={metrics.get('rouge1', 0.0):.4f}"
                f" ROUGE-2={metrics.get('rouge2', 0.0):.4f}"
                f" ROUGE-L={metrics.get('rougeL', 0.0):.4f}"
                f" | BLEU={metrics.get('bleu', 0.0):.4f}"
                if cfg.enable_string_metrics
                else ""
            )
        )

    os.makedirs(cfg.output_dir, exist_ok=True)
    model.save_pretrained(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir)
    print(f"Saved model and tokenizer to: {cfg.output_dir}")


if __name__ == "__main__":
    main()
