import json
import os
import random
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import GPT2LMHeadModel, GPT2Tokenizer, AdamW
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class IntentSnippetDataset(Dataset):
    def __init__(self, data: List[Dict], tokenizer: GPT2Tokenizer, max_length: int = 512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.data[idx]
        input_text = item["intent"]
        target_text = item["snippet"]

        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            padding=False,
            truncation=True,
            max_length=self.max_length,
        )
        targets = self.tokenizer(
            target_text,
            return_tensors="pt",
            padding=False,
            truncation=True,
            max_length=self.max_length,
        )

        return {
            "input_ids": inputs["input_ids"].squeeze(0),
            "labels": targets["input_ids"].squeeze(0),
        }


@dataclass
class CollateConfig:
    pad_token_id: int
    label_pad_id: int = -100  # ignore padding in loss


def collate_fn(batch: List[Dict[str, torch.Tensor]], cfg: CollateConfig) -> Dict[str, torch.Tensor]:
    max_len = max(max(x["input_ids"].size(0), x["labels"].size(0)) for x in batch)

    input_ids_list = []
    labels_list = []

    for x in batch:
        inp = x["input_ids"]
        lab = x["labels"]

        if inp.size(0) < max_len:
            inp = torch.cat([inp, torch.full((max_len - inp.size(0),), cfg.pad_token_id, dtype=inp.dtype)])
        if lab.size(0) < max_len:
            lab = torch.cat([lab, torch.full((max_len - lab.size(0),), cfg.label_pad_id, dtype=lab.dtype)])

        input_ids_list.append(inp)
        labels_list.append(lab)

    input_ids = torch.stack(input_ids_list, dim=0)
    labels = torch.stack(labels_list, dim=0)
    attention_mask = (input_ids != cfg.pad_token_id).long()

    return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}


def evaluate_model(
    model: GPT2LMHeadModel,
    dataloader: DataLoader,
    tokenizer: GPT2Tokenizer,
    device: torch.device,
    max_new_tokens: int = 50,
) -> Dict[str, float]:
    model.eval()
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

    total_bleu = 0.0
    total_rouge = {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
    count = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            generated_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.eos_token_id,
            )

            generated_texts = [tokenizer.decode(g, skip_special_tokens=True) for g in generated_ids]

            # Replace -100 with pad token id for decoding labels safely
            labels_for_decode = labels.clone()
            labels_for_decode[labels_for_decode == -100] = tokenizer.pad_token_id
            target_texts = [tokenizer.decode(l, skip_special_tokens=True) for l in labels_for_decode]

            for pred, gold in zip(generated_texts, target_texts):
                reference = [gold.split()]
                candidate = pred.split()
                total_bleu += sentence_bleu(reference, candidate)

                rs = scorer.score(gold, pred)
                for k in total_rouge:
                    total_rouge[k] += rs[k].fmeasure

                count += 1

    if count == 0:
        return {"bleu": 0.0, "rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}

    return {
        "bleu": total_bleu / count,
        "rouge1": total_rouge["rouge1"] / count,
        "rouge2": total_rouge["rouge2"] / count,
        "rougeL": total_rouge["rougeL"] / count,
    }


def main() -> None:
    # Configuration via environment variables (safe for public repos)
    data_path = os.getenv("DATA_PATH", "data.json")
    output_dir = os.getenv("OUTPUT_DIR", "./fine_tuned_gpt2")
    model_name = os.getenv("MODEL_NAME", "gpt2")
    batch_size = int(os.getenv("BATCH_SIZE", "4"))
    lr = float(os.getenv("LR", "5e-5"))
    epochs = int(os.getenv("EPOCHS", "3"))
    seed = int(os.getenv("SEED", "42"))
    max_length = int(os.getenv("MAX_LENGTH", "512"))
    max_new_tokens = int(os.getenv("MAX_NEW_TOKENS", "50"))

    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    tokenizer = GPT2Tokenizer.from_pretrained(model_name, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token  # GPT-2 has no pad token by default

    dataset = IntentSnippetDataset(data=data, tokenizer=tokenizer, max_length=max_length)
    cfg = CollateConfig(pad_token_id=tokenizer.eos_token_id)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, cfg),
    )

    model = GPT2LMHeadModel.from_pretrained(model_name).to(device)
    optimizer = AdamW(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        last_loss = None

        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            last_loss = loss.item()

        metrics = evaluate_model(model, dataloader, tokenizer, device, max_new_tokens=max_new_tokens)
        print(
            f"Epoch {epoch + 1}/{epochs} | loss={last_loss:.4f} "
            f"| BLEU={metrics['bleu']:.4f} "
            f"| ROUGE-1={metrics['rouge1']:.4f} ROUGE-2={metrics['rouge2']:.4f} ROUGE-L={metrics['rougeL']:.4f}"
        )

    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Saved model and tokenizer to: {output_dir}")


if __name__ == "__main__":
    main()
