import os
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class ClinicalBERTPairClassifier(nn.Module):
    """
    A simple text-pair classifier built on top of Bio_ClinicalBERT.
    This implementation is for binary/multi-class classification (e.g., semantic similarity).
    """

    def __init__(
        self,
        model_name: str = "emilyalsentzer/Bio_ClinicalBERT",
        num_labels: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_repr = outputs.last_hidden_state[:, 0, :]  # [CLS]
        cls_repr = self.dropout(cls_repr)
        logits = self.classifier(cls_repr)
        return logits


class TextPairDataset(Dataset):
    def __init__(
        self,
        tokenizer: AutoTokenizer,
        texts1: List[str],
        texts2: List[str],
        labels: List[int],
        max_length: int = 128,
    ):
        assert len(texts1) == len(texts2) == len(labels)
        self.tokenizer = tokenizer
        self.texts1 = texts1
        self.texts2 = texts2
        self.labels = labels
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        enc = self.tokenizer(
            self.texts1[idx],
            self.texts2[idx],
            add_special_tokens=True,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


@dataclass
class TrainConfig:
    model_name: str = "emilyalsentzer/Bio_ClinicalBERT"
    num_labels: int = 2
    max_length: int = 128
    batch_size: int = 8
    lr: float = 2e-5
    weight_decay: float = 0.0
    epochs: int = 3
    seed: int = 42
    output_dir: str = "./clinicalbert_pair_classifier"


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0

    for batch in tqdm(dataloader, desc="Training", leave=False):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        logits = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / max(1, len(dataloader))


@torch.no_grad()
def evaluate_accuracy(model: nn.Module, dataloader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0

    for batch in tqdm(dataloader, desc="Evaluating", leave=False):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        logits = model(input_ids=input_ids, attention_mask=attention_mask)
        preds = torch.argmax(logits, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return correct / max(1, total)


def save_checkpoint(model: nn.Module, tokenizer: AutoTokenizer, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    # Save encoder weights + classifier head
    torch.save(model.state_dict(), os.path.join(output_dir, "pytorch_model.bin"))
    tokenizer.save_pretrained(output_dir)


def main() -> None:
    # Allow override via environment variables (safe for public repos)
    cfg = TrainConfig(
        model_name=os.getenv("MODEL_NAME", "emilyalsentzer/Bio_ClinicalBERT"),
        num_labels=int(os.getenv("NUM_LABELS", "2")),
        max_length=int(os.getenv("MAX_LENGTH", "128")),
        batch_size=int(os.getenv("BATCH_SIZE", "8")),
        lr=float(os.getenv("LR", "2e-5")),
        weight_decay=float(os.getenv("WEIGHT_DECAY", "0.0")),
        epochs=int(os.getenv("EPOCHS", "3")),
        seed=int(os.getenv("SEED", "42")),
        output_dir=os.getenv("OUTPUT_DIR", "./clinicalbert_pair_classifier"),
    )

    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)

    # Demo data (replace with your real dataset loader)
    texts1 = ["Patient has liver cancer", "Subject aged 18-75 years"]
    texts2 = ["Diagnosis: hepatocellular carcinoma", "Eligibility: age between 18 and 75"]
    labels = [1, 1]

    dataset = TextPairDataset(
        tokenizer=tokenizer,
        texts1=texts1,
        texts2=texts2,
        labels=labels,
        max_length=cfg.max_length,
    )
    dataloader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)

    model = ClinicalBERTPairClassifier(model_name=cfg.model_name, num_labels=cfg.num_labels).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(cfg.epochs):
        avg_loss = train_one_epoch(model, dataloader, optimizer, criterion, device)
        acc = evaluate_accuracy(model, dataloader, device)
        print(f"Epoch {epoch + 1}/{cfg.epochs} | loss={avg_loss:.4f} | acc={acc:.4f}")

    save_checkpoint(model, tokenizer, cfg.output_dir)
    print(f"Saved checkpoint to: {cfg.output_dir}")


if __name__ == "__main__":
    main()
