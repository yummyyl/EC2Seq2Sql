import argparse
import json
import os
import random
from datetime import datetime
from typing import Any, Dict, List

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer
from rouge_score import rouge_scorer
import sacrebleu
from tqdm import tqdm


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


def save_json(path: str, obj: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


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


@torch.no_grad()
def embed_texts(
    model: AutoModel,
    tokenizer: AutoTokenizer,
    texts: List[str],
    device: torch.device,
    batch_size: int = 32,
    max_length: int = 256,
) -> np.ndarray:
    """
    Encode each text into a normalized CLS embedding for cosine similarity retrieval.
    Returns: (N, D) float32 numpy array.
    """
    model.eval()
    vecs: List[np.ndarray] = []

    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding", leave=False):
        batch = texts[i : i + batch_size]
        enc = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        out = model(**enc)
        cls = out.last_hidden_state[:, 0, :]  # (B, H)

        cls = cls.detach().cpu().numpy().astype(np.float32)
        # normalize for cosine
        cls /= (np.linalg.norm(cls, axis=1, keepdims=True) + 1e-12)
        vecs.append(cls)

    return np.vstack(vecs)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_path", type=str, default="data/splits/train.jsonl")
    ap.add_argument("--valid_path", type=str, default="data/splits/valid.jsonl")  # unused (kept for interface)
    ap.add_argument("--test_path", type=str, default="data/splits/test.jsonl")

    ap.add_argument("--model_name", type=str, default="emilyalsentzer/Bio_ClinicalBERT")
    ap.add_argument("--output_dir", type=str, default="outputs/clinicalbert")
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--max_length", type=int, default=256)
    ap.add_argument("--embed_batch_size", type=int, default=32)

    ap.add_argument("--max_train_examples", type=int, default=20000, help="0 = use all")
    ap.add_argument("--max_test_examples", type=int, default=0, help="0 = no limit")
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_rows = read_jsonl(args.train_path)
    test_rows = read_jsonl(args.test_path)

    if args.max_train_examples and args.max_train_examples > 0:
        train_rows = train_rows[: args.max_train_examples]
    if args.max_test_examples and args.max_test_examples > 0:
        test_rows = test_rows[: args.max_test_examples]

    print(f"Using train={len(train_rows)} for retrieval, test={len(test_rows)} for evaluation")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModel.from_pretrained(args.model_name).to(device)

    train_intents = [x["intent"] for x in train_rows]
    train_snippets = [x["snippet"] for x in train_rows]
    test_intents = [x["intent"] for x in test_rows]
    golds = [x["snippet"] for x in test_rows]

    # Embed train + test intents
    train_mat = embed_texts(
        model, tokenizer, train_intents, device,
        batch_size=args.embed_batch_size, max_length=args.max_length
    )  # (N, D)
    test_mat = embed_texts(
        model, tokenizer, test_intents, device,
        batch_size=args.embed_batch_size, max_length=args.max_length
    )  # (M, D)

    # Cosine similarity via dot product (already normalized)
    preds: List[str] = []
    for i in tqdm(range(test_mat.shape[0]), desc="Retrieval", leave=False):
        sims = train_mat @ test_mat[i]
        idx = int(np.argmax(sims))
        preds.append(train_snippets[idx])

    m = compute_text_metrics(preds, golds)

    metrics_path = os.path.join(args.output_dir, "metrics.json")
    summary = {
        "baseline": "clinicalbert",
        "model": args.model_name,
        "created_at": datetime.utcnow().isoformat() + "Z",
        "seed": args.seed,
        "paths": {"train": args.train_path, "test": args.test_path},
        "retrieval": {
            "max_train_examples": args.max_train_examples,
            "max_test_examples": args.max_test_examples,
            "embed_batch_size": args.embed_batch_size,
            "max_length": args.max_length,
        },
        "metrics": m,
    }
    save_json(metrics_path, summary)

    print("Test metrics:", m)
    print("Saved:", metrics_path)


if __name__ == "__main__":
    main()
