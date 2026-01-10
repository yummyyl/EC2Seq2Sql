import argparse
import json
import os
import random
from datetime import datetime
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from transformers import TapasTokenizer, TapasModel
from rouge_score import rouge_scorer
import sacrebleu
import pandas as pd
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
def embed_text(model: TapasModel, tokenizer: TapasTokenizer, intent: str, device: torch.device) -> np.ndarray:
    
    table = pd.DataFrame({"text": [intent]})
    enc = tokenizer(table=table, queries=["query"], return_tensors="pt", padding="max_length", truncation=True)
    enc = {k: v.to(device) for k, v in enc.items()}
    out = model(**enc)
    
    vec = out.pooler_output[0].detach().cpu().numpy()
   
    vec = vec / (np.linalg.norm(vec) + 1e-12)
    return vec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_path", type=str, default="data/splits/train.jsonl")
    ap.add_argument("--valid_path", type=str, default="data/splits/valid.jsonl") 
    ap.add_argument("--test_path", type=str, default="data/splits/test.jsonl")
    ap.add_argument("--model_name", type=str, default="google/tapas-base")
    ap.add_argument("--output_dir", type=str, default="outputs/tapas")
    ap.add_argument("--seed", type=int, default=42)

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

    tokenizer = TapasTokenizer.from_pretrained(args.model_name)
    model = TapasModel.from_pretrained(args.model_name).to(device)
    model.eval()

    
    train_vecs = []
    train_snips = []
    for ex in tqdm(train_rows, desc="Embedding train"):
        train_vecs.append(embed_text(model, tokenizer, ex["intent"], device))
        train_snips.append(ex["snippet"])
    train_mat = np.vstack(train_vecs)  

    preds, golds = [], []
    for ex in tqdm(test_rows, desc="Retrieval on test"):
        q = embed_text(model, tokenizer, ex["intent"], device)
        sims = train_mat @ q  
        idx = int(np.argmax(sims))
        preds.append(train_snips[idx])
        golds.append(ex["snippet"])

    m = compute_text_metrics(preds, golds)

    metrics_path = os.path.join(args.output_dir, "metrics.json")
    summary = {
        "baseline": "tapas",
        "model": args.model_name,
        "created_at": datetime.utcnow().isoformat() + "Z",
        "seed": args.seed,
        "paths": {"train": args.train_path, "test": args.test_path},
        "retrieval": {"max_train_examples": args.max_train_examples, "max_test_examples": args.max_test_examples},
        "metrics": m,
    }
    save_json(metrics_path, summary)
    print("Test metrics:", m)
    print("Saved:", metrics_path)


if __name__ == "__main__":
    main()
