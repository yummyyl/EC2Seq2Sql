import argparse
import json
import os
from datetime import datetime
from typing import Any, Dict, List

import numpy as np
from rouge_score import rouge_scorer
import sacrebleu

# optional deps
from openai import OpenAI


SYSTEM_PROMPT = (
    "You convert a clinical trial eligibility criterion into a DSL snippet.\n"
    "Return ONLY the DSL snippet, without explanations."
)


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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_path", type=str, default="data/splits/train.jsonl")  # used for few-shot
    ap.add_argument("--valid_path", type=str, default="data/splits/valid.jsonl")  # unused
    ap.add_argument("--test_path", type=str, default="data/splits/test.jsonl")
    ap.add_argument("--output_dir", type=str, default="outputs/gpt_3_5_turbo")
    ap.add_argument("--model", type=str, default="gpt-3.5-turbo")
    ap.add_argument("--shots", type=int, default=3)
    ap.add_argument("--max_test_examples", type=int, default=200)  # keep API cost bounded
    ap.add_argument("--temperature", type=float, default=0.0)
    args = ap.parse_args()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set.")

    client = OpenAI(api_key=api_key)

    train_rows = read_jsonl(args.train_path)
    test_rows = read_jsonl(args.test_path)
    if args.max_test_examples and args.max_test_examples > 0:
        test_rows = test_rows[: args.max_test_examples]

    # build few-shot messages
    fewshot = []
    for ex in train_rows[: max(0, args.shots)]:
        fewshot.append({"role": "user", "content": ex["intent"]})
        fewshot.append({"role": "assistant", "content": ex["snippet"]})

    preds, golds = [], []
    for ex in test_rows:
        messages = [{"role": "system", "content": SYSTEM_PROMPT}] + fewshot + [{"role": "user", "content": ex["intent"]}]
        resp = client.chat.completions.create(
            model=args.model,
            messages=messages,
            temperature=args.temperature,
        )
        out = resp.choices[0].message.content.strip()
        preds.append(out)
        golds.append(ex["snippet"])

    m = compute_text_metrics(preds, golds)

    metrics_path = os.path.join(args.output_dir, "metrics.json")
    summary = {
        "baseline": "gpt_3_5_turbo",
        "model": args.model,
        "created_at": datetime.utcnow().isoformat() + "Z",
        "settings": {"shots": args.shots, "temperature": args.temperature, "max_test_examples": args.max_test_examples},
        "paths": {"train": args.train_path, "test": args.test_path},
        "metrics": m,
    }
    save_json(metrics_path, summary)
    print("Test metrics:", m)
    print("Saved:", metrics_path)


if __name__ == "__main__":
    main()
