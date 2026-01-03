"""
GPT-3.5-turbo baseline (no training).
This script calls the OpenAI API to generate DSL snippets and evaluates ROUGE/BLEU/EM.
"""
import argparse
import hashlib
import json
import os
import random
import time
from typing import Any, Dict, List, Optional

import numpy as np
from tqdm import tqdm
from rouge_score import rouge_scorer
import sacrebleu

from openai import OpenAI


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        return []
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def split_data(data: List[Dict[str, str]], test_size: float, seed: int) -> Dict[str, List[Dict[str, str]]]:
    set_seed(seed)
    data = data.copy()
    random.shuffle(data)
    n_val = int(len(data) * test_size)
    val = data[:n_val]
    train = data[n_val:]
    return {"train": train, "val": val}


def build_prompt(intent: str) -> List[Dict[str, str]]:
    # Keep it strict: output only the DSL snippet string.
    system = (
        "You are a semantic parser. Convert the user's eligibility criterion into the target DSL snippet.\n"
        "Rules:\n"
        "1) Output ONLY the DSL snippet.\n"
        "2) Do not add explanations.\n"
        "3) Do not wrap in code fences.\n"
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": intent},
    ]


def call_gpt35(
    client: OpenAI,
    model: str,
    intent: str,
    temperature: float,
    max_tokens: int,
    retries: int = 6,
    backoff_base: float = 1.5,
) -> str:
    messages = build_prompt(intent)

    last_err: Optional[Exception] = None
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            text = resp.choices[0].message.content or ""
            # Normalize: take the first non-empty line, strip whitespace.
            lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
            return lines[0] if lines else text.strip()
        except Exception as e:
            last_err = e
            sleep_s = (backoff_base ** attempt) + random.random()
            time.sleep(sleep_s)

    raise RuntimeError(f"OpenAI API failed after {retries} retries. Last error: {last_err}")


def compute_metrics(preds: List[str], golds: List[str]) -> Dict[str, float]:
    assert len(preds) == len(golds)
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

    r1 = r2 = rl = 0.0
    em = 0.0

    for p, g in zip(preds, golds):
        scores = scorer.score(g, p)
        r1 += scores["rouge1"].fmeasure
        r2 += scores["rouge2"].fmeasure
        rl += scores["rougeL"].fmeasure
        em += 1.0 if p == g else 0.0

    n = max(1, len(preds))
    bleu = sacrebleu.corpus_bleu(preds, [golds]).score

    return {
        "rouge1": r1 / n,
        "rouge2": r2 / n,
        "rougeL": rl / n,
        "bleu": bleu,
        "exact_match": em / n,
        "n": float(n),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", type=str, default="data.json")
    ap.add_argument("--model", type=str, default="gpt-3.5-turbo")
    ap.add_argument("--test_size", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--max_samples", type=int, default=-1, help="Use -1 for all.")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max_tokens", type=int, default=256)

    ap.add_argument("--cache_jsonl", type=str, default="baselines/gpt_3_5_turbo/cache.jsonl")
    ap.add_argument("--pred_jsonl", type=str, default="baselines/gpt_3_5_turbo/preds_val.jsonl")
    args = ap.parse_args()

    # OpenAI API key must be provided via environment variable.
    # See OpenAI API authentication guidance.
    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError("Missing OPENAI_API_KEY environment variable.")

    client = OpenAI()

    data = load_json(args.data_path)
    if not isinstance(data, list):
        raise ValueError("data.json must be a list of {intent, snippet} objects.")

    splits = split_data(data, test_size=args.test_size, seed=args.seed)
    val = splits["val"]

    if args.max_samples > 0:
        val = val[: args.max_samples]

    # Load cache
    cache_rows = load_jsonl(args.cache_jsonl)
    cache: Dict[str, str] = {}
    for r in cache_rows:
        if "key" in r and "prediction" in r:
            cache[r["key"]] = r["prediction"]

    preds: List[str] = []
    golds: List[str] = []
    out_rows: List[Dict[str, Any]] = []

    new_cache_rows: List[Dict[str, Any]] = []

    for ex in tqdm(val, desc=f"Running {args.model}"):
        intent = ex["intent"]
        gold = ex["snippet"]
        key = sha1(intent)

        if key in cache:
            pred = cache[key]
        else:
            pred = call_gpt35(
                client=client,
                model=args.model,
                intent=intent,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
            )
            cache[key] = pred
            new_cache_rows.append({"key": key, "intent": intent, "prediction": pred})

        preds.append(pred)
        golds.append(gold)
        out_rows.append({"intent": intent, "gold": gold, "pred": pred})

    # Save outputs
    save_jsonl(args.pred_jsonl, out_rows)

    # Append new cache entries (rewrite full cache for simplicity)
    merged_cache_rows = [{"key": k, "prediction": v} for k, v in cache.items()]
    save_jsonl(args.cache_jsonl, merged_cache_rows)

    metrics = compute_metrics(preds, golds)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
