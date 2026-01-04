from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List, Optional

from sql_agent import SQLAgent, SQLAgentConfig
from sql_agent.generator import load_snippet_from_example
from evaluation.sql_metrics import compute_metrics


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_path", type=str, required=True, help="JSONL input with at least 'snippet'.")
    ap.add_argument("--schema_path", type=str, default="data/schemas/demo_schema.json")
    ap.add_argument("--provider", type=str, default=os.getenv("PROVIDER", "langchain_openai"))
    ap.add_argument("--model", type=str, default=os.getenv("MODEL_NAME", "gpt-4o-mini"))
    ap.add_argument("--temperature", type=float, default=float(os.getenv("TEMPERATURE", "0.0")))
    ap.add_argument("--max_tokens", type=int, default=int(os.getenv("MAX_TOKENS", "512")))
    ap.add_argument("--timeout_s", type=int, default=int(os.getenv("TIMEOUT_S", "120")))
    ap.add_argument("--max_retries", type=int, default=int(os.getenv("MAX_RETRIES", "2")))
    ap.add_argument("--db_path", type=str, default=os.getenv("DB_PATH", ""), help="Optional sqlite db path for EX.")
    ap.add_argument("--outputs_dir", type=str, default="outputs")
    args = ap.parse_args()

    cfg = SQLAgentConfig(
        schema_path=args.schema_path,
        provider=args.provider,
        model=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        timeout_s=args.timeout_s,
        max_retries=args.max_retries,
    )
    agent = SQLAgent(cfg)

    rows = read_jsonl(args.in_path)
    preds = []
    golds = []
    out_rows = []

    for r in rows:
        snippet = load_snippet_from_example(r)
        pred_sql, meta = agent.generate_sql(snippet)

        out = dict(r)
        out["pred_sql"] = pred_sql
        out["meta"] = meta
        out_rows.append(out)

        if "sql" in r:
            preds.append(pred_sql)
            golds.append(r["sql"])

    model_tag = args.model.replace("/", "_")
    out_dir = os.path.join(args.outputs_dir, model_tag)
    os.makedirs(out_dir, exist_ok=True)

    pred_path = os.path.join(out_dir, "predictions.jsonl")
    write_jsonl(pred_path, out_rows)

    metrics = {}
    if golds:
        db_path = args.db_path.strip() or None
        m = compute_metrics(preds, golds, db_path=db_path)
        metrics = m.to_dict()
    else:
        metrics = {"N": len(rows), "note": "No gold 'sql' provided in input; metrics not computed."}

    metrics_path = os.path.join(out_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print("Done.")
    print("Predictions:", pred_path)
    print("Metrics:", metrics_path)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
