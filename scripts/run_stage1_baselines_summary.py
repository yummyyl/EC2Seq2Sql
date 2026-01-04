import argparse
import json
import os
from typing import Any, Dict, List, Optional, Tuple


# Stage-1 baselines (Table 4 in the paper)
BASELINES: List[Tuple[str, str]] = [
    ("gpt2", "GPT-2"),
    ("t5_small", "T5-small"),
    ("t5_base", "T5-base"),
    ("biobert", "BioBERT"),
    ("clinicalbert", "ClinicalBERT"),
    ("tapas", "TAPAS"),
    ("gpt_3_5_turbo", "GPT-3.5-turbo"),
    ("bart_large_cnn", "BART-large-CNN"),
]


def _safe_get(d: Dict[str, Any], *keys: str) -> Optional[Any]:
    cur: Any = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return None
        cur = cur[k]
    return cur


def _normalize_metric_keys(metrics: Dict[str, Any]) -> Dict[str, float]:
    """
    Accept multiple possible formats:
      1) {"rouge1":..., "rouge2":..., "rougeL":..., "bleu":...}
      2) {"ROUGE_1":..., "ROUGE_2":..., "ROUGE_L":..., "BLEU":...}
      3) {"metrics": {...}}  (nested)
      4) {"ROUGE-1":..., ...}
    Return canonical keys: ROUGE_1, ROUGE_2, ROUGE_L, BLEU
    """
    # If nested under "metrics"
    nested = _safe_get(metrics, "metrics")
    if isinstance(nested, dict):
        metrics = nested

    key_map = {}
    for k, v in metrics.items():
        if not isinstance(v, (int, float)):
            continue
        kk = k.strip().lower().replace("-", "_")
        key_map[kk] = float(v)

    def pick(*cands: str) -> Optional[float]:
        for c in cands:
            if c in key_map:
                return key_map[c]
        return None

    out = {
        "ROUGE_1": pick("rouge1", "rouge_1"),
        "ROUGE_2": pick("rouge2", "rouge_2"),
        "ROUGE_L": pick("rougel", "rouge_l", "rouge_lsum"),
        "BLEU": pick("bleu", "bleu_score"),
    }
    return out


def _read_metrics(path: str) -> Optional[Dict[str, float]]:
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        return None
    return _normalize_metric_keys(obj)


def _fmt(x: Optional[float]) -> str:
    if x is None:
        return "NA"
    # match your paper table style (4 decimals)
    return f"{x:.4f}"


def _write_csv(path: str, rows: List[Dict[str, str]]) -> None:
    import csv

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["Model", "ROUGE_1", "ROUGE_2", "ROUGE_L", "BLEU"]
        )
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def _write_md(path: str, rows: List[Dict[str, str]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    header = "| Model | ROUGE-1 | ROUGE-2 | ROUGE-L | BLEU |\n|---|---:|---:|---:|---:|\n"
    lines = [header]
    for r in rows:
        lines.append(
            f"| {r['Model']} | {r['ROUGE_1']} | {r['ROUGE_2']} | {r['ROUGE_L']} | {r['BLEU']} |\n"
        )
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(lines)


def _write_tex(path: str, rows: List[Dict[str, str]], caption: str, label: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    lines = []
    lines.append("\\begin{table}[htbp]\n")
    lines.append("\\centering\n")
    lines.append(f"\\caption{{{caption}}}\n")
    lines.append(f"\\label{{{label}}}\n")
    lines.append("\\renewcommand{\\arraystretch}{1.1}\n")
    lines.append("\\begin{tabular}{lcccc}\n")
    lines.append("\\hline\n")
    lines.append("Model & ROUGE-1 & ROUGE-2 & ROUGE-L & BLEU \\\\\n")
    lines.append("\\hline\n")
    for r in rows:
        lines.append(
            f"{r['Model']} & {r['ROUGE_1']} & {r['ROUGE_2']} & {r['ROUGE_L']} & {r['BLEU']} \\\\\n"
        )
    lines.append("\\hline\n")
    lines.append("\\end{tabular}\n")
    lines.append("\\end{table}\n")

    with open(path, "w", encoding="utf-8") as f:
        f.writelines(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--outputs_dir",
        type=str,
        default="outputs",
        help="Base outputs directory that contains <baseline>/metrics.json",
    )
    ap.add_argument(
        "--out_dir",
        type=str,
        default="outputs/stage1",
        help="Where to write the summarized Stage-1 table files",
    )
    ap.add_argument(
        "--caption",
        type=str,
        default="Performance comparison of different sequence models on the Stage-1 task (EC-to-snippet).",
        help="LaTeX caption",
    )
    ap.add_argument(
        "--label",
        type=str,
        default="tab:stage1_baselines",
        help="LaTeX label",
    )
    args = ap.parse_args()

    rows: List[Dict[str, str]] = []
    missing: List[str] = []

    for folder, display in BASELINES:
        metrics_path = os.path.join(args.outputs_dir, folder, "metrics.json")
        m = _read_metrics(metrics_path)
        if m is None:
            missing.append(folder)
            rows.append(
                {
                    "Model": display,
                    "ROUGE_1": "NA",
                    "ROUGE_2": "NA",
                    "ROUGE_L": "NA",
                    "BLEU": "NA",
                }
            )
            continue

        rows.append(
            {
                "Model": display,
                "ROUGE_1": _fmt(m.get("ROUGE_1")),
                "ROUGE_2": _fmt(m.get("ROUGE_2")),
                "ROUGE_L": _fmt(m.get("ROUGE_L")),
                "BLEU": _fmt(m.get("BLEU")),
            }
        )

    csv_path = os.path.join(args.out_dir, "table_stage1.csv")
    md_path = os.path.join(args.out_dir, "table_stage1.md")
    tex_path = os.path.join(args.out_dir, "table_stage1.tex")

    _write_csv(csv_path, rows)
    _write_md(md_path, rows)
    _write_tex(tex_path, rows, caption=args.caption, label=args.label)

    print("Stage-1 summary written:")
    print(" -", csv_path)
    print(" -", md_path)
    print(" -", tex_path)

    if missing:
        print("\nWarning: missing metrics.json for:")
        for m in missing:
            print(" -", m)
        print("\nExpected path pattern:")
        print("  outputs/<baseline>/metrics.json")


if __name__ == "__main__":
    main()
