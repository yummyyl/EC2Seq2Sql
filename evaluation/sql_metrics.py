from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple


def normalize_sql(sql: str) -> str:
    sql = sql.strip()
    sql = re.sub(r"\s+", " ", sql)
    sql = sql.replace("`", "")
    return sql.lower()


def token_set(sql: str) -> set:
    # Very simple tokenizer: split by non-word characters
    tokens = re.split(r"[^a-zA-Z0-9_]+", normalize_sql(sql))
    return set(t for t in tokens if t)


def exact_match_token_set(pred_sql: str, gold_sql: str) -> float:
    return 1.0 if token_set(pred_sql) == token_set(gold_sql) else 0.0


def _fetch_all(cur: sqlite3.Cursor) -> List[Tuple[Any, ...]]:
    rows = cur.fetchall()
    # Sort to compare as sets (order-insensitive) like execution match
    return sorted(rows)


def execution_match_sqlite(pred_sql: str, gold_sql: str, db_path: str) -> float:
    """
    Execute both SQL queries on the same sqlite DB and compare result sets.
    """
    con = sqlite3.connect(db_path)
    try:
        cur = con.cursor()
        cur.execute(pred_sql)
        pred_rows = _fetch_all(cur)

        cur.execute(gold_sql)
        gold_rows = _fetch_all(cur)

        return 1.0 if pred_rows == gold_rows else 0.0
    finally:
        con.close()


@dataclass
class SQLMetrics:
    em: float
    ex: Optional[float] = None
    n: int = 0

    def to_dict(self) -> Dict[str, Any]:
        out = {"EM": self.em, "N": self.n}
        if self.ex is not None:
            out["EX"] = self.ex
        return out


def compute_metrics(
    preds: List[str],
    golds: List[str],
    db_path: Optional[str] = None,
) -> SQLMetrics:
    assert len(preds) == len(golds)
    n = len(preds)
    em_total = 0.0
    ex_total = 0.0
    ex_count = 0

    for p, g in zip(preds, golds):
        em_total += exact_match_token_set(p, g)
        if db_path:
            try:
                ex_total += execution_match_sqlite(p, g, db_path)
                ex_count += 1
            except Exception:
                # If either query fails, treat as 0 for EX
                ex_total += 0.0
                ex_count += 1

    em = em_total / max(1, n)
    ex = (ex_total / max(1, ex_count)) if db_path else None
    return SQLMetrics(em=em, ex=ex, n=n)
