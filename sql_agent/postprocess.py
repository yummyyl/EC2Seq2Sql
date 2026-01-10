from __future__ import annotations

import re
from typing import Tuple


def strip_code_fences(text: str) -> str:
    text = text.strip()
   
    text = re.sub(r"^```(?:sql)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def ensure_semicolon(sql: str) -> str:
    sql = sql.strip()
    if not sql.endswith(";"):
        sql += ";"
    return sql


def basic_safety_checks(sql: str) -> Tuple[bool, str]:
    """
    Reject obviously dangerous SQL (DDL/DML).
    """
    lowered = sql.lower()
    banned = ["drop ", "delete ", "update ", "insert ", "alter ", "create ", "truncate "]
    for b in banned:
        if b in lowered:
            return False, f"Rejected SQL contains banned keyword: {b.strip()}"
    return True, "ok"


def postprocess_sql(raw: str) -> str:
    sql = strip_code_fences(raw)
    sql = ensure_semicolon(sql)
    return sql
