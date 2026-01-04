from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class TableSchema:
    name: str
    columns: List[str]
    primary_key: Optional[str] = None


@dataclass
class DatabaseSchema:
    dialect: str
    tables: List[TableSchema]
    cohort_table: str = "patients"
    cohort_id_column: str = "subject_id"

    @staticmethod
    def from_json(path: str) -> "DatabaseSchema":
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)

        tables = []
        for t in obj["tables"]:
            tables.append(
                TableSchema(
                    name=t["name"],
                    columns=list(t["columns"]),
                    primary_key=t.get("primary_key"),
                )
            )
        return DatabaseSchema(
            dialect=obj.get("dialect", "sqlite"),
            tables=tables,
            cohort_table=obj.get("cohort_table", "patients"),
            cohort_id_column=obj.get("cohort_id_column", "subject_id"),
        )

    def as_compact_text(self, max_cols_per_table: int = 30) -> str:
        """
        Compact schema text for prompt injection.
        """
        lines: List[str] = [f"SQL dialect: {self.dialect}"]
        lines.append(f"Cohort table: {self.cohort_table} (id column: {self.cohort_id_column})")
        lines.append("Tables:")
        for t in self.tables:
            cols = t.columns[:max_cols_per_table]
            col_str = ", ".join(cols)
            if len(t.columns) > max_cols_per_table:
                col_str += f", ... (+{len(t.columns) - max_cols_per_table} more)"
            pk = f" [PK={t.primary_key}]" if t.primary_key else ""
            lines.append(f"- {t.name}{pk}: {col_str}")
        return "\n".join(lines)

    def allowed_table_names(self) -> List[str]:
        return [t.name for t in self.tables]

    def allowed_columns_by_table(self) -> Dict[str, List[str]]:
        return {t.name: t.columns for t in self.tables}
