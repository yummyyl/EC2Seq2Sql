from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from .llm import LLMConfig, build_llm
from .postprocess import basic_safety_checks, postprocess_sql
from .prompts import build_prompts
from .schema import DatabaseSchema


@dataclass
class SQLAgentConfig:
    schema_path: str = "data/schemas/demo_schema.json"
    provider: str = "langchain_openai"   
    model: str = "gpt-4o-mini"          
    temperature: float = 0.0
    max_tokens: int = 512
    timeout_s: int = 120
    max_retries: int = 2


class SQLAgent:
    """
    Stage-2 SQL generation agent:
      structured snippet (7-domain patterns) + schema context -> SQL
    As described in the paper (system + human prompt).:contentReference[oaicite:5]{index=5}
    """

    def __init__(self, cfg: Optional[SQLAgentConfig] = None):
        self.cfg = cfg or SQLAgentConfig()
        self.schema = DatabaseSchema.from_json(self.cfg.schema_path)

        llm_cfg = LLMConfig(
            provider=self.cfg.provider,
            model=self.cfg.model,
            temperature=self.cfg.temperature,
            max_tokens=self.cfg.max_tokens,
            timeout_s=self.cfg.timeout_s,
        )
        self.llm = build_llm(llm_cfg)

    def generate_sql(self, structured_snippet: str) -> Tuple[str, Dict[str, Any]]:
        schema_ctx = self.schema.as_compact_text()
        prompts = build_prompts(schema_context=schema_ctx, structured_snippet=structured_snippet)

        last_err = ""
        for attempt in range(self.cfg.max_retries + 1):
            raw = self.llm.generate(system=prompts.system, user=prompts.human)
            sql = postprocess_sql(raw)

            ok, msg = basic_safety_checks(sql)
            if ok:
                meta = {
                    "attempt": attempt,
                    "provider": self.cfg.provider,
                    "model": self.cfg.model,
                }
                return sql, meta

            last_err = msg

        raise RuntimeError(f"Failed to generate safe SQL after retries. Last error: {last_err}")


def load_snippet_from_example(example: Dict[str, Any]) -> str:
    """
    Your Stage-1 outputs are stored in the field 'snippet' (DSL/pattern).
    This helper extracts it consistently.
    """
    if "snippet" not in example:
        raise KeyError("Input example must contain 'snippet'.")
    return str(example["snippet"])
