from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class LLMConfig:
    provider: str = "langchain_openai"  # or "dummy"
    model: str = "gpt-4o-mini"          # you can set to "gpt-4" for paper reproduction
    temperature: float = 0.0
    max_tokens: int = 512
    timeout_s: int = 120


class BaseLLM:
    def generate(self, system: str, user: str) -> str:
        raise NotImplementedError


class DummyLLM(BaseLLM):
    """
    A fallback LLM for CI / no-API-key environments.
    It returns a minimal valid SQL.
    """

    def generate(self, system: str, user: str) -> str:
        _ = system
        _ = user
        return "SELECT DISTINCT subject_id FROM patients;"


class LangChainOpenAIChatLLM(BaseLLM):
    """
    Uses LangChain's ChatOpenAI wrapper.
    Requires: langchain, langchain-openai
    Env:
      - OPENAI_API_KEY
      - (optional) OPENAI_BASE_URL
    """

    def __init__(self, cfg: LLMConfig):
        self.cfg = cfg

        try:
            from langchain_openai import ChatOpenAI
            from langchain_core.messages import SystemMessage, HumanMessage
        except Exception as e:
            raise RuntimeError(
                "LangChain OpenAI backend not available.\n"
                "Install: pip install langchain langchain-openai\n"
                f"Original error: {e}"
            )

        self._SystemMessage = SystemMessage
        self._HumanMessage = HumanMessage

        base_url = os.getenv("OPENAI_BASE_URL", None)
        self.client = ChatOpenAI(
            model=cfg.model,
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens,
            timeout=cfg.timeout_s,
            base_url=base_url,
        )

    def generate(self, system: str, user: str) -> str:
        msgs = [self._SystemMessage(content=system), self._HumanMessage(content=user)]
        resp = self.client.invoke(msgs)
        return getattr(resp, "content", str(resp)).strip()


def build_llm(cfg: Optional[LLMConfig] = None) -> BaseLLM:
    cfg = cfg or LLMConfig()
    if cfg.provider == "dummy":
        return DummyLLM()
    if cfg.provider == "langchain_openai":
        return LangChainOpenAIChatLLM(cfg)
    raise ValueError(f"Unknown provider: {cfg.provider}")
