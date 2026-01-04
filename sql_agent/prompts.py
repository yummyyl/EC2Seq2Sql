from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PromptPack:
    system: str
    human: str


DEFAULT_SYSTEM_PROMPT = (
    "You are an expert in generating SQL queries for clinical trials. "
    "Your task is to create SQL queries based on lightweight, structured semantic "
    "representations to filter patients who meet specific inclusion and exclusion criteria "
    "from EHR databases. Each query should comprehensively address the following seven core "
    "fields: condition, procedure, observation, laboratory, drug, age, and gender. "
    "It is essential to ensure that the generated SQL query is syntactically correct, "
    "logically sound, and accurately reflects the conditions and constraints outlined in the "
    "eligibility criteria."
)

DEFAULT_HUMAN_PROMPT = (
    "Using the provided lightweight and structured semantic representation, generate an SQL query "
    "that parses and incorporates information from the following fields: condition, procedure, "
    "observation, laboratory, drug, age, and gender. Additionally, ensure that time expressions "
    "and logical operators are correctly converted into syntactically valid database expressions "
    "so that the query accurately reflects the time constraints and logical relationships specified "
    "in the eligibility criteria."
)


def build_prompts(schema_context: str, structured_snippet: str) -> PromptPack:
    """
    Build the hierarchical prompts (system + human) described in the paper.
    The schema_context is injected as "database context" in the system prompt.
    """
    system = (
        DEFAULT_SYSTEM_PROMPT
        + "\n\n"
        + "Database context (schema summary):\n"
        + schema_context.strip()
    )

    human = (
        DEFAULT_HUMAN_PROMPT
        + "\n\n"
        + "Structured semantic representation (input):\n"
        + structured_snippet.strip()
        + "\n\n"
        + "Return ONLY the SQL query. Do not include explanations."
    )
    return PromptPack(system=system, human=human)
