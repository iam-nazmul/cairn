"""Environment-driven settings and the graph's per-invocation runtime context."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

Env = Literal["dev", "local", "prod"]

# "chat" retrieves once. "agent" may search repeatedly, refining the query from
# what came back. Both must cite: the mode changes how evidence is gathered, not
# whether an answer needs any.
Mode = Literal["chat", "agent"]


@dataclass(frozen=True)
class Context:
    """Per-invocation runtime context, declared to the graph via ``context_schema``.

    ``user_id`` travels here and is read as ``runtime.context.user_id``.
    ``thread_id`` does NOT live here -- it stays in
    ``config={"configurable": {"thread_id": ...}}`` where LangGraph's checkpointer
    reads it. See ``.claude/references/langgraph-current-api.md``.
    """

    user_id: str
    # Per turn, not per thread: one conversation can mix both.
    mode: Mode = "chat"


class Settings(BaseSettings):
    """Settings resolved from the environment / ``.env``."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Selects the backend. Nothing outside src/memory/ may branch on it.
    env: Env = "dev"

    # "fake" = scripted model for tests; "ollama" = local server; else init_chat_model.
    llm_provider: str = "fake"
    llm_model: str = ""
    llm_temperature: float = 0.0

    # num_ctx must exceed prompt + history; Ollama's 4096 default silently drops
    # the oldest tokens, which looks exactly like forgetting.
    ollama_base_url: str = "http://localhost:11434"
    ollama_num_ctx: int = 8192

    # Backend locations (consumed in M2/M4).
    sqlite_path: str = "cairn-checkpoints.db"
    database_url: str = ""

    # SPEC §11: "rules" is deterministic; "llm" trades precision for recall.
    memory_extraction: Literal["rules", "llm", "off"] = "rules"
    max_long_term_facts: int = 50

    # Applies to the "cairn" logger only, so it pulls in no third-party noise.
    log_level: str = "INFO"

    # Retrieval + context budget (SPEC §10 cost control).
    retrieval_top_k: int = 4
    retrieval_min_score: float = 0.05
    max_context_chars: int = 4000
    # Trimmed before the LLM call only; the checkpoint keeps every turn.
    max_history_tokens: int = 1500

    # Agent mode. Each extra search costs a model call to rewrite the query, so
    # the budget is the cost ceiling for a turn (SPEC §10) -- it counts the first
    # search too, so 1 makes agent mode behave like chat.
    agent_max_searches: int = 3
    # Stop early once retrieval is already this good; refining past it spends a
    # model call to re-find what has been found.
    agent_good_score: float = 0.5


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
