"""Environment-driven settings and the graph's per-invocation runtime context."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

Env = Literal["dev", "local", "prod"]


@dataclass(frozen=True)
class Context:
    """Per-invocation runtime context, declared to the graph via ``context_schema``.

    ``user_id`` travels here and is read as ``runtime.context.user_id``.
    ``thread_id`` does NOT live here -- it stays in
    ``config={"configurable": {"thread_id": ...}}`` where LangGraph's checkpointer
    reads it. See ``.claude/references/langgraph-current-api.md``.
    """

    user_id: str


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


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
