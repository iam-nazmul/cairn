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

    # `ENV` selects the checkpointer/store backend. Nothing outside src/memory/
    # is allowed to branch on this value -- graph code is identical everywhere.
    env: Env = "dev"

    # LLM seam. "fake" is a deterministic scripted model used by the test suite
    # and offline runs; "ollama" talks to a local Ollama server; anything else
    # goes through LangChain's init_chat_model. See src/rag/llm.py.
    llm_provider: str = "fake"
    llm_model: str = ""
    llm_temperature: float = 0.0

    # Ollama. num_ctx must comfortably hold the system prompt (retrieved context
    # + long-term facts) plus the checkpointed history; Ollama's own default of
    # 4096 silently truncates the oldest tokens, which looks exactly like the
    # chatbot "forgetting" earlier turns.
    ollama_base_url: str = "http://localhost:11434"
    ollama_num_ctx: int = 8192

    # Backend locations (consumed in M2/M4).
    sqlite_path: str = "cairn-checkpoints.db"
    database_url: str = ""

    # Retrieval + context budget (SPEC §10 cost control).
    retrieval_top_k: int = 4
    retrieval_min_score: float = 0.05
    max_context_chars: int = 4000


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
