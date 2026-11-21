import os

from langfuse.langchain import CallbackHandler

import docassist.config  # noqa: F401  config.py volá load_dotenv() → klíče z .env jsou v os.environ


def tracing_enabled() -> bool:
    """Tracing jen když jsou klíče. Testy a CI klíče nemají → žádné síťové volání, žádné chyby v logu."""
    return bool(os.getenv("LANGFUSE_PUBLIC_KEY")) and bool(os.getenv("LANGFUSE_SECRET_KEY"))


def tracing_config(session_id: str, user_id: str, tags: list[str]) -> dict:
    """Vrátí část LangGraph configu pro tracing, nebo {} když je tracing vypnutý.
    """
    if not tracing_enabled():
        return {}
    return {
        "callbacks": [CallbackHandler()],
        "metadata": {
            "langfuse_session_id": session_id,
            "langfuse_user_id": user_id,
            "langfuse_tags": tags,
        },
    }


def last_trace_id(config: dict) -> str | None:
    """ID trace z posledního běhu (pro /feedback a pro skóre z evaluace). None, když tracing neběží."""
    callbacks = config.get("callbacks") or []
    return callbacks[0].last_trace_id if callbacks else None
