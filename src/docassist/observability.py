"""
Víkend týdne 3, soubor 1/3: tracing přes Langfuse
Kam: src/docassist/observability.py   (nový soubor)

Instalace:      pip install langfuse          (+ přidej "langfuse" do dependencies v pyproject.toml, jinak spadne CI)
Účet:           https://cloud.langfuse.com → New project → Settings → API Keys
Do .env:        LANGFUSE_PUBLIC_KEY=pk-lf-...
                LANGFUSE_SECRET_KEY=sk-lf-...
                LANGFUSE_BASE_URL=https://cloud.langfuse.com      (EU region, data zůstanou v EU)

Logy vs. trace vs. metriky:
    log      = jeden řádek "něco se stalo"             (POST /ask 200 1834 ms)
    trace    = strom všeho, co se dělo při JEDNOM dotazu (router → agent → search_documents → agent → odpověď)
               každý uzel = span: vstup, výstup, čas, tokeny, cena
    metrika  = číslo v čase                             (průměrná latence, cena za den, pass rate)
Log ti řekne, ŽE byl /ask pomalý. Trace ti řekne, KTERÝ krok a PROČ (3 kola nástrojů, router 1,2 s ...).

Jak se to napojí: LangChain/LangGraph posílá události (začal uzel, LLM vrátil tokeny, nástroj skončil)
všem "callbackům" v configu. CallbackHandler od Langfuse je posbírá a pošle na server NA POZADÍ
(dávkově, neblokuje odpověď). Kód agenta se NEMĚNÍ: stačí přidat callback do configu.
"""

import os

from langfuse.langchain import CallbackHandler

import docassist.config  # noqa: F401  config.py volá load_dotenv() → klíče z .env jsou v os.environ


def tracing_enabled() -> bool:
    """Tracing jen když jsou klíče. Testy a CI klíče nemají → žádné síťové volání, žádné chyby v logu."""
    return bool(os.getenv("LANGFUSE_PUBLIC_KEY")) and bool(os.getenv("LANGFUSE_SECRET_KEY"))


def tracing_config(session_id: str, user_id: str, tags: list[str]) -> dict:
    """Vrátí část LangGraph configu pro tracing, nebo {} když je tracing vypnutý.

    Použití:  config = {"configurable": {...}, "recursion_limit": ..., **tracing_config(...)}
    **{} nepřidá nic, takže kód volající tuhle funkci nemusí řešit, jestli je tracing zapnutý.

    session_id  = thread_id → Langfuse seskupí všechny otázky jedné konverzace (záložka Sessions)
    user_id     = ROLE, ne e-mail ani jméno. Do tracingu posíláš data třetí straně: žádné osobní údaje,
                  které nepotřebuješ (GDPR). Na filtrování "co se ptá HR vs. zaměstnanci" role stačí.
    tags        = např. ["api"] nebo ["eval", "2026-11-14_0930"] → v Langfuse filtruješ produkci vs. evaluaci
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
