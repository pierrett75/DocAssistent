"""Testy tracingu (Langfuse) a request ID.

Spuštění: python -m pytest -v -k "tracing or request_id"
"""

from docassist.observability import tracing_config, tracing_enabled


def test_tracing_bez_klicu_je_vypnuty(monkeypatch):
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)   # i kdyby je měl tvůj lokální .env
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)

    true_or_false = tracing_enabled()
    assert true_or_false is False
    assert tracing_config("t", "hr", ["api"]) == {}


def test_kazda_odpoved_ma_request_id(client):
    r = client.get("/health")
    assert "x-request-id" in r.headers                      # hlavičky jsou bez ohledu na velikost písmen
    r = client.get("/health", headers={"X-Request-ID": "abc123"})
    assert r.headers["x-request-id"] == "abc123"            # poslané ID se vrátí beze změny
            # poslané ID se vrátí beze změny