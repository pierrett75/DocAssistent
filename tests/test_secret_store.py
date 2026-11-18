"""
Den 23, soubor 2/3: testy pro secret_store
Kam: tests/test_secret_store.py

Žádné volání Azure: Key Vault nahradíme falešným klientem (stejně jako FakeAgent v den 17).
Spuštění: python -m pytest tests/test_secret_store.py -v
"""

import logging
import os

import pytest

from docassist import secret_store
from docassist.secret_store import MissingSecretError, get_secret, load_secrets, to_vault_name


class _Secret:
    def __init__(self, value):
        self.value = value


class FakeVault:
    """Chová se jako SecretClient: get_secret(name).value, chybějící → ResourceNotFoundError."""

    def __init__(self, data: dict[str, str]):
        self.data = data
        self.calls: list[str] = []

    def get_secret(self, name):
        from azure.core.exceptions import ResourceNotFoundError

        self.calls.append(name)
        if name not in self.data:
            raise ResourceNotFoundError(f"{name} not found")
        return _Secret(self.data[name])


@pytest.fixture
def clean_env(monkeypatch):
    """Smaže klíče z prostředí, aby test nezávisel na tvém .env."""
    for name in secret_store.REQUIRED_SECRETS + secret_store.OPTIONAL_SECRETS + ["KEY_VAULT_URL"]:
        # setenv + delenv: monkeypatch si tak zapamatuje původní stav i u proměnné, která neexistovala,
        # a po testu smaže i to, co do os.environ zapsal load_secrets(). Samotný delenv(raising=False)
        # by chybějící proměnnou nezaznamenal a hodnota z testu by "prosákla" do dalších testů.
        monkeypatch.setenv(name, "placeholder")
        monkeypatch.delenv(name)
    monkeypatch.setattr(secret_store, "_client", None)


def use_vault(monkeypatch, data):
    fake = FakeVault(data)
    monkeypatch.setattr(secret_store, "_get_vault_client", lambda: fake)
    return fake


def test_vault_name():
    assert to_vault_name("OPENAI_API_KEY") == "openai-api-key"
    assert to_vault_name("LANGFUSE_SECRET_KEY") == "langfuse-secret-key"


def test_env_wins_over_vault(clean_env, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "z-env")
    fake = use_vault(monkeypatch, {"openai-api-key": "z-vaultu"})

    assert get_secret("OPENAI_API_KEY") == "z-env"
    assert fake.calls == []


def test_empty_env_falls_back_to_vault(clean_env, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "")
    use_vault(monkeypatch, {"openai-api-key": "z-vaultu"})

    assert get_secret("OPENAI_API_KEY") == "z-vaultu"


def test_falls_back_to_vault(clean_env, monkeypatch):
    fake = use_vault(monkeypatch, {"openai-api-key": "z-vaultu"})

    assert get_secret("OPENAI_API_KEY") == "z-vaultu"
    assert fake.calls == ["openai-api-key"]


def test_missing_returns_none(clean_env, monkeypatch):
    use_vault(monkeypatch, {})

    assert get_secret("OPENAI_API_KEY") is None


def test_no_vault_configured_returns_none(clean_env):
    # Bez KEY_VAULT_URL se klient nevytvoří a get_secret vrátí None (žádné volání Azure).
    assert get_secret("OPENAI_API_KEY") is None


def test_other_vault_errors_are_not_swallowed(clean_env, monkeypatch):
    from azure.core.exceptions import HttpResponseError

    class ForbiddenVault:
        def get_secret(self, name):
            raise HttpResponseError("403 Forbidden")

    monkeypatch.setattr(secret_store, "_get_vault_client", lambda: ForbiddenVault())

    with pytest.raises(HttpResponseError):
        get_secret("OPENAI_API_KEY")


def test_load_secrets_fails_fast_without_required(clean_env, monkeypatch):
    use_vault(monkeypatch, {})

    with pytest.raises(MissingSecretError, match="OPENAI_API_KEY"):
        load_secrets()


def test_load_secrets_puts_value_into_environ(clean_env, monkeypatch):
    use_vault(monkeypatch, {"openai-api-key": "sk-test"})

    load_secrets()  # chybějící Langfuse klíče nesmí vyhodit výjimku

    assert os.environ["OPENAI_API_KEY"] == "sk-test"
    assert "LANGFUSE_PUBLIC_KEY" not in os.environ
    assert "LANGFUSE_SECRET_KEY" not in os.environ


def test_load_secrets_warns_about_missing_optional(clean_env, monkeypatch, caplog):
    use_vault(monkeypatch, {"openai-api-key": "sk-test"})

    with caplog.at_level(logging.WARNING, logger=secret_store.__name__):
        load_secrets()

    assert "LANGFUSE_PUBLIC_KEY" in caplog.text
    assert "LANGFUSE_SECRET_KEY" in caplog.text


def test_secret_value_never_logged(clean_env, monkeypatch, caplog):
    use_vault(monkeypatch, {"openai-api-key": "sk-tajne-123"})

    # DEBUG, ať zachytíme úplně všechno, co modul zaloguje
    with caplog.at_level(logging.DEBUG, logger=secret_store.__name__):
        load_secrets()

    assert "sk-tajne-123" not in caplog.text
