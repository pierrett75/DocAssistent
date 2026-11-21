import logging
import os

from azure.core.exceptions import ResourceNotFoundError

logger = logging.getLogger(__name__)

# Klíče, bez kterých aplikace nemá smysl startovat.
REQUIRED_SECRETS = ["OPENAI_API_KEY"]
# Klíče, které jsou volitelné (bez Langfuse aplikace jede, jen netrasuje).
OPTIONAL_SECRETS = ["LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY"]


class MissingSecretError(RuntimeError):
    """Povinný klíč není ani v prostředí, ani v Key Vaultu."""


# ---------------------------------------------------------------------------
# 1. Jména: proměnná prostředí → jméno v Key Vaultu
# ---------------------------------------------------------------------------
# Key Vault povoluje ve jménu tajemství jen písmena, číslice a POMLČKY (žádná podtržítka).
# Konvence: OPENAI_API_KEY → openai-api-key
def to_vault_name(env_name: str) -> str:
    return env_name.lower().replace("_","-")


# ---------------------------------------------------------------------------
# 2. Klient Key Vaultu (jen když je potřeba)
# ---------------------------------------------------------------------------
_client = None


def _get_vault_client():
    """Vytvoří SecretClient jednou a pak ho vrací znovu (lazy singleton)."""
    global _client
    if _client is not None:
        return _client

    vault_url = os.getenv("KEY_VAULT_URL")  # např. https://kv-docassist-pierr.vault.azure.net/
    if not vault_url:
        return None

    # Importy uvnitř funkce: kdo Key Vault nepoužívá (testy, CI), nemusí mít azure balíčky.
    from azure.identity import DefaultAzureCredential
    from azure.keyvault.secrets import SecretClient

    # DefaultAzureCredential zkouší identity v pořadí, mimo jiné:
    #   proměnné prostředí (service principal) → MANAGED IDENTITY (v Azure) → přihlášení z `az login` (lokálně)
    # Stejný kód tedy funguje lokálně i v Container Apps, mění se jen KDO se přihlašuje.
    credential = DefaultAzureCredential()
    _client = SecretClient(vault_url=vault_url, credential=credential)
    
    return _client


# ---------------------------------------------------------------------------
# 3. Jeden klíč
# ---------------------------------------------------------------------------
def get_secret(env_name: str) -> str | None:
    """Vrátí hodnotu klíče, nebo None, když není nikde."""
    
    value = os.environ.get(env_name)
    if value:
        return value

    client = _get_vault_client()
    if client is None:
        return None

    try:
        return client.get_secret(to_vault_name(env_name)).value
    except ResourceNotFoundError:
        return None

def _secret_lists() -> tuple[list[str], list[str]]:
    """(povinné, volitelné) podle LLM_PROVIDER. Čte se až při volání, ne při importu."""
    if os.getenv("LLM_PROVIDER", "openai") == "azure":
        # Chat a embeddingy jdou přes Entra ID (bez klíče), OPENAI_API_KEY potřebuje jen soudce v evalu.
        return [], ["OPENAI_API_KEY"] + OPTIONAL_SECRETS
    return REQUIRED_SECRETS, OPTIONAL_SECRETS

# ---------------------------------------------------------------------------
# 4. Načtení všeho při startu
# ---------------------------------------------------------------------------
def load_secrets() -> None:
    """Zavolej jednou při startu (v config.py hned po load_dotenv)."""
    required, optional = _secret_lists()
    for name in required + optional:
        value = get_secret(name)
        
        if not value:
            if name in required:
                raise MissingSecretError(f"Chybí {name}: není v prostředí ani v Key Vaultu.")
            logger.warning("Volitelný klíč %s chybí, pokračuju bez něj.", name)
            continue

        os.environ[name]=value
