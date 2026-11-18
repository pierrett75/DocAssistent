import os
from functools import lru_cache

from openai import AzureOpenAI, OpenAI

from docassist.config import CHAT_MODEL, EMBEDDING_MODEL

# Rozsah (scope) tokenu pro Azure OpenAI. Říká: "chci token pro Cognitive Services".
AZURE_COGNITIVE_SCOPE = "https://cognitiveservices.azure.com/.default"


def provider() -> str:
    """Vratí llm provider"""
    return os.getenv("LLM_PROVIDER", "openai")
    


def _azure_token_provider():
    """Funkce, která vrací vždy čerstvý Entra ID token (SDK si ho samo obnovuje, platí ~1 h)."""
    from azure.identity import DefaultAzureCredential, get_bearer_token_provider

    return get_bearer_token_provider(DefaultAzureCredential(), AZURE_COGNITIVE_SCOPE)
    


# lru_cache = klient se vytvoří jednou a pak se vrací ten samý (jako _client v secret_store)
@lru_cache
def get_openai_client() -> OpenAI:
    """Klient pro chat.completions, embeddings, beta...parse. Rozhraní je u obou STEJNÉ."""
    if provider() == "azure":
         return AzureOpenAI(
                  azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
                  api_version=os.environ["AZURE_OPENAI_API_VERSION"],
                  azure_ad_token_provider=_azure_token_provider(),
             )
        
    return OpenAI()


def chat_model_name() -> str:
    """Co poslat jako model= ve volání. U Azure je to jméno DEPLOYMENTU, ne modelu."""
    if provider() == "azure":
        return os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT"]
    return CHAT_MODEL

    


def embedding_model_name() -> str:
    """Co poslat jako model= u embeddings. U Azure je to jméno DEPLOYMENTU, ne modelu."""
    if provider() == "azure":
        return os.environ["AZURE_OPENAI_EMBEDDING_DEPLOYMENT"]
    return EMBEDDING_MODEL


def get_chat_llm(temperature: float = 0):
    """LangChain model pro agenta (agent/graph.py)."""
    if provider() == "azure":
        from langchain_openai import AzureChatOpenAI

        return AzureChatOpenAI(
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            api_version=os.environ["AZURE_OPENAI_API_VERSION"],
            azure_deployment=chat_model_name(),
            azure_ad_token_provider=_azure_token_provider(),
            temperature=temperature
        )
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(model=CHAT_MODEL, temperature=temperature)


# ---------------------------------------------------------------------------
# Úkol pro ostatní moduly (uprav je sama):
#   client = OpenAI()               →  client = get_openai_client()
#   model=CHAT_MODEL                →  model=chat_model_name()
#   model=EMBEDDING_MODEL           →  model=embedding_model_name()
#   ChatOpenAI(model=..., ...)      →  get_chat_llm()
# A JUDGE_MODEL (gpt-4o)? Buď druhý deployment "judge", nebo nech soudce na OpenAI.
# Rozhodni a zdůvodni v poznámkách (nápověda: soudce má být silnější model než hodnocený).
#
# Pozor na secret_store: OPENAI_API_KEY je teď POVINNÝ. S LLM_PROVIDER=azure ho nepotřebuješ.
# Jak upravíš REQUIRED_SECRETS, aby to platilo? (Testy z dne 23 ti řeknou, jestli jsi nic nerozbila.)
# ---------------------------------------------------------------------------
