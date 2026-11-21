from pypdf import PdfReader

from docassist.config import PDF_PATH
from docassist.llm import chat_model_name, get_openai_client

client = get_openai_client()

# Dlouhý statický kontext: celý text směrnice (~ několik tisíc tokenů), aby prompt měl > 1024 tokenů
DIRECTIVE = "\n".join(page.extract_text() or "" for page in PdfReader(PDF_PATH).pages)
SYSTEM = "Jsi HR asistent. Odpovídej jen podle směrnice níže, stručně, česky.\n\n" + DIRECTIVE


def call(messages: list[dict], label: str) -> None:
    response = client.chat.completions.create(model=chat_model_name(), messages=messages, max_tokens=50)
    usage = response.usage
    # prompt_tokens_details může být None (a cached_tokens taky) → v obou případech 0
    details = usage.prompt_tokens_details
    cached = (details.cached_tokens or 0) if details else 0
    print(f"{label}: prompt_tokens={usage.prompt_tokens} cached_tokens={cached}")


# Volání 1: statický kontext NA ZAČÁTKU, otázka na konci
call([{"role": "system", "content": SYSTEM},
      {"role": "user", "content": "Kolik mám sick days?"}], "1 (studená cache)")

# Volání 2: STEJNÝ system, jiná otázka.
# Očekávání: cached_tokens > 0 (násobek 128): začátek promptu je stejný jako u volání 1.
call([{"role": "system", "content": SYSTEM},
      {"role": "user", "content": "Kolik dní dovolené mám?"}], "2 (stejný začátek)")

# Volání 3: otázka NA ZAČÁTKU, směrnice až za ní.
# Očekávání: cached_tokens = 0. Cache funguje jen na shodný ZAČÁTEK (prefix) promptu.
# Tady se prompt liší hned od prvních tokenů, takže se nic znovu nepoužije,
# i když je text směrnice úplně stejný. Pravidlo: statické věci dopředu, proměnné na konec.
call([{"role": "user", "content": "Kolik mám sick days?\n\n" + SYSTEM}], "3 (otázka na začátku)")
