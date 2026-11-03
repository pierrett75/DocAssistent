
import os
 
import tiktoken
from dotenv import load_dotenv
from openai import OpenAI
 
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
MODEL = "gpt-4o-mini"  
 
 
# --- 1. Tokeny: kolik "stojí" věta ----------------------------------------
def count_tokens(text: str) -> int:
    enc = tiktoken.get_encoding("o200k_base")  
    return len(enc.encode(text))
 
 
cz = "Jaké jsou hlavní povinnosti zaměstnance podle této směrnice?"
en = "What are the main obligations of an employee under this policy?"
print("=== Tokeny ===")
print(f"CZ: {count_tokens(cz)} tokenů | {cz}")
print(f"EN: {count_tokens(en)} tokenů | {en}\n")
 
 
# --- 2. Role zpráv + temperature ------------------------------------------
messages = [
    {
        "role": "system",
        "content": "Jsi stručný asistent. Odpovídej jednou větou česky.",
    },
    {"role": "user", "content": """Odpovídej pouze z dodaného kontextu.
Rozlišuj pojmy: dovolená, sick days a další druhy volna nejsou totéž.
Nejdřív vypiš fakta z kontextu, která se týkají otázky, a teprve pak odpověz.
Pokud otázka závisí na informaci, která v kontextu chybí (např. zákonné minimum), řekni to."""},
]
 
for temperature in (0.0, 1.2):
    print(f"=== temperature = {temperature} ===")
    for i in range(3):
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=temperature,
        )
        answer = response.choices[0].message.content
        usage = response.usage
        print(f"{i + 1}. {answer}  [vstup {usage.prompt_tokens} / výstup {usage.completion_tokens} tokenů]")
    print()
 
 
# --- 3. Halucinace: model bez kontextu vs. s kontextem --------------------
question = "Kolik dní dovolené navíc mají zaměstnanci firmy Acme Brno s.r.o.?"
context = "Směrnice Acme Brno s.r.o.: Zaměstnanci mají nárok na 5 týdnů dovolené a 3 dny volna navíc (sick days)."
 
print("=== Bez kontextu ===")
r = client.chat.completions.create(
    model=MODEL,
    temperature=0,
    messages=[{"role": "user", "content": question}],
)
print(r.choices[0].message.content, "\n")
 
print("=== S kontextem a pravidlem 'jen z kontextu' ===")
r = client.chat.completions.create(
    model=MODEL,
    temperature=0,
    messages=[
        {
            "role": "system",
            "content": "Odpovídej pouze z dodaného kontextu. Pokud odpověď v kontextu není, napiš: 'V dokumentech to nenacházím.'",
        },
        {"role": "user", "content": f"Kontext:\n{context}\n\nOtázka: {question}"},
    ],
)
print(r.choices[0].message.content)