from email.mime import message
import json
from langchain_core import messages

from openai import OpenAI

from docassist.config import CHAT_MODEL, DATA_DIR, TOP_K
from docassist.db.vector_store import get_collection, retrieve
from docassist.extract.company_rule import extract_policy, VacationPolicy
from docassist.ingest.pdf_loader import load_pdf
from docassist.rag.pipeline import format_context

client = OpenAI()
MAX_STEPS = 5   # pojistka: model nesmí volat nástroje donekonečna


# ---------------------------------------------------------------------------
# 1. Python funkce = co nástroj doopravdy dělá
# ---------------------------------------------------------------------------

def search_documents(query: str) -> str:
    """Najde pasáže ve směrnicích a vrátí je očíslované se stranou."""
    results = retrieve(get_collection(), query, k=TOP_K)
    return format_context(results)


def extract_vacation_policy() -> str:
    """Vytáhne strukturovaná data o dovolené ze směrnice a dopočítá dny navíc."""
    pages = load_pdf(DATA_DIR / "smernice_dovolena.pdf")
    full_text = "\n\n".join(f"[strana {p['page']}]\n{p['text']}" for p in pages)
    policy = extract_policy(full_text)
    data = policy.model_dump()
    data["extra_vacation_days"] = policy.extra_vacation_days()   # výpočet v kódu (den 8)
    return json.dumps(data, ensure_ascii=False)


# ---------------------------------------------------------------------------
# 2. Popis nástrojů pro model (JSON schéma). Model vidí JEN tohle, ne tvůj kód.
# ---------------------------------------------------------------------------

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_documents",
            "description": (
                "Vyhledá v interních směrnicích pasáže k otázce. Použij pro otázky na "
                "pravidla, postupy a podmínky. Vrací text s čísly zdrojů [n] a stranou."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Hledaný dotaz česky, formulovaný slovy, jaká mohou být v dokumentu.",
                    }
                },
                "required": ["query"],
                "additionalProperties": False,
            },
            "strict": True,
        },
    },
    {
        "type": "function",
        "function": {
            "name": "extract_vacation_policy",
            "description": (
                "Vytáhne ze směrnice o dovolené všechna čísla najednou a dopočítá dny navíc. "
                "Použij JEN pro přehled více čísel nebo pro výpočet, např. "
                "kolik dní dovolené mám navíc nad zákonné minimum' nebo 'dej mi přehled čísel o dovolené'. "
                "Na otázku na jedno pravidlo nebo podmínku (např. sick days, převod dovolené) použij search_documents."
),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": False,
            },
            "strict": True,
        },
    }
]

SYSTEM_PROMPT = """Jsi asistent DocAssist pro zaměstnance. Na otázky o firemních pravidlech
odpovídej jen podle výsledků nástrojů a cituj zdroje [n] se stranou.
Na každou otázku kromě pozdravu nejdřív zavolej nástroj.
Když nástroje nic relevantního nenajdou, řekni, že odpověď v dokumentech není.
U výsledku extract_vacation_policy necituj [n], uveď jen označení dokumentu.
Uživatele oslovuj zdvořile a stručně a neutrálně"""


# ---------------------------------------------------------------------------
# 3. Spuštění nástroje podle jména
# ---------------------------------------------------------------------------

def run_tool(name: str, arguments: dict) -> str:
    """Zavolá Python funkci podle jména, které vybral model."""
   
    available_functions = {
        "extract_vacation_policy": extract_vacation_policy,
        "search_documents": search_documents,
    }  

    if name not in available_functions:
        return f"Neznámý nástroj: {name}"
    return available_functions[name](**arguments)

   


# ---------------------------------------------------------------------------
# 4. Smyčka agenta: model → nástroj → model → ... → odpověď
# ---------------------------------------------------------------------------

def ask(question: str) -> tuple[str, list[str]]:
    """Vrátí (odpověď, seznam zavolaných nástrojů)."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    used_tools = []

    for step in range(MAX_STEPS):
        response = client.chat.completions.create(
            model=CHAT_MODEL, messages=messages, tools=TOOLS, temperature=0
        )
        message = response.choices[0].message
        

       
        if message.tool_calls is None or len(message.tool_calls) == 0:
            return message.content, used_tools

       
        messages.append(message)

        for tool_call in message.tool_calls:
            name = tool_call.function.name
            arguments = json.loads(tool_call.function.arguments)
            result = run_tool(name, arguments)
            used_tools.append(f"{name}({arguments})")
            messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": result})


    return "Překročen limit kroků, zkuste otázku zjednodušit.", used_tools


if __name__ == "__main__":
    questions = [
        "Ahoj, co umíš?",                                              # žádný nástroj
        "Kolik mám sick days?",                                        # search
        "Můžu si vzít sick day v pátek před dovolenou?",               # search
        "Kolik pracovních dní dovolené mám navíc nad zákonné minimum?",  # extrakce
        "Dej mi přehled všech čísel o dovolené.",                      # extrakce
        "Jaké je heslo na wifi?",                                      # search → nic
    ]
    for q in questions:
        answer_text, tools = ask(q)
        print(f"Otázka: {q}")
        print(f"Nástroje: {tools or 'žádné'}")
        print(f"Odpověď: {answer_text}\n")

    # TODO 4: Pozoruj sloupec "Nástroje":
    #         - Vybral model u každé otázky ten nástroj, který bys vybrala ty?
    #         - Jaký query poslal do search_documents? Přepsal otázku do jazyka dokumentu
    #           (třeba "výměra")? To je query rewriting zadarmo (den 6).
    #         - Když vybere špatně, uprav DESCRIPTION nástroje, ne kód, a spusť znovu.

    # Otázky k zamyšlení (odpověz si do poznámek):
    # a) Kdo nástroj doopravdy spouští: model, nebo tvůj kód? Co model vrací?
    # b) Proč je v ask() MAX_STEPS? Co by se mohlo stát bez něj?
    # c) Proč u "kolik dní navíc" počítá kód (extra_vacation_days), a ne model?
    # d) Čím se tohle liší od RAG ze dne 6, kde se vyhledávalo vždycky?
