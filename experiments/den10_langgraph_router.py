import json
import time 
from typing import Literal, TypedDict

from langgraph.graph import END, START, StateGraph
from openai import OpenAI
from pydantic import BaseModel, Field

from docassist.config import CHAT_MODEL, DATA_DIR, TOP_K
from docassist.db.vector_store import get_collection, retrieve
from docassist.extract.company_rule import extract_policy
from docassist.ingest.pdf_loader import load_pdf
from docassist.rag.pipeline import format_context

client = OpenAI()


# ---------------------------------------------------------------------------
# 1. STAV = sdílený slovník, který putuje grafem. Každý uzel z něj čte
#    a vrací jen ty klíče, které změnil. LangGraph je do stavu sám doplní.
# ---------------------------------------------------------------------------

class State(TypedDict, total=False):   # total=False: klíče se doplňují postupně
    question: str      # vstup od uživatele
    route: str         # "search" | "extract" | "chitchat", vyplní router
    context: str       # text pro generování: pasáže [n] nebo JSON z extrakce
    answer: str        # konečná odpověď


# ---------------------------------------------------------------------------
# 2. ROUTER = klasifikace přes structured outputs (den 8)
# ---------------------------------------------------------------------------

ROUTER_PROMPT = """Zařaď otázku zaměstnance do jedné kategorie:
- search: otázka na konkrétní pravidlo, postup nebo podmínku ve firemních směrnicích
  (i když si myslíš, že odpověď v dokumentech není)
- extract: přehled všech čísel o dovolené nebo výpočet (např. kolik dní navíc nad zákonné minimum), 
- chitchat: pouzw pozdrav a poděkování a otázka "co umíš". Všechno ostatní je search
"""


class RouteDecision(BaseModel):
    route: Literal["search", "extract", "chitchat"]
    reason: str = Field(description="Jedna věta, proč tato kategorie")


def router_node(state: State) -> dict:
    """Zařadí otázku. Vrací {"route": ...}."""
    
    completion = client.chat.completions.parse(
        model=CHAT_MODEL,
        temperature=0,
        messages=[
            {"role": "system", "content": ROUTER_PROMPT},
            {"role": "user", "content": state["question"]},
        ],
        response_format=RouteDecision,
    )
    decision = completion.choices[0].message.parsed
    print(f"  router: {decision.route} ({decision.reason})")
    return {"route": decision.route}


# ---------------------------------------------------------------------------
# 3. UZLY = obyčejné Python funkce: dostanou stav, vrátí změny
# ---------------------------------------------------------------------------

def search_node(state: State) -> dict:
    results = retrieve(get_collection(), state["question"], k=TOP_K)
    context = format_context(results)        # ← uložím si text do proměnné
    return {"context": format_context(results)}


def extract_node(state: State) -> dict:
    pages = load_pdf(DATA_DIR / "smernice_dovolena.pdf")
    full_text = "\n\n".join(f"[strana {p['page']}]\n{p['text']}" for p in pages)
    policy = extract_policy(full_text)
    data = policy.model_dump()
    data["extra_vacation_days"] = policy.extra_vacation_days()
    return {"context": "Data ze směrnice (JSON):\n" + json.dumps(data, ensure_ascii=False)}


def chitchat_node(state: State) -> dict:
    # Bez LLM: na pozdrav nepotřebuju platit za volání API.
    return {"answer": "Ahoj, jsem DocAssist. Odpovídám na otázky o firemních směrnicích, "
                      "třeba o dovolené nebo sick days, a uvádím zdroje."}


GENERATE_PROMPT = """Jsi asistent DocAssist. Odpovídej jen podle kontextu.
Když kontext neříká odpověď doslova, ale obsahuje obecné pravidlo, které na otázku dopadá,
odpověz podle něj a řekni, ze kterého pravidla vycházíš.
Rozlišuj zákaz a podmínku: když pravidlo platí "bez reálného důvodu", vysvětli, za jakých podmínek to jde.
Pokud jsou v kontextu pasáže [n], cituj je jako [n]. U JSON dat necituj [n], uveď označení dokumentu.
Oslovuj neutrálně, bez rodu.
Jen když kontext s otázkou vůbec nesouvisí, odpověz: "Odpověď v dokumentech není." """


def generate_node(state: State) -> dict:
    response = client.chat.completions.create(
        model=CHAT_MODEL,
        temperature=0,
        messages=[
            {"role": "system", "content": GENERATE_PROMPT},
            {"role": "user", "content": f"<kontext>\n{state['context']}\n</kontext>\n\nOtázka: {state['question']}"},
        ],
    )
    return {"answer": response.choices[0].message.content}


# ---------------------------------------------------------------------------
# 4. PODMÍNĚNÁ HRANA = funkce, která podle stavu řekne, kam dál
# ---------------------------------------------------------------------------

def pick_route(state: State) -> Literal["search", "extract", "chitchat"]:
    """Vrátí jméno dalšího uzlu podle route, kterou vybral router."""
    
    return state["route"]


# ---------------------------------------------------------------------------
# 5. SESTAVENÍ GRAFU
# ---------------------------------------------------------------------------

def build_graph():
    graph = StateGraph(State)
    graph.add_node("router", router_node)
    graph.add_node("search", search_node)
    graph.add_node("extract", extract_node)
    graph.add_node("chitchat", chitchat_node)
    graph.add_node("generate", generate_node)

 
    graph.add_edge(START, "router")
    graph.add_conditional_edges("router", pick_route)
    graph.add_edge("search", "generate")
    graph.add_edge("extract", "generate") 
    graph.add_edge("generate", END)
    graph.add_edge("chitchat", END)

    return graph.compile()


if __name__ == "__main__":
    app = build_graph()

    # Graf jako text v Mermaid (vlož na https://mermaid.live a uvidíš obrázek)
    print(app.get_graph().draw_mermaid())

    questions = [
        "Ahoj, co umíš?",
        "Kolik mám sick days?",
        "Můžu si vzít sick day v pátek před dovolenou?",
        "Kolik pracovních dní dovolené mám navíc nad zákonné minimum?",
        "Dej mi přehled všech čísel o dovolené.",
        "Jaké je heslo na wifi?",
    ]
    
    for q in questions:
        print(f"\nOtázka: {q}")
        start = time.perf_counter()
        final_state = app.invoke({"question": q})
        print(f"  Čas: {time.perf_counter() - start:.1f} s")
        print(f"Cesta: {final_state['route']}")
        print(f"Odpověď: {final_state['answer']}")


