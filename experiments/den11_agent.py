"""
Týden 2, den 4 (den 11): Chain vs. agent + nástroj pro shrnutí
Cvičení: postavíš stejného asistenta jako AGENTA v LangGraph se 3 nástroji
(vyhledání, extrakce, shrnutí) a porovnáš ho s routerem ze dne 10.

Tři přístupy, které teď znáš:
    chain  (den 6):  retrieve → generate            pevná cesta, vždy stejná
    router (den 10): router → 1 nástroj → generate  LLM vybere JEDNU cestu z nakreslených
    agent  (dnes):   agent ⇄ tools → konec          LLM volá nástroje v cyklu, kolik chce

Graf agenta:
    START ─> agent ──(chce nástroj?)──> tools ─┐
               ▲                               │
               └───────────────────────────────┘
             agent ──(nechce)──> END

Instalace:  pip install langchain-openai   (a přidej do pyproject.toml)
Kam:        experiments/den11_agent.py
Spuštění:   python experiments/den11_agent.py
"""

import json
import time

from langchain_core import messages
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.errors import GraphRecursionError
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from openai import OpenAI

from docassist.config import CHAT_MODEL, DATA_DIR, TOP_K
from docassist.db.vector_store import get_collection, retrieve
from docassist.extract.company_rule import extract_policy
from docassist.ingest.pdf_loader import load_pdf
from docassist.rag.pipeline import format_context

client = OpenAI()
tool_tokens = 0
PDF_PATH = DATA_DIR / "smernice_dovolena.pdf"


def load_full_text() -> str:
    pages = load_pdf(PDF_PATH)
    return "\n\n".join(f"[strana {p['page']}]\n{p['text']}" for p in pages)


# ---------------------------------------------------------------------------
# 1. NÁSTROJE přes @tool
#    @tool udělá z funkce nástroj. JSON schéma z dne 9 se vytvoří samo:
#      - jméno = jméno funkce
#      - parametry = typy argumentů (query: str)
#      - DESCRIPTION = DOCSTRING  ← docstring je teď prompt pro model!
# ---------------------------------------------------------------------------

@tool
def search_documents(query: str) -> str:
    """Vyhledá v interních směrnicích pasáže k jednomu pravidlu, postupu nebo podmínce.
    Použij i pro otázku na jedno číslo (např. kolik mám sick days) a i když si myslíš,
    že odpověď v dokumentech není. Vrací pasáže s čísly [n] a stranou.
    query: dotaz česky, slovy, jaká mohou být v dokumentu."""
    results = retrieve(get_collection(), query, k=TOP_K)
    return format_context(results)


@tool
def extract_vacation_policy() -> str:
    """Vytáhne ze směrnice o dovolené všechna čísla najednou a dopočítá dny navíc.
    Použij JEN pro přehled více čísel nebo pro výpočet, např.
    kolik dní dovolené mám navíc nad zákonné minimum' nebo 'dej mi přehled čísel o dovolené'.
    Na otázku na jedno pravidlo nebo podmínku (např. sick days, převod dovolené) použij search_documents."""
    global tool_tokens
    data_policy = extract_policy(load_full_text())
    tool_tokens += getattr(data_policy, "last_tokens", 0)
    data = data_policy.model_dump()
    data["extra_vacation_days"] = data_policy.extra_vacation_days()
    return json.dumps(data, ensure_ascii=False)


SUMMARY_PROMPT = """Shrň text firemní směrnice česky v 5 až 8 odrážkách.
Drž se jen textu, nic nedomýšlej. U každé odrážky uveď stranu ve tvaru (str. N).
Pokud je zadané téma, shrň jen část, která se ho týká."""


@tool
def summarize_document(topic: str) -> str:
    """Shrne směrnici o dovolené a sick days do přehledných odrážek se stranami.
    Použij, když uživatel chce shrnutí, přehled pravidel nebo "o čem směrnice je".
    topic: téma shrnutí (např. "sick days", "převod dovolené"), nebo "celý dokument"."""

    response = client.chat.completions.create(
        model=CHAT_MODEL,
        temperature=0,
        messages=[
            {"role": "system", "content": SUMMARY_PROMPT},
            {"role": "user", "content": f"Téma: {topic}\n\n<dokument>\n{load_full_text()}\n</dokument>"},
        ],
    )

    global tool_tokens
    tool_tokens += response.usage.total_tokens
    return response.choices[0].message.content


TOOLS = [search_documents, extract_vacation_policy, summarize_document]


# ---------------------------------------------------------------------------
# 2. AGENT = uzel, který zavolá LLM s připojenými nástroji
#    MessagesState = hotový stav se seznamem zpráv "messages".
#    Uzel vrací {"messages": [nová_zpráva]} a LangGraph ji PŘIDÁ na konec
#    (ve dni 9 jsi dělala messages.append ručně).
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """Jsi asistent DocAssist pro zaměstnance. Na každou otázku kromě pozdravu
nejdřív použij nástroj. Otázka může potřebovat i víc nástrojů za sebou.
Odpovídej jen podle výsledků nástrojů. U pasáží cituj [n], u JSON a shrnutí uveď stranu
nebo označení dokumentu. Když obecné pravidlo dopadá na otázku, odpověz podle něj.
Oslovuj neutrálně, bez rodu. Když nástroje nic relevantního nenajdou, řekni:
"Odpověď v dokumentech není." """

llm = ChatOpenAI(model=CHAT_MODEL, temperature=0)
llm_with_tools = llm.bind_tools(TOOLS)


def agent_node(state: MessagesState) -> dict:
    """Zavolá LLM s připojenými nástroji. Vrátí {"messages": [nová_zpráva]}."""
    llm_response = llm_with_tools.invoke(state["messages"])
    return {"messages": [llm_response]}


# ---------------------------------------------------------------------------
# 3. GRAF
#    ToolNode(TOOLS)  = hotový uzel, který spustí všechny tool_calls z poslední
#                       AIMessage a výsledky přidá jako ToolMessage (= tvůj run_tool z dne 9)
#    tools_condition  = hotová podmíněná hrana: má poslední zpráva tool_calls?
#                       ano → "tools", ne → END   (= tvůj `if not message.tool_calls` z dne 9)
# ---------------------------------------------------------------------------

def build_agent():
    graph = StateGraph(MessagesState)
   
    graph.add_node("agent", agent_node)
    graph.add_node("tools", ToolNode(TOOLS))

    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", tools_condition)
    graph.add_edge("tools", "agent")

    return graph.compile()


# ---------------------------------------------------------------------------
# 4. Výpis: jaké nástroje agent zavolal a kolikrát volal LLM
# ---------------------------------------------------------------------------

def show_trace(messages: list) -> int:
    llm_calls = 0
    tokens = 0
    for m in messages:
        if isinstance(m, AIMessage):
            llm_calls += 1
            usage = getattr(m, "usage_metadata", None) or {}
            tokens += usage.get("total_tokens", 0)
            for call in getattr(m, "tool_calls", []) or []:
                print(f"  → {call['name']}({call['args']})")
        elif isinstance(m, ToolMessage):
            print(f"  ← {m.name}: {len(m.content)} znaků")
    print(f"  LLM voláno: {llm_calls}×")
    print(f"  Celkový počet tokenů: {tokens}")
    return tokens


if __name__ == "__main__":
    app = build_agent()
    print(app.get_graph().draw_mermaid())

    questions = [
        "Ahoj, co umíš?",
        "Kolik mám sick days?",
        "Shrň mi, o čem je směrnice.",
        "Shrň pravidla pro sick days a spočítej, kolik dní dovolené mám navíc nad zákonné minimum.",
        "Jaké je heslo na wifi?",
    ]
    grand_total = 0
    for q in questions:
        tool_tokens = 0                              # vynuluj pro každou otázku
        print(f"\nOtázka: {q}")
        start = time.perf_counter()                  # ← stopky START
        try:
            result = app.invoke(
                {"messages": [SystemMessage(SYSTEM_PROMPT), HumanMessage(q)]},
                {"recursion_limit": 10},
            )
        except GraphRecursionError:
            print("  Překročen limit kroků.")
            continue
        seconds = time.perf_counter() - start        # ← stopky STOP
        agent_tokens = show_trace(result["messages"])
        question_total = agent_tokens + tool_tokens
        grand_total += question_total
        print(f"  Tokeny: agent {agent_tokens} + nástroje {tool_tokens} = {question_total}")
        print(f"  Čas: {seconds:.1f} s")
        print(f"Odpověď: {result['messages'][-1].content}")

    print(f"\nCelkem za všechny otázky: {grand_total} tokenů")
