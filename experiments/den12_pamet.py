from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, trim_messages
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.errors import GraphRecursionError
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from docassist.agent.prompts import SYSTEM_PROMPT
from docassist.agent.tools import TOOLS
from docassist.config import CHAT_MODEL, MAX_HISTORY_TOKENS, RECURSION_LIMIT

llm = ChatOpenAI(model=CHAT_MODEL, temperature=0)
llm_with_tools = llm.bind_tools(TOOLS)

RECURSION_LIMIT = 10        # kolik kroků grafu smí jedna otázka udělat


# ---------------------------------------------------------------------------
# 1. AGENT s oříznutou historií
#    Ve dni 11 byl SystemMessage uložený ve stavu. Teď ho přidávám až tady,
#    při každém volání. Ve stavu (= paměti) je jen konverzace, system prompt
#    se neukládá s každou otázkou znovu.
# ---------------------------------------------------------------------------

def agent_node(state: MessagesState) -> dict:
    """Zavolá LLM s připojenými nástroji. Vrátí {"messages": [nová_zpráva]}."""
    history = trim_messages(
        state["messages"],
        strategy="last",
        token_counter=llm,
        max_tokens=MAX_HISTORY_TOKENS,  # ~1000 tokenů na zprávu
        start_on="human",
    )
    response = llm_with_tools.invoke([SystemMessage(SYSTEM_PROMPT)] + history)
    return {"messages": [response]}


def build_agent():
    graph = StateGraph(MessagesState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", ToolNode(TOOLS))
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", tools_condition)
    graph.add_edge("tools", "agent")

    #         (InMemorySaver = v RAM, po skončení skriptu zmizí. V produkci SQLite/Postgres.)
    
    return graph.compile(checkpointer=InMemorySaver())


# ---------------------------------------------------------------------------
# 2. KONVERZACE: posílám jen NOVOU otázku, historii drží checkpointer
# ---------------------------------------------------------------------------

def ask(app, question: str, thread_id: str) -> str:
    """Vrátí odpověď agenta na otázku. thread_id = paměť konverzace."""
    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": RECURSION_LIMIT}
    try:
        result = app.invoke({"messages": [HumanMessage(question)]}, config)
    except GraphRecursionError:
        return "Omlouvám se, ale tato otázka je příliš složitá a vyžaduje víc kroků, než je povoleno. Zkus ji prosím zjednodušit."
    return result["messages"][-1].content


def show_memory(app, thread_id: str) -> None:
    """Kolik zpráv je v paměti a kolik tokenů stálo poslední volání agenta."""
    state = app.get_state({"configurable": {"thread_id": thread_id}})
    messages = state.values["messages"]
    last_ai = [m for m in messages if isinstance(m, AIMessage)][-1]
    print(f"  [paměť: {len(messages)} zpráv, poslední volání: "
          f"{last_ai.usage_metadata['input_tokens']} vstupních tokenů]")


if __name__ == "__main__":
    app = build_agent()

    conversation = [
        "Kolik mám sick days?",
        "A kolik, když nastoupím v srpnu?",          # "kolik" čeho? potřebuje paměť
        "Můžu je převést do dalšího roku?",           # "je" = sick days
        "Kolik dní dovolené mám navíc nad zákonné minimum?",
        "A do kdy musím požádat o její převod?",      # "její" = dovolené navíc
    ]

    print("=== Konverzace A (paměť zapnutá, thread_id='uzivatel_a') ===")
    for q in conversation:
        print(f"\nOtázka: {q}")
        print(f"Odpověď: {ask(app, q, thread_id='uzivatel_a')}")
        show_memory(app, "uzivatel_a")

    print("\n=== Konverzace B (nové thread_id = nový uživatel) ===")
    q = "A kolik, když nastoupím v srpnu?"
    print(f"\nOtázka: {q}")
    print(f"Odpověď: {ask(app, q, thread_id='uzivatel_b')}")
