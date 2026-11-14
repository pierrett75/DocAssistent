import logging
from typing import Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, trim_messages
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from pydantic import BaseModel, Field

from docassist.agent.prompts import ROUTER_PROMPT, SYSTEM_PROMPT
from docassist.agent.tools import TOOLS
from docassist.config import CHAT_MODEL, MAX_HISTORY_TOKENS

OFF_TOPIC_ANSWER = ("Pomáhám jen s otázkami k firemním směrnicím, třeba k dovolené nebo sick days. "
                    "S tímhle vám bohužel neporadím.")
ATTACK_ANSWER = "Tuhle žádost nemůžu splnit. Můžu ale odpovědět na otázku k firemním směrnicím."

logger = logging.getLogger("docassist.api")

llm = ChatOpenAI(model=CHAT_MODEL, temperature=0)
llm_free = llm.bind_tools(TOOLS)                                # smí odpovědět textem
llm_must_use_tool = llm.bind_tools(
    TOOLS, tool_choice={"type": "function", "function": {"name": "search_documents"}}
)
llm_no_tools = llm.bind_tools(TOOLS, tool_choice="none")   # nástroje zná, ale nesmí je volat

# Max. počet kol "agent → nástroj" k jedné otázce. Kroky grafu: router 1 + 3 × 2 + odpověď 1 = 8 < RECURSION_LIMIT 10.
MAX_TOOL_ROUNDS = 3
FINAL_ANSWER_PROMPT = (
    "Nástroje už volat nemůžeš. Odpověz teď na PŮVODNÍ otázku uživatele jen podle výsledků nástrojů výše. "
    "Nepiš žádnou novou otázku ani dotaz. Když ve výsledcích odpověď není, napiš přesně: "
    "\"Odpověď v dokumentech není.\""
)

CHITCHAT_ANSWER = ("Ahoj, jsem DocAssist. Odpovídám na otázky o firemních směrnicích, "
                   "třeba o dovolené nebo sick days, a uvádím zdroje.")


class AgentState(MessagesState):
    """MessagesState (seznam zpráv) + cesta, kterou vybral router."""
    route: str
    refusal_reason: str 

class RouteDecision(BaseModel):
    """Výsledek routeru: kam má jít hrana z routeru."""
    route: Literal["chitchat", "off_topic", "attack", "agent"]
    reason: str = Field(description="Jedna věta, proč")




router_llm = llm.with_structured_output(RouteDecision)   

def router_node(state: AgentState) -> dict:
    """Zvolí, jestli je otázka chitchat (pozdrav, small talk) nebo dotaz na směrnici."""
    decision = router_llm.invoke([SystemMessage(ROUTER_PROMPT), state["messages"][-1]])
    if decision.route in ("off_topic", "attack"):
        logger.warning("router %s: %s", decision.route, decision.reason)
        return {"route": "refusal", "refusal_reason": decision.route}
    return {"route": decision.route, "refusal_reason": ""}
   

def pick_route(state: AgentState) -> Literal["chitchat", "agent", "refusal"]:
    """Vrátí, kam má jít hrana z routeru: chitchat nebo agent."""
    return state["route"]


def chitchat_node(state: AgentState) -> dict:
    """Vrátí odpověď na pozdrav nebo small talk. Nepoužívá nástroje."""
    return {"messages": [AIMessage(CHITCHAT_ANSWER)]}

def refusal_node(state: AgentState) -> dict:
    """Odmítne otázku mimo téma nebo útok. Žádný nástroj, žádné LLM."""
    answer = ATTACK_ANSWER if state["refusal_reason"] == "attack" else OFF_TOPIC_ANSWER
    return {"messages": [AIMessage(answer)]}


def agent_node(state: AgentState) -> dict:
    history = trim_messages(
        state["messages"],
        strategy="last",
        token_counter=llm,
        max_tokens=MAX_HISTORY_TOKENS,
        start_on="human",
    )

    # Kolikrát už agent k AKTUÁLNÍ otázce volal nástroje (od poslední HumanMessage).
    messages = state["messages"]
    last_human = max(i for i, m in enumerate(messages) if isinstance(m, HumanMessage))
    tool_rounds = sum(1 for m in messages[last_human:] if isinstance(m, AIMessage) and m.tool_calls)

    if isinstance(messages[-1], HumanMessage):
        llm_to_use = llm_must_use_tool          # první krok: vždy hledat
    elif tool_rounds >= MAX_TOOL_ROUNDS:
        llm_to_use = llm_no_tools               # pojistka v kódu: dost hledání, odpověz z toho, co máš
    else:
        llm_to_use = llm_free

    prompt = [SystemMessage(SYSTEM_PROMPT)] + history
    if llm_to_use is llm_no_tools:
        # Bez téhle věty model občas napíše jako "odpověď" dotaz, který chtěl poslat do nástroje.
        prompt.append(SystemMessage(FINAL_ANSWER_PROMPT))
    response = llm_to_use.invoke(prompt)
    return {"messages": [response]}


def build_agent(checkpointer=None):
    """checkpointer=None → bez paměti (testy), InMemorySaver() → s pamětí (chat)."""
    graph = StateGraph(AgentState)
   
    graph.add_node("router", router_node)
    graph.add_node("chitchat", chitchat_node)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", ToolNode(TOOLS))
    graph.add_node("refusal", refusal_node) 

    graph.add_edge(START, "router")
    graph.add_conditional_edges("router", pick_route)
    graph.add_edge("chitchat", END)
    graph.add_conditional_edges("agent", tools_condition)
    graph.add_edge("tools", "agent")
    graph.add_edge("refusal", END)  

    return graph.compile(checkpointer=checkpointer)


def default_agent():
    return build_agent(checkpointer=InMemorySaver())
