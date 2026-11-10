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

llm = ChatOpenAI(model=CHAT_MODEL, temperature=0)
llm_free = llm.bind_tools(TOOLS)                                # smí odpovědět textem
llm_must_use_tool = llm.bind_tools(
    TOOLS, tool_choice={"type": "function", "function": {"name": "search_documents"}}
)

CHITCHAT_ANSWER = ("Ahoj, jsem DocAssist. Odpovídám na otázky o firemních směrnicích, "
                   "třeba o dovolené nebo sick days, a uvádím zdroje.")


class AgentState(MessagesState):
    """MessagesState (seznam zpráv) + cesta, kterou vybral router."""
    route: str


class RouteDecision(BaseModel):
    route: Literal["chitchat", "agent"]
    reason: str = Field(description="Jedna věta, proč")




router_llm = llm.with_structured_output(RouteDecision)   

def router_node(state: AgentState) -> dict:
    """Zvolí, jestli je otázka chitchat (pozdrav, small talk) nebo dotaz na směrnici."""
    decision = router_llm.invoke([SystemMessage(ROUTER_PROMPT), state["messages"][-1]])
    return {"route": decision.route}


def pick_route(state: AgentState) -> Literal["chitchat", "agent"]:
    """Vrátí, kam má jít hrana z routeru: chitchat nebo agent."""
    return state["route"]


def chitchat_node(state: AgentState) -> dict:
    """Vrátí odpověď na pozdrav nebo small talk. Nepoužívá nástroje."""
    return {"messages": [AIMessage(CHITCHAT_ANSWER)]}


def agent_node(state: AgentState) -> dict:
    history = trim_messages(
        state["messages"],
        strategy="last",
        token_counter=llm,
        max_tokens=MAX_HISTORY_TOKENS,
        start_on="human",
    )
  
    if isinstance(state["messages"][-1], HumanMessage):
        response = llm_must_use_tool.invoke([SystemMessage(SYSTEM_PROMPT)] + history)
    else:
        response = llm_free.invoke([SystemMessage(SYSTEM_PROMPT)] + history)
    return {"messages": [response]}




def build_agent(checkpointer=None):
    """checkpointer=None → bez paměti (testy), InMemorySaver() → s pamětí (chat)."""
    graph = StateGraph(AgentState)
   
    graph.add_node("router", router_node)
    graph.add_node("chitchat", chitchat_node)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", ToolNode(TOOLS))

    graph.add_edge(START, "router")
    graph.add_conditional_edges("router", pick_route)
    graph.add_edge("chitchat", END)
    graph.add_conditional_edges("agent", tools_condition)
    graph.add_edge("tools", "agent")  

    return graph.compile(checkpointer=checkpointer)


def default_agent():
    return build_agent(checkpointer=InMemorySaver())
