from langchain_core.messages import HumanMessage
from langgraph.errors import GraphRecursionError

from docassist.agent.graph import default_agent
from docassist.config import RECURSION_LIMIT

if __name__ == "__main__":
    app = default_agent()
    config = {"configurable": {"thread_id": "terminal"}, "recursion_limit": RECURSION_LIMIT}
    print("DocAssist. Ptej se na směrnice, 'konec' ukončí.\n")
    while True:
        question = input("Ty: ").strip()
        if question.lower() in {"konec", "exit", "quit"}:
            break
        if not question:
            continue
        try:
            result = app.invoke({"messages": [HumanMessage(question)]}, config)
            print(f"DocAssist: {result['messages'][-1].content}\n")
        except GraphRecursionError:
            print("DocAssist: Tahle otázka potřebuje příliš mnoho kroků, zkus ji zjednodušit.\n")
