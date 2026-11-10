import time
import uuid

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.errors import GraphRecursionError

from docassist.agent.graph import build_agent
from docassist.config import RECURSION_LIMIT

# Každý případ: otázka, které nástroje se MUSÍ zavolat, co MUSÍ být v odpovědi.
# "conversation" = víc otázek za sebou ve stejném vlákně (test paměti), kontroluje se poslední.
CASES = [
    {"name": "pozdrav",       "ask": ["Ahoj, co umíš?"],
     "tools": set(),                      "contains": ["DocAssist"]},
    {"name": "sick days",     "ask": ["Kolik mám sick days?"],
     "tools": {"search_documents"},       "contains": ["3", "1,5"]},
    {"name": "pátek",         "ask": ["Můžu si vzít sick day v pátek před dovolenou?"],
     "tools": {"search_documents"},       "contains": ["zdravotní"]},
    {"name": "dny navíc",     "ask": ["Kolik pracovních dní dovolené mám navíc nad zákonné minimum?"],
     "tools": {"extract_vacation_policy"}, "contains": ["5"]},
    {"name": "wifi",          "ask": ["Jaké je heslo na wifi?"],
     "tools": {"search_documents"},       "contains": ["není"]},
    {"name": "shrnutí + výpočet",
     "ask": ["Shrň pravidla pro sick days a spočítej, kolik dní dovolené mám navíc nad zákonné minimum."],
     "tools": {"summarize_document", "extract_vacation_policy"}, "contains": ["5"]},
    {"name": "paměť",         "ask": ["Kolik mám sick days?", "A kolik, když nastoupím v srpnu?"],
     "tools": {"search_documents"},       "contains": ["1,5"]},
]


def tools_used(messages: list) -> set[str]:
    """Jména všech nástrojů, které agent v POSLEDNÍ otázce zavolal."""
    # najdi index poslední HumanMessage a projdi jen zprávy za ní
    last_human = max(i for i, m in enumerate(messages) if isinstance(m, HumanMessage))
    names = set()
    for m in messages[last_human:]:
        if isinstance(m, AIMessage):
            names |= {call["name"] for call in m.tool_calls}
    return names


def check(case: dict, answer: str, used: set[str]) -> list[str]:
    """Vrátí seznam problémů. Prázdný seznam = PASS."""
    problems = []
    # TODO 5:
    #   1) Chybí nějaký očekávaný nástroj?
    #        missing = case["tools"] - used          ← rozdíl množin: co je v prvním a není v druhém
    #        když missing není prázdné: problems.append(f"chybí nástroj {missing}")
    #   2) U pozdravu (case["tools"] je prázdné) se nesmí volat žádný nástroj:
    #        když not case["tools"] and used: problems.append(f"zbytečně volal {used}")
    #   3) Pro každé slovo v case["contains"]: když není v answer, problems.append(f"v odpovědi chybí '{slovo}'")
    missing = case["tools"] - used
    if missing:
        problems.append(f"chybí nástroj {missing}")
    if not case["tools"] and used:
        problems.append(f"zbytečně volal {used}")
    for word in case["contains"]:
        if word.lower() not in answer.lower():
            problems.append(f"v odpovědi chybí '{word}'")
    return problems


if __name__ == "__main__":
    app = build_agent(checkpointer=InMemorySaver())
    passed = 0
    for case in CASES:
        config = {"configurable": {"thread_id": str(uuid.uuid4())},   # každý případ = nový uživatel
                  "recursion_limit": RECURSION_LIMIT}
        start = time.perf_counter()
        try:
            for question in case["ask"]:
                result = app.invoke({"messages": [HumanMessage(question)]}, config)
        except GraphRecursionError:
            print(f"FAIL  {case['name']:<20} překročen limit kroků")
            continue
        seconds = time.perf_counter() - start
        answer = result["messages"][-1].content
        used = tools_used(result["messages"])
        problems = check(case, answer, used)
        status = "PASS" if not problems else "FAIL"
        passed += not problems
        print(f"{status}  {case['name']:<20} {seconds:4.1f} s  nástroje={sorted(used)}")
        for p in problems:
            print(f"        - {p}")
            print(f"        odpověď: {answer[:200]}")

    print(f"\n{passed}/{len(CASES)} prošlo")
    # Spusť DVAKRÁT za sebou. Druhý běh: shrnutí a extrakce z cache → o kolik je rychlejší?
