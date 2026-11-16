import json
import re
import statistics
import time
import uuid
from datetime import datetime
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.errors import GraphRecursionError

from docassist.agent.graph import build_agent
from docassist.config import PROJECT_ROOT, RECURSION_LIMIT
from docassist.eval.judge import judge
from langfuse import get_client

from docassist.observability import last_trace_id, tracing_config, tracing_enabled

RUN_NAME = f"{datetime.now():%Y-%m-%d_%H%M}" 
EVAL_DIR = PROJECT_ROOT / "eval"
PASS_THRESHOLD = 4          # případ projde, když jsou všechna tři skóre >= 4 a deterministické kontroly OK


# ---------------------------------------------------------------------------
# 1. Spuštění agenta na jedné otázce
# ---------------------------------------------------------------------------

def run_case(app, question: str, case_id: str) -> dict:
    """Zeptá se agenta (nové vlákno = nový uživatel) a vrátí odpověď, kontext a nástroje."""
    thread_id = str(uuid.uuid4())
    
    config = {"configurable": {"thread_id": thread_id}, 
              "recursion_limit": RECURSION_LIMIT,
              **tracing_config(session_id=f"eval-{RUN_NAME}", user_id="eval", tags=["eval", RUN_NAME, case_id])
              }

    
    start = time.perf_counter()
    result = app.invoke({"messages": [HumanMessage(question)]}, config)
    seconds = time.perf_counter() - start

    messages = result["messages"]
    answer = messages[-1].content
    tools = sorted(call["name"] for m in messages if isinstance(m, AIMessage) for call in m.tool_calls)
    context = "\n\n".join(m.content for m in messages if isinstance(m, ToolMessage))
    return {"answer": answer, "tools": tools, "context": context, "seconds": round(seconds, 1), "trace_id": last_trace_id(config)}


def send_scores(row: dict) -> None:
    """Pošle skóre soudce a deterministické kontroly k trace v Langfuse."""
    if not tracing_enabled() or not row.get("trace_id"):
        return
    langfuse = get_client()

    for metric in ("relevance", "groundedness", "correctness"):
        langfuse.create_score(trace_id=row["trace_id"], name=metric, value=row[metric],
                              comment=row["reasoning"])

    langfuse.create_score(
        trace_id=row["trace_id"],
        name="passed",                          # název skóre v Langfuse (volíš ty)
        value=1 if row["passed"] else 0,        # True → 1, False → 0
        data_type="BOOLEAN",
    )

# ---------------------------------------------------------------------------
# 2. Deterministické kontroly
# ---------------------------------------------------------------------------

CITATION = re.compile(r"\[\d+\]")          # [1], [12] ...


def deterministic_checks(case: dict, run: dict) -> dict:
    """Vrátí {"tool_ok": bool, "citation_ok": bool}. Bez LLM, zdarma."""
    tool_ok = case["expected_tool"] in run["tools"]

    answer = run["answer"]
    if "v dokumentech není" in answer:
        citation_ok = True                                  # fallback: není co citovat
    elif "extract_vacation_policy" in run["tools"] and "VP-HR-04/2026" in answer:
        citation_ok = True                                  # data z extrakce: prompt chce (směrnice VP-HR-04/2026)
    elif "search_documents" in run["tools"]:
        citation_ok = bool(CITATION.search(answer))         # pasáže z vyhledávání: musí být [n]
    else:
        citation_ok = True                                  # žádný nástroj s citacemi
    return {"tool_ok": tool_ok, "citation_ok": citation_ok}


# ---------------------------------------------------------------------------
# 3. Souhrn
# ---------------------------------------------------------------------------

def summarize(rows: list[dict]) -> dict:
    """Průměry a pass rate celkem i po kategoriích."""
    relevance_avg = round(statistics.mean(r["relevance"] for r in rows), 2)
    groundedness_avg = round(statistics.mean(r["groundedness"] for r in rows), 2)
    correctness_avg = round(statistics.mean(r["correctness"] for r in rows), 2)
    tool_accuracy = sum(r["tool_ok"] for r in rows) / len(rows)
    citation_accuracy = sum(r["citation_ok"] for r in rows) / len(rows)
    pass_rate = sum(r["passed"] for r in rows) / len(rows)
    by_category = {}
    for category in set(r["category"] for r in rows):
        category_rows = [r for r in rows if r["category"] == category]
        by_category[category] = sum(r["passed"] for r in category_rows) / len(category_rows)
    return {
        "relevance_avg": relevance_avg,
        "groundedness_avg": groundedness_avg,
        "correctness_avg": correctness_avg,
        "tool_accuracy": tool_accuracy,
        "citation_accuracy": citation_accuracy,
        "pass_rate": pass_rate,
        "by_category": by_category
    }


# ---------------------------------------------------------------------------
# 4. Hlavní běh
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cases = json.loads((EVAL_DIR / "golden_set.json").read_text(encoding="utf-8"))
    app = build_agent(checkpointer=InMemorySaver())
    rows = []

    for case in cases:
        try:
            run = run_case(app, case["question"], case_id=case["id"])
            run["error"] = False
        except GraphRecursionError:
            run = {"answer": "(překročen limit kroků)", "tools": [], "context": "",
                   "seconds": None, "error": True}

        checks = deterministic_checks(case, run)

        if run["error"]:            # chyba agenta: soudce nevolám, nejhorší skóre
            scores = {"reasoning": "chyba: překročen limit kroků",
                      "relevance": 1, "groundedness": 1, "correctness": 1}
        else:
            scores = judge(case["question"], run["context"], case["reference"], run["answer"]).model_dump()

        row = {**case, **run, **checks, **scores}
        
        row["passed"] = (row["relevance"] >= PASS_THRESHOLD and
                         row["groundedness"] >= PASS_THRESHOLD and
                         row["correctness"] >= PASS_THRESHOLD and
                         row["tool_ok"] and
                         row["citation_ok"])
        send_scores(row)
        rows.append(row)

        mark = "PASS" if row["passed"] else "FAIL"
        print(f"{mark}  {case['id']}  R{row['relevance']} G{row['groundedness']} C{row['correctness']}"
              f"  tool={'ok' if row['tool_ok'] else 'X'} cit={'ok' if row['citation_ok'] else 'X'}"
              f"  {case['question'][:50]}")
        if not row["passed"]:
            print(f"        odpověď: {run['answer'][:150]}")
            print(f"        soudce:  {row['reasoning'][:200]}")

    summary = summarize(rows)
    print("\n" + json.dumps(summary, ensure_ascii=False, indent=2))

    out_dir = EVAL_DIR / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{datetime.now():%Y-%m-%d_%H%M}.json"
    out_file.write_text(json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    print(f"\nUloženo: {out_file}")
    if tracing_enabled():
        get_client().flush()