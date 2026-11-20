import json
import uuid
 
from langchain_core.callbacks import get_usage_metadata_callback
from langchain_core.messages import HumanMessage
 
from docassist.agent.graph import build_agent
from docassist.config import PROJECT_ROOT, RECURSION_LIMIT
 
N_QUESTIONS = 10
 
# USD za 1 milion tokenů (ověř ceník, viz NAVOD A2)
PRICE_INPUT = 0.40
PRICE_CACHED_INPUT = 0.10
PRICE_OUTPUT = 1.60
 
 
def ask(app, question: str) -> dict:
    """Položí otázku a vrátí {"input": int, "cached": int, "output": int, "calls": int}."""
    config = {"configurable": {"thread_id": str(uuid.uuid4())}, "recursion_limit": RECURSION_LIMIT}
 
    with get_usage_metadata_callback() as cb:
        app.invoke({"messages": [HumanMessage(question)]}, config)
    
    total_input = total_output = total_cached = 0
    for metriky in cb.usage_metadata.values():
        total_input += metriky.get("input_tokens", 0)
        total_output += metriky.get("output_tokens", 0)
        total_cached += metriky.get("input_token_details", {}).get("cache_read", 0)

    return {"input": total_input, "cached": total_cached, "output": total_output,
            "models": len(cb.usage_metadata)}
 
 
def cost_usd(input_tokens: float, cached: float, output_tokens: float) -> float:
    return ((input_tokens - cached) * PRICE_INPUT
            + cached * PRICE_CACHED_INPUT
            + output_tokens * PRICE_OUTPUT) / 1_000_000

 
def main() -> None:
    cases = json.loads((PROJECT_ROOT / "eval" / "golden_set.json").read_text(encoding="utf-8"))
    app = build_agent()
 
    rows = []
    for case in cases[:N_QUESTIONS]:
        usage = ask(app, case["question"])
        rows.append(usage)
        print(f"{case['id']}: in={usage['input']:>6} cached={usage['cached']:>6} out={usage['output']:>5}")
 
    n = len(rows)
    avg_input = sum(r["input"] for r in rows) / n
    avg_cached = sum(r["cached"] for r in rows) / n
    avg_output = sum(r["output"] for r in rows) / n
    print(f"\nPrůměr na dotaz: in={avg_input:.0f} cached={avg_cached:.0f} out={avg_output:.0f}")

    
    per_query = cost_usd(avg_input, avg_cached, avg_output)
    print(f"Cena za 1 dotaz:     ${per_query:.5f}")
    print(f"Cena za 1000 dotazů: ${per_query * 1000:.2f}")

    
    print(f"Poměr vstup:výstup = {avg_input / avg_output:.0f}:1")
 
 
if __name__ == "__main__":
    main()
 