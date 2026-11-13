"""
Den 19, ÚKOL 3: Skript pro vytažení traces za posledních 7 dní a výpis ceny podle tagů (tenant, feature)
Spuštění:
    python scripts/cost_report.py
"""

from collections import defaultdict
from datetime import datetime, timedelta, timezone

from dotenv import load_dotenv
from langfuse import Langfuse

from docassist.config import PROJECT_ROOT

load_dotenv(PROJECT_ROOT / ".env")


def main() -> None:
    langfuse = Langfuse()
    since = datetime.now(timezone.utc) - timedelta(days=7)

    cost_by_tenant: dict[str, float] = defaultdict(float)
    cost_by_feature: dict[str, float] = defaultdict(float)
    cost_by_pair: dict[tuple[str, str], float] = defaultdict(float)

    page = 1
    total_traces = 0

    while True:
        # Stáhneme stránku traces od `since`
        traces = langfuse.fetch_traces(from_timestamp=since, page=page, limit=50).data
        if not traces:
            break

        for t in traces:
            total_traces += 1
            cost = float(getattr(t, "total_cost", 0.0) or 0.0)
            tags = getattr(t, "tags", []) or []

            # Vytáhneme hodnoty z tagů "tenant:xyz" a "feature:abc"
            tenant = next((tag.split(":", 1)[1] for tag in tags if tag.startswith("tenant:")), "unknown")
            feature = next((tag.split(":", 1)[1] for tag in tags if tag.startswith("feature:")), "unknown")

            cost_by_tenant[tenant] += cost
            cost_by_feature[feature] += cost
            cost_by_pair[(tenant, feature)] += cost

        page += 1

    print(f"=== Přehled nákladů za posledních 7 dní (celkem {total_traces} traces) ===\n")

    print("1) Cena podle TENANTA:")
    for tenant, cost in sorted(cost_by_tenant.items(), key=lambda x: -x[1]):
        print(f"   - {tenant:<15} ${cost:.6f}")

    print("\n2) Cena podle FEATURE:")
    for feature, cost in sorted(cost_by_feature.items(), key=lambda x: -x[1]):
        print(f"   - {feature:<15} ${cost:.6f}")

    print("\n3) Detail (Tenant × Feature):")
    for (tenant, feature), cost in sorted(cost_by_pair.items(), key=lambda x: -x[1]):
        print(f"   - {tenant:<12} | {feature:<12} ${cost:.6f}")


if __name__ == "__main__":
    main()
