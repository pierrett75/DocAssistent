"""Položí otázku z příkazové řádky.

Spuštění: python scripts/ask.py "Kolik mám sick days?"
"""

import sys

from docassist.rag.pipeline import answer


def main() -> None:
  
    sys.argv[0] = "scripts/ask.py"
    if len(sys.argv) < 2:
        print("Použití: python scripts/ask.py \"Otázka\"")
        return

    question = sys.argv[1]
    result = answer(question)
    print(f"Odpověď: {result['answer']}")
    print(f"Zdroje: {', '.join(result['sources'])}")

if __name__ == "__main__":
    main()
