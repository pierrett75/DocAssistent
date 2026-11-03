"""
Týden 2, den 1 (den 8): Structured outputs a Pydantic
Cvičení: vytáhni ze směrnice čísla do Pydantic modelu a výpočet udělej v kódu.

Kam: experiments/den8_structured_outputs.py
Využívá tvůj balíček docassist (proto jsme včera dělali src):
    from docassist.ingest.pdf_loader import load_pdf

Spuštění (z kořene projektu): python experiments/den8_structured_outputs.py
"""

import math

from openai import OpenAI
from pydantic import BaseModel, Field, ValidationError, model_validator

from docassist.config import CHAT_MODEL, DATA_DIR
from docassist.ingest.pdf_loader import load_pdf

client = OpenAI()
PDF_PATH = DATA_DIR / "smernice_dovolena.pdf"


# ---------------------------------------------------------------------------
# ČÁST A: Pydantic bez LLM (jak model funguje)
# ---------------------------------------------------------------------------

class VacationPolicy(BaseModel):
    """Strukturovaná data ze směrnice o dovolené. Každé pole = jedna informace."""

    document_id: str = Field(description="Identifikační označení předpisu, např. VP-HR-04/2026")
    employer: str = Field(description="Název zaměstnavatele")
    total_vacation_weeks: float = Field(ge=0, description="Celková výměra dovolené v týdnech")
    statutory_vacation_weeks: float = Field(ge=0, description="Zákonná část v týdnech")
    extra_vacation_weeks: float = Field(ge=0, description="Část nad rámec zákona v týdnech")
    extra_vacation_request_deadline: str | None = Field(description="Do kdy požádat o převod 5. týdne, null pokud v textu není")
    sick_days_per_year: float = Field(ge=0, description="Počet sick days za rok")
    meal_voucher_value_czk: int | None = Field(description="Hodnota stravenky v Kč, null pokud v textu není")

    # TODO 4 (část C): metoda, která spočítá dny navíc v kódu, ne v LLM.
    def extra_vacation_days(self, workdays_per_week: int = 5) -> float:
        """Kolik pracovních dní dovolené má zaměstnanec navíc nad zákonné minimum."""
        # Spočítej (total_vacation_weeks - statutory_vacation_weeks) * workdays_per_week.
        # K polím modelu se dostaneš přes self, např. self.total_vacation_weeks
        return (self.total_vacation_weeks - self.statutory_vacation_weeks) * workdays_per_week


def demo_validation() -> None:
    """Ukáže, co Pydantic hlídá. Nevolá API, nic nestojí."""
    good = {
        "document_id": "TEST-1",
        "employer": "Firma s.r.o.",
        "total_vacation_weeks": 5,
        "statutory_vacation_weeks": 4,
        "extra_vacation_weeks": 1,
        "extra_vacation_request_deadline": None,
        "sick_days_per_year": "3",       # text "3" → Pydantic převede na 3.0
        "meal_voucher_value_czk": None,
    }
    # TODO 2a: Vytvoř model z dict: policy = VacationPolicy(**good)
    #          a vypiš ho: print(policy). Všimni si, co se stalo se "3".

    # TODO 2b: Zkus špatná data: zkopíruj good, nastav "total_vacation_weeks": -2
    #          a vytvoř model v try/except ValidationError. Vypiš chybu.
    #          Pak zkus smazat povinné pole (del bad["employer"]) a sleduj hlášku.
    policy = VacationPolicy(**good)
    bad = good.copy()
    print(policy)

    good["total_vacation_weeks"] = -2
    try:
        bad = VacationPolicy(**good)
    except ValidationError as e:
        print("ValidationError:", e)


# ---------------------------------------------------------------------------
# ČÁST B: LLM vyplní model (structured outputs)
# ---------------------------------------------------------------------------

EXTRACTION_PROMPT = """Jsi asistent, který z firemních předpisů vytahuje přesná data.
Vyplň pole podle textu dokumentu. Nic nedomýšlej:
pokud informace v textu není, vrať u pole null."""


def extract_policy(document_text: str) -> VacationPolicy:
    """Pošle celý text směrnice LLM a vrátí vyplněný VacationPolicy."""
   
    completion = client.chat.completions.parse(
        model=CHAT_MODEL,
        temperature=0,
        messages=[
            {"role": "system", "content": EXTRACTION_PROMPT},
            {"role": "user", "content": f"<dokument>\n{document_text}\n</dokument>"},
        ],
        response_format=VacationPolicy,     # ← tady předáváš celou TŘÍDU, ne instanci
    )
    message = completion.choices[0].message
    if message.parsed is None:
        raise ValueError(message.refusal)
    return message.parsed


if __name__ == "__main__":
    print("=== Část A: validace ===")
    demo_validation()

    print("\n=== Část B: extrakce ===")
    pages = load_pdf(PDF_PATH)
    full_text = "\n\n".join(f"[strana {p['page']}]\n{p['text']}" for p in pages)
    policy = extract_policy(full_text)
    print(policy.model_dump_json(indent=2))   # hezký JSON výpis

    print("\n=== Část C: výpočet v kódu ===")
    print(f"Dny dovolené navíc nad zákonné minimum: {policy.extra_vacation_days()}")



    @model_validator(mode="after")
    def check_total(self):
        if not math.isclose(self.total_vacation_weeks,
                            self.statutory_vacation_weeks + self.extra_vacation_weeks):
            raise ValueError("Celková výměra neodpovídá součtu zákonné a nadstandardní části")
        return self

  