from typing import Literal

from openai import OpenAI
from pydantic import BaseModel, Field

from docassist.config import JUDGE_MODEL

client = OpenAI()

Score = Literal[1, 2, 3, 4, 5]     # Literal místo int + ge/le: OpenAI structured outputs to umí jako výčet


JUDGE_PROMPT = """Jsi přísný hodnotitel odpovědí interního HR asistenta.
Dostaneš otázku zaměstnance, kontext (výsledky nástrojů, ze kterých asistent směl čerpat),
referenční odpověď od člověka a odpověď asistenta.

Nejdřív napiš stručné zdůvodnění (reasoning), až potom skóre. Hodnoť každou metriku zvlášť.

relevance (odpovídá na položenou otázku?):
  5 = přímo a úplně odpovídá na otázku
  3 = odpovídá částečně nebo s velkou vatou
  1 = odpovídá na něco jiného

groundedness (je každé tvrzení v odpovědi podložené kontextem?):
  5 = všechna tvrzení a čísla jsou v kontextu
  3 = hlavní tvrzení podložená, ale jedno vedlejší tvrzení nebo dopočet v kontextu není
  1 = klíčové tvrzení nebo číslo v kontextu není (halucinace)
  Odpověď "v dokumentech není", když kontext opravdu nic relevantního neobsahuje = 5.

correctness (souhlasí s referenční odpovědí?):
  5 = obsahuje všechna fakta z reference a nic jí neodporuje
  3 = hlavní fakt správně, ale chybí podstatná podmínka nebo výjimka z reference
  1 = odporuje referenci, nebo odpoví, i když reference říká, že odpověď v dokumentech není

  Chybová hláška nebo žádná odpověď = 1 ve všech metrikách.

Nepřidávej body za délku ani zdvořilost. Delší odpověď není lepší."""


class Verdict(BaseModel):
    """Verdikt soudce: zdůvodnění + tři skóre 1–5."""
    reasoning: str = Field(description="Stručné zdůvodnění, 2–4 věty, pro každou metriku")
    relevance: Score
    groundedness: Score
    correctness: Score


def judge(question: str, context: str, reference: str, answer: str) -> Verdict:
    """Ohodnotí jednu odpověď. Vrací Verdict."""
    user_content = f"""<otazka>
{question}
</otazka>

<kontext>
{context or "(asistent nepoužil žádný nástroj)"}
</kontext>

<reference>
{reference}
</reference>

<odpoved_asistenta>
{answer}
</odpoved_asistenta>"""

    
    completion = client.chat.completions.parse(
        model=JUDGE_MODEL, temperature=0,
        messages=[
            {"role": "system", "content": JUDGE_PROMPT},
            {"role": "user", "content": user_content},
        ],
        response_format=Verdict,
    )
    message = completion.choices[0].message
    if message.parsed is None:
        raise ValueError(message.refusal)
    return message.parsed
