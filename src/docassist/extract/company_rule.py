from openai import OpenAI
from pydantic import BaseModel, Field, model_validator
import math 

from docassist.config import CHAT_MODEL

last_tokens = 0

client = OpenAI()


EXTRACTION_PROMPT = """Jsi asistent, který z firemních předpisů vytahuje přesná data.
Vyplň pole podle textu dokumentu. Nic nedomýšlej:
pokud informace v textu není, vrať u pole null."""


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

    
    def extra_vacation_days(self, workdays_per_week: int = 5) -> float:
        """Kolik pracovních dní dovolené má zaměstnanec navíc nad zákonné minimum."""
        return (self.total_vacation_weeks - self.statutory_vacation_weeks) * workdays_per_week

    @model_validator(mode="after")
    def check_total(self):
        if not math.isclose(self.total_vacation_weeks,
                                    self.statutory_vacation_weeks + self.extra_vacation_weeks):
             raise ValueError("Celková výměra neodpovídá součtu zákonné a nadstandardní části")
        return self

def extract_policy(document_text: str) -> VacationPolicy:
    """Pošle celý text směrnice LLM a vrátí vyplněný VacationPolicy."""
   
    completion = client.chat.completions.parse(
        model=CHAT_MODEL,
        temperature=0,
        messages=[
            {"role": "system", "content": EXTRACTION_PROMPT},
            {"role": "user", "content": f"<dokument>\n{document_text}\n</dokument>"},
        ],
        response_format=VacationPolicy, 
    )
    message = completion.choices[0].message
    if message.parsed is None:
        raise ValueError(message.refusal)

    global last_tokens
    last_tokens = completion.usage.total_tokens
    return message.parsed


    