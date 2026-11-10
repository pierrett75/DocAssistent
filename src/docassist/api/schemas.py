from uuid import UUID

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Požadavek /ask
# ---------------------------------------------------------------------------

class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000, description="Otázka zaměstnance")

    thread_id: UUID | None = Field(
        default=None,
        description="ID konverzace (UUID). Pošli stejné pro doplňující otázky, nebo nic pro novou.",
    )

    
    @field_validator("question")
    @classmethod
    def strip_question(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Otázka nesmí být prázdná.")
        return value

   
    model_config = {
        "json_schema_extra": {"examples": [{"question": "Kolik mám sick days?"}]}
    }

# ---------------------------------------------------------------------------
# Odpovědi
# ---------------------------------------------------------------------------

class AskResponse(BaseModel):
    answer: str
    thread_id: UUID          
    tools_used: list[str]


class UploadResponse(BaseModel):
    filename: str
    chunks: int = Field(ge=1, description="Kolik chunků se uložilo do indexu")


class ErrorResponse(BaseModel):
    error: str = Field(description="Strojový kód chyby, např. 'unsupported_file'")
    message: str = Field(description="Hláška pro uživatele, česky")
    details: list[dict] | None = None