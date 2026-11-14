"""
Den 16, soubor 3/3: main.py s modely ze schemas.py a chybami z errors.py
Kam: src/docassist/api/main.py   (UPRAV svůj soubor ze dne 15, tohle je cílový stav s TODO)

Do config.py přidej:
    MAX_UPLOAD_MB = 10

Spuštění beze změny:
    uvicorn docassist.api.main:app --reload    →  http://127.0.0.1:8000/docs
"""

import logging
import uuid
from pathlib import Path

from fastapi import FastAPI, Form, Header, UploadFile
from langchain_core.messages import AIMessage, HumanMessage
from langchain_protocol import Literal
from langgraph.errors import GraphRecursionError
from pypdf.errors import PdfReadError

from docassist.agent.graph import default_agent
from docassist.api.errors import (
    AgentStepLimitError,
    BlockedError,
    BlockedError,
    EmptyDocumentError,
    FileTooLargeError,
    ForbiddenError,
    InvalidPdfError,
    UnsupportedFileError,
    register_error_handlers,
)
from docassist.api.schemas import AskRequest, AskResponse, ErrorResponse, UploadResponse
from docassist.config import DATA_DIR, MAX_UPLOAD_MB, RECURSION_LIMIT
from docassist.db.vector_store import get_collection
from docassist.guardrails import DEFAULT_ROLE, allowed_access, check_input
from docassist.ingest.pipeline import index_pdf

# Základní nastavení logů: úroveň INFO a výš, formát s časem.
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("docassist.api")

UPLOAD_ROLES = {"hr", "management"}

app = FastAPI(title="DocAssist", description="Odpovědi na otázky z firemních směrnic")

register_error_handlers(app)

agent = default_agent()


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


# `responses=` jen DOKUMENTUJE, jaké chyby endpoint může vrátit. Uvidíš je v /docs pod endpointem.
# Na chování to nemá vliv (to řeší handlery), ale frontend vývojář hned ví, s čím počítat.
@app.post(
    "/ask",
    response_model=AskResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Neznámá role nebo blokovaný vstup"},
        403: {"model": ErrorResponse, "description": "Přístup odepřen"},
        422: {"model": ErrorResponse, "description": "Neplatná otázka nebo příliš mnoho kroků agenta"},
        503: {"model": ErrorResponse, "description": "Jazykový model je dočasně nedostupný"},
    },
)

def ask(request: AskRequest, x_user_role: str = Header(default=DEFAULT_ROLE)) -> AskResponse:
    """Zeptá se agenta na otázku a vrátí odpověď, thread_id a nástroje, které použil."""

    access = allowed_access(x_user_role)
    if access is None:
        raise ForbiddenError(f"Neznámá role: {x_user_role}")
    violation = check_input(request.question)
    if violation is not None:
        logger.warning("blocked_input role=%s pattern=%s", x_user_role, violation)
        raise BlockedError()

    thread_id = request.thread_id or uuid.uuid4()
    config = {"configurable": {"thread_id": str(thread_id), "access": access}, "recursion_limit": RECURSION_LIMIT}
    try:
        result = agent.invoke({"messages": [HumanMessage(request.question)]}, config)
    except GraphRecursionError:
        raise AgentStepLimitError() from None

    messages = result["messages"]
    last_human = max(i for i, m in enumerate(messages) if isinstance(m, HumanMessage))
    tools_used = {
        call["name"]
        for m in messages[last_human:] if isinstance(m, AIMessage)
        for call in m.tool_calls
    }
    return AskResponse(answer=messages[-1].content, thread_id=thread_id, tools_used=sorted(tools_used))


@app.post(
    "/upload",
    response_model=UploadResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Není to PDF, nebo je PDF poškozené"},
        413: {"model": ErrorResponse, "description": "Soubor je moc velký"},
        422: {"model": ErrorResponse, "description": "PDF neobsahuje text (např. sken)"},
        403: {"model": ErrorResponse, "description": "Přístup odepřen"},
    },
)
def upload(
    file: UploadFile,
    access: Literal["all", "hr", "management"] = Form("all"),     # úroveň DOKUMENTU
    x_user_role: str = Header(default=DEFAULT_ROLE),) -> UploadResponse:
    role = x_user_role.strip().lower()
    if role not in UPLOAD_ROLES:
        raise ForbiddenError("Nahrávat dokumenty smí jen HR a management.")
    if access not in allowed_access(role):
        raise ForbiddenError("Nemůžete nahrát dokument s vyšší úrovní, než sami vidíte.")
    if Path(file.filename).suffix.lower() != ".pdf":
        raise UnsupportedFileError()
    
    content = file.file.read()
    if len(content) > MAX_UPLOAD_MB * 1024 * 1024:
        raise FileTooLargeError(f"Maximální velikost je {MAX_UPLOAD_MB} MB.")
    filename = Path(file.filename).name
    path = DATA_DIR / filename
    path.write_bytes(content)
    try:    
        chunks = index_pdf(get_collection(), path, access)  
    except PdfReadError:
        path.unlink()                     # nenechávej rozbitý soubor v data/
        raise InvalidPdfError() from None
    if chunks == 0:
        path.unlink()
        raise EmptyDocumentError()
    return UploadResponse(filename=filename, chunks=chunks)
