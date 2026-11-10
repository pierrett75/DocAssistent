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

from fastapi import FastAPI, UploadFile
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.errors import GraphRecursionError
from pypdf.errors import PdfReadError

from docassist.agent.graph import default_agent
from docassist.api.errors import (
    AgentStepLimitError,
    EmptyDocumentError,
    FileTooLargeError,
    InvalidPdfError,
    UnsupportedFileError,
    register_error_handlers,
)
from docassist.api.schemas import AskRequest, AskResponse, ErrorResponse, UploadResponse
from docassist.config import DATA_DIR, MAX_UPLOAD_MB, RECURSION_LIMIT
from docassist.db.vector_store import get_collection
from docassist.ingest.pipeline import index_pdf

# Základní nastavení logů: úroveň INFO a výš, formát s časem.
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

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
        422: {"model": ErrorResponse, "description": "Neplatná otázka nebo příliš mnoho kroků agenta"},
        503: {"model": ErrorResponse, "description": "Jazykový model je dočasně nedostupný"},
    },
)
def ask(request: AskRequest) -> AskResponse:
    """Zeptá se agenta."""

    thread_id = request.thread_id or uuid.uuid4()
    config = {"configurable": {"thread_id": str(thread_id)}, "recursion_limit": RECURSION_LIMIT}
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
    },
)
def upload(file: UploadFile) -> UploadResponse:
    """Nahraje PDF, uloží ho do data/ a zaindexuje do Chroma."""
    if Path(file.filename).suffix.lower() != ".pdf":
        raise UnsupportedFileError()
    content = file.file.read()
    if len(content) > MAX_UPLOAD_MB * 1024 * 1024:
        raise FileTooLargeError(f"Maximální velikost je {MAX_UPLOAD_MB} MB.")
    filename = Path(file.filename).name
    path = DATA_DIR / filename
    path.write_bytes(content)
    try:    
        chunks = index_pdf(get_collection(), path)
    except PdfReadError:
        path.unlink()                     # nenechávej rozbitý soubor v data/
        raise InvalidPdfError() from None
    if chunks == 0:
        path.unlink()
        raise EmptyDocumentError()
    return UploadResponse(filename=filename, chunks=chunks)
