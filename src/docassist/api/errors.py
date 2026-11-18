import logging

import openai
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError

from docassist.guardrails import BLOCKED_MESSAGE


# logging místo print: má úroveň (INFO/ERROR), čas a u logger.exception i celý traceback.
# V Azure (týden 4) se logy sbírají automaticky, print by se ztratil.
logger = logging.getLogger("docassist.api")
DEFAULT_RETRY_AFTER = 10 

# ---------------------------------------------------------------------------
# 1. VÝJIMKY
#    Jedna rodičovská třída, potomci jen přepíšou tři atributy.
# ---------------------------------------------------------------------------

class DocAssistError(Exception):
    """Předek všech chyb, o kterých víme a umíme je klientovi vysvětlit."""
    status_code: int = 400
    error: str = "bad_request"
    message: str = "Neplatný požadavek."

    def __init__(self, message: str | None = None):
        # Volitelně jde při raise poslat konkrétnější hlášku: raise FileTooLargeError("Max 10 MB.")
        if message:
            self.message = message
        super().__init__(self.message)


class UnsupportedFileError(DocAssistError):
    status_code = 400
    error = "unsupported_file"
    message = "Podporuji jen PDF soubory."


class InvalidPdfError(DocAssistError):
    status_code = 400
    error = "invalid_pdf"
    message = "Soubor je poškozený nebo není platný PDF."

class EmptyDocumentError(DocAssistError):
    status_code = 422
    error = "empty_document"
    message = "PDF neobsahuje žádný text, není co indexovat."

class FileTooLargeError(DocAssistError):
    status_code = 413
    error = "file_too_large"
    message = "Soubor je příliš velký, překračuje povolený limit."

class AgentStepLimitError(DocAssistError):
    status_code = 422
    error = "too_many_steps"
    message = "Agent narazil na limit kroků, zkuste to zjednodušit."

class ForbiddenError(DocAssistError):
    status_code = 403
    error = "forbidden"
    message = "K tomuto požadavku nemáte oprávnění."

class BlockedError(DocAssistError):
    status_code = 400
    error = "blocked_input"
    message = BLOCKED_MESSAGE 

# ---------------------------------------------------------------------------
# 2. HANDLERY: výjimka → JSONResponse
#    FastAPI zavolá handler místo toho, aby vrátilo 500.
# ---------------------------------------------------------------------------

def _error_response(status_code: int, error: str, message: str, details: list[dict]|None = None) -> JSONResponse:
    """Jediné místo, které skládá tělo chyby. Tvar odpovídá ErrorResponse ze schemas.py."""
    content = {"error": error, "message": message}
    if details is not None:
        content["details"] = details
    return JSONResponse(status_code=status_code, content=content)


async def docassist_error_handler(request: Request, exc: DocAssistError) -> JSONResponse:
    """Přeloží vlastní výjimku na JSON odpověď pro klienta."""
    logger.warning("%s: %s", exc.error, exc.message)
    return _error_response(exc.status_code, exc.error, exc.message)

async def rate_limit_handler(request: Request, exc: openai.RateLimitError) -> JSONResponse:
    """429 z LLM → 429 pro klienta i s hlavičkou Retry-After."""
    # Hlavičky odpovědi z Azure jsou v exc.response.headers (httpx.Headers, klíče bez ohledu na velikost písmen).
    try:
        retry_after = int(exc.response.headers.get("retry-after"))
    except (TypeError, ValueError):   # TypeError = hlavička chybí (int(None)), ValueError = není to číslo
        retry_after = DEFAULT_RETRY_AFTER

    logger.warning("Rate limit LLM, retry za %s s", retry_after)

    response = _error_response(429, "rate_limited", f"Příliš mnoho požadavků, zkuste to za {retry_after} s.")
    response.headers["Retry-After"] = str(retry_after)
    return response
    

LLM_TEMPORARY_ERRORS = (
    openai.APIConnectionError,      # zahrnuje i APITimeoutError
    openai.InternalServerError,
)


async def llm_error_handler(request: Request, exc: Exception) -> JSONResponse:
    """Přeloží výjimku z OpenAI na JSON odpověď pro klienta."""
    logger.error("OpenAI nedostupné: %r", exc)
    return _error_response(503, "llm_unavailable", "Jazykový model je dočasně nedostupný, zkuste to za chvíli.")


async def unhandled_error_handler(request: Request, exc: Exception) -> JSONResponse:
    """Přeloží cokoli jiného (chyba v mém kódu) na JSON odpověď pro klienta."""
    logger.exception("Neočekávaná chyba na %s", request.url.path)
    return _error_response(500, "internal_error", "Došlo k neočekávané chybě, zkuste to později.")


def register_error_handlers(app: FastAPI) -> None:
    """Zavolá se jednou v main.py hned po app = FastAPI(...)."""
    app.add_exception_handler(DocAssistError, docassist_error_handler)
    app.add_exception_handler(RequestValidationError, validation_error_handler)
    app.add_exception_handler(openai.RateLimitError, rate_limit_handler)
    for exc_class in LLM_TEMPORARY_ERRORS:
        app.add_exception_handler(exc_class, llm_error_handler)
    app.add_exception_handler(Exception, unhandled_error_handler)   


async def validation_error_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    """Přeloží 422 z validace na JSON odpověď pro klienta."""
    errors = exc.errors()                         # volat jednou, ne dvakrát
    first = errors[0]
    field = first["loc"][-1]                      # ("body", "question") → "question"
    message = first["msg"].removeprefix("Value error, ")
    details = [
        {"field": e["loc"][-1], "message": e["msg"].removeprefix("Value error, ")}
        for e in exc.errors()
    ]
    logger.warning("Validace selhala: %s", details)
    return _error_response(422, "validation_error", f"{field}: {message}", details=details)