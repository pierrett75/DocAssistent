from os import access
import uuid

import httpx
import openai
from langgraph.errors import GraphRecursionError
from pypdf.errors import PdfReadError

import docassist.api.main as main


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


# ---------------------------------------------------------------------------
# /ask
# ---------------------------------------------------------------------------

def test_ask_vrati_odpoved_a_nastroje(client, fake_agent):
    response = client.post("/ask", json={"question": "Kolik mám sick days?"})

    assert response.status_code == 200
    body = response.json()

    body["answer"] == fake_agent.answer
    body["tools_used"] == ["search_documents"]  
    body["thread_id"] == str(uuid.UUID(body["thread_id"]))
    len(fake_agent.calls) == 1


def test_ask_preda_thread_id_agentovi(client, fake_agent):
    thread_id = str(uuid.uuid4())
    response = client.post("/ask", json={"question": "Kolik mám sick days?", "thread_id": thread_id})
    fake_agent.calls[0]["config"]["configurable"]["thread_id"] == thread_id


def test_ask_prazdna_otazka_nevola_llm(client, fake_agent):
    response = client.post("/ask", json={"question": "   "})
    assert response.status_code == 422
    assert response.json()["error"] == "validation_error"


def test_ask_vypadek_openai_vrati_503(client, fake_agent):
    # Simulace výpadku: FakeAgent při invoke vyhodí stejnou výjimku jako OpenAI SDK bez sítě.
    # APIConnectionError chce objekt požadavku, tak mu dáme falešný.
    fake_agent.error = openai.APIConnectionError(request=httpx.Request("POST", "https://api.openai.com/v1"))
    response = client.post("/ask", json={"question": "Kolik mám sick days?"})
    assert response.status_code == 503
    assert response.json()["error"] == "llm_unavailable"


def test_ask_prilis_mnoho_kroku_vrati_422(client, fake_agent):
    fake_agent.error = GraphRecursionError("limit")
    response = client.post("/ask", json={"question": "Kolik mám sick days?"})
    assert response.status_code == 422
    assert response.json()["error"] == "too_many_steps"


# ---------------------------------------------------------------------------
# /upload
# Soubor se posílá jako files={"file": (název, obsah_v_bajtech, content_type)}
# Obsah nemusí být skutečné PDF: index_pdf je v těchto testech nahrazené.
# ---------------------------------------------------------------------------

PDF_BYTES = b"%PDF-1.4 falesne pdf pro testy"


def test_upload_odmitne_docx(client, monkeypatch):
    monkeypatch.setattr(main, "index_pdf", lambda collection, path, access="all": 5)
    response = client.post("/upload", files={"file": ("smlouva.docx", b"neco", "application/octet-stream")}, headers={"X-User-Role": "hr"})
    assert response.status_code == 400
    assert response.json()["error"] == "unsupported_file"


def test_upload_ulozi_a_zaindexuje(client, monkeypatch, tmp_path):
    # index_pdf nahradím funkcí, která jen vrátí 5 (jako by vzniklo 5 chunků)
    monkeypatch.setattr(main, "index_pdf", lambda collection, path, access="all": 5)
    case = client.post("/upload", files={"file": ("smernice.pdf", PDF_BYTES, "application/pdf")} ,headers={"X-User-Role": "hr"})
    assert case.status_code == 200
    assert case.json() == {"filename": "smernice.pdf", "chunks": 5}
    assert (tmp_path / "smernice.pdf").read_bytes() == PDF_BYTES


def test_upload_pdf_bez_textu_vrati_422_a_smaze_soubor(client, monkeypatch, tmp_path):
    monkeypatch.setattr(main, "index_pdf", lambda collection, path, access="all": 0)
    case = client.post("/upload", files={"file": ("sken.pdf", PDF_BYTES, "application/pdf")} ,headers={"X-User-Role": "hr"})
    assert case.status_code == 422
    assert case.json()["error"] == "empty_document" 
    assert not (tmp_path / "sken.pdf").exists()


def test_upload_rozbite_pdf_vrati_400_a_smaze_soubor(client, monkeypatch, tmp_path):
    def rozbite(collection, path, access="all"):
        raise PdfReadError("rozbité")
    monkeypatch.setattr(main, "index_pdf", rozbite)
    case = client.post("/upload", files={"file": ("rozbite.pdf", PDF_BYTES, "application/pdf")}, headers={"X-User-Role": "hr"})
    assert case.status_code == 400
    assert case.json()["error"] == "invalid_pdf"
    assert not (tmp_path / "rozbite.pdf").exists()


def test_upload_path_traversal(client, monkeypatch, tmp_path):
    monkeypatch.setattr(main, "index_pdf", lambda collection, path, access="all": 1)
    case = client.post("/upload", files={"file": ("../../hack.pdf", PDF_BYTES, "application/pdf")}, headers={"X-User-Role": "hr"})
    assert case.status_code == 200
    assert case.json()["filename"] == "hack.pdf"
    assert (tmp_path / "hack.pdf").exists() 
