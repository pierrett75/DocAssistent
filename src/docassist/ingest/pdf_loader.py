from pypdf import PdfReader
from pathlib import Path

def load_pdf(path: Path) -> list[dict]:
    """Načte PDF a vrátí seznam stran.

    Výstup vypadá takto:
    [
        {"text": "Text první strany...", "page": 1, "source": "dokument.pdf"},
        {"text": "Text druhé strany...", "page": 2, "source": "dokument.pdf"},
    ]
    """
    reader = PdfReader(path)
    pages = []
   
    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text()
        if not text or not text.strip():
            continue
        pages.append({"text": text, "page": page_number, "source": path.name})
    return pages

def normalize_text(text: str) -> str:
    """Odstraní z textu nadbytečné mezery a zalomení řádků."""
    return " ".join(text.split())