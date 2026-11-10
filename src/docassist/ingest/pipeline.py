from docassist.config import CHUNK_OVERLAP, CHUNK_SIZE, DATA_DIR
from docassist.db.vector_store import get_collection, index_chunks, reset_collection
from docassist.ingest.chunking import chunk_pages
from docassist.ingest.pdf_loader import load_pdf

def index_pdf(collection, pdf_path) -> int:
    """Načte PDF, rozdělí na chunky, uloží do Chroma. Vrátí počet chunků."""
    pages = load_pdf(pdf_path)
    chunks = chunk_pages(pages, CHUNK_SIZE, CHUNK_OVERLAP)
    if not chunks:
        return 0
    index_chunks(collection, chunks)
    return len(chunks)