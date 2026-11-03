"""Postaví index ze všech PDF ve složce data/.

Spuštění (z kořene projektu): python scripts/build_index.py
"""

from docassist.config import CHUNK_OVERLAP, CHUNK_SIZE, DATA_DIR
from docassist.db.vector_store import get_collection, index_chunks, reset_collection
from docassist.ingest.chunking import chunk_pages
from docassist.ingest.pdf_loader import load_pdf


def main() -> None:

    pdf_paths = sorted(DATA_DIR.glob("*.pdf"))
    if not pdf_paths:
        print("Ve složce data/ nejsou žádná PDF.")
        return

    reset_collection()
    collection = get_collection()

    for pdf_path in pdf_paths:
        pages = load_pdf(pdf_path)
        chunks = chunk_pages(pages, CHUNK_SIZE, CHUNK_OVERLAP)
        index_chunks(collection, chunks)
        print(f"{pdf_path.name}: {len(chunks)} chunků")
        
    print(f"Celkem v indexu: {collection.count()} chunků")


if __name__ == "__main__":
    main()
