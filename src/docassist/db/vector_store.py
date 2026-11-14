"""Uložení chunků do Chroma a vyhledávání (den 5)."""

import chromadb

from docassist.config import CHROMA_DIR, COLLECTION_NAME
from docassist.db.embeddings import get_embeddings


def _client():
    """Pomocná funkce jen pro tento soubor."""
    return chromadb.PersistentClient(path=str(CHROMA_DIR))


def get_collection():
    """Vrátí kolekci s cosine vzdáleností."""
    return _client().get_or_create_collection(        # ← změna: _client()
        name=COLLECTION_NAME,                         # ← změna: konstanta z config
        configuration={"hnsw": {"space": "cosine"}},
    )


def reset_collection() -> None:
    """Smaže celý index."""                           # ← nové
    try:
        _client().delete_collection(COLLECTION_NAME)
    except Exception:
        pass  # kolekce ještě neexistuje (první spuštění), není co mazat


def index_chunks(collection, chunks: list[dict], access: str = "all") -> None:
    """Uloží chunky do kolekce s úrovní přístupu v metadatech."""
    ids = [f"{c['source']}-{c['chunk_id']}" for c in chunks]
    documents = [c["text"] for c in chunks]
    metadatas = [{"source": c["source"], "page": c["page"], "access": access} for c in chunks]
    embeddings = get_embeddings(documents)
 
    collection.delete(where={"source": chunks[0]["source"]})
    collection.upsert(ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings.tolist())


def retrieve(collection, question: str, k: int, where: dict | None = None) -> list[dict]:
    """Najde k chunků nejpodobnějších otázce."""
    question_embedding = get_embeddings([question]).tolist()
    results = collection.query(query_embeddings=question_embedding, n_results=k, where=where)
    return [
        {"text": text, "page": meta["page"], "source": meta["source"], "similarity": 1 - dist}
        for text, meta, dist in zip(
            results["documents"][0], results["metadatas"][0], results["distances"][0]
        )
    ]