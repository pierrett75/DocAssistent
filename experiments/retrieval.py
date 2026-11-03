import chromadb
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

from chunks import CHUNK_OVERLAP, CHUNK_SIZE, PDF_PATH, chunk_pages, load_pdf

load_dotenv()
client = OpenAI()
EMBEDDING_MODEL = "text-embedding-3-small"


def get_embeddings(texts: list[str]) -> np.ndarray:
    response = client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
    data = sorted(response.data, key=lambda item: item.index)
    return np.array([item.embedding for item in data])


def get_collection():
    """Otevře databázi na disku a vrátí kolekci (něco jako tabulka v SQL)."""
    db = chromadb.PersistentClient(path="chroma_db")
    return db.get_or_create_collection(
        name="docassist",
        # Výchozí měření vzdálenosti v Chroma je jiné (L2). My chceme cosine
        # jako ve dni 3. Pozor: Chroma vrací VZDÁLENOST, ne podobnost:
        #     distance = 1 - cosine_similarity
        #     0 = stejné, čím větší číslo, tím méně podobné
        configuration={"hnsw": {"space": "cosine"}},
    )


def index_chunks(collection, chunks: list[dict]) -> None:
    """Uloží chunky do kolekce: ID, text, metadata a embedding."""
   
    chunks_count = collection.count()
    if chunks_count > 0:
        print(f"Index už existuje ({chunks_count} chunků), přeskakuji.")
        return
 
    ids = [f"{c['source']}-{c['chunk_id']}" for c in chunks]
    documents = [c['text'] for c in chunks]
    metadatas = [{"source": c['source'], "page": c['page']} for c in chunks]
    embeddings = get_embeddings(documents)

    collection.add(ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings.tolist())
    print(f"Index vytvořen ({collection.count()} chunků).")

def retrieve(collection, question: str, k: int = 3, where: dict | None = None) -> list[dict]:  # ← nové: where
    """Najde k chunků nejpodobnějších otázce. where = volitelný filtr podle metadat."""
    question_embedding = get_embeddings([question]).tolist()
    results = collection.query(
        query_embeddings=question_embedding,
        n_results=k,
        where=where,  # ← nové: předá filtr do Chroma (None = bez filtru)
    )
    return [
        {
            "text": text,
            "page": meta["page"],
            "source": meta["source"],
            "similarity": 1 - dist,
        }
        for text, meta, dist in zip(
            results["documents"][0], results["metadatas"][0], results["distances"][0]
        )
    ]


def print_results(question: str, results: list[dict]) -> None:
    """Hezký výpis výsledků s citací."""
    print(f"=== Retrieval: {len(results)} nejpodobnější chunky ===")
    print(f"Otázka: {question}")
    for result in results:
        print(f"  [{result['source']}, strana {result['page']}] ({result['similarity']:.2f}) {result['text'][:150].replace('\n', ' ')}")


if __name__ == "__main__":
    collection = get_collection()

    pages = load_pdf(PDF_PATH)
    chunks = chunk_pages(pages, CHUNK_SIZE, CHUNK_OVERLAP)
    index_chunks(collection, chunks)

    questions = [
        "Kolik žádostí o informace úřad dostal?",        # odpověď je v textu: 272
        "Kolik bylo podáno odvolání?",                   # 17
        "O čem rozhodoval Městský soud v Praze?",
        "Kolik dní dovolené mají zaměstnanci?",          # v dokumentu NENÍ
    ]
    for q in questions:
        print_results(q, retrieve(collection, q, k=3))
        print()

    # TODO 5 (bonus): Filtrování podle metadat. Zkus totéž hledání omezit jen
    #         na stranu 1: collection.query(..., where={"page": 1})
    #         Tohle je základ přístupových práv v týdnu 3 (filtr podle oprávnění).
    print("\n=== Retrieval omezený na stranu 1 ===")
    for q in questions:
            print_results(q, retrieve(collection, q, k=3, where={"page": 1}))
            print()
    
    

    # Otázky k zamyšlení (odpověz si do poznámek):
    # a) Ve dni 3 jsi podobnost počítala smyčkou přes všechny věty. Co dělá
    #    vektorová DB jinak, když má milion chunků?
    # b) Proč Chroma vrací vzdálenost, a ne podobnost? Jak je mezi nimi převedeš?
    # c) Proč při druhém spuštění skript neplatí za embeddingy chunků znovu?
    #    A co by se stalo, kdybys změnila CHUNK_SIZE a index nesmazala?
    # d) Najde retrieval vždycky správný chunk? Co dostane LLM, když ne?
