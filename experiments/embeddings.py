import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI()

EMBEDDING_MODEL = "text-embedding-3-small"  # 1536 dimenzí, levný

# Věty schválně z HR směrnice: dvě dvojice znamenají skoro totéž jinými slovy,
# jedna věta je "past" (sdílí slova, ale význam je jiný).
SENTENCES = [
    "Zaměstnanec má nárok na 25 dní dovolené ročně.",           # 0
    "Každý pracovník si může vybrat pět týdnů volna za rok.",    # 1  (parafráze 0)
    "Při nemoci dostane zaměstnanec 5 sick days.",               # 2
    "Když je pracovník nemocný, má k dispozici pět dnů zdravotního volna.",  # 3  (parafráze 2)
    "Firemní parkoviště je otevřené od 6 do 22 hodin.",          # 4  (mimo téma)
]


def get_embeddings(texts: list[str]) -> np.ndarray:
    """Vrátí matici embeddingů, jeden řádek = jedna věta."""
    client_response = client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
    #  print(client_response) pro debug, abych viděl, co se vrací   
    data = sorted(client_response.data, key=lambda item: item.index)
    embeddings = [item.embedding for item in data]
    return np.array(embeddings)



def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Kosinová podobnost dvou vektorů: 1 = stejný směr, 0 = nesouvisí."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
   


def similarity_matrix(vectors: np.ndarray) -> np.ndarray:
    """Matice n x n, kde [i][j] je podobnost věty i a věty j."""
    #normalized = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    #matrix = normalized @ normalized.T
    n = vectors.shape[0]
    matrix = np.zeros((n, n))
  
    for i in range(n):
        for j in range(n):
            matrix[i, j] = cosine_similarity(vectors[i], vectors[j])
    return matrix


def search(query: str, vectors: np.ndarray, top_k: int = 2) -> list[tuple[int, float]]:
    query_embedding = get_embeddings([query])[0]
    scores = np.zeros(vectors.shape[0])
    for i in range(vectors.shape[0]):
        scores[i] = cosine_similarity(query_embedding, vectors[i])
    top_indices = np.argsort(-scores)[:top_k]
    return [(int(i), float(scores[i])) for i in top_indices]


if __name__ == "__main__":
    vectors = get_embeddings(SENTENCES)
    print(f"Tvar matice: {vectors.shape}")  # čekáš (5, 1536)

    # Bonusový postřeh: jsou embeddings od OpenAI už normalizované?
    print(f"Délka prvního vektoru: {np.linalg.norm(vectors[0]):.4f}")

    matrix = similarity_matrix(vectors)
    print("\nMatice podobnosti:")
    for i, row in enumerate(matrix):
        print(i, " ".join(f"{s:.2f}" for s in row))

    best_score = -1
    best_pair = None
    n = len(SENTENCES)
    for i in range(n):
        for j in range(i + 1, n):  # jen dvojice nad diagonálou
            if matrix[i, j] > best_score:
                best_score = matrix[i, j]
                best_pair = (i, j)

    i, j = best_pair
    print(f"\nNejpodobnější: {i} a {j} ({best_score:.2f})")
    print(f"  {SENTENCES[i]}\n  {SENTENCES[j]}")


    print("\nHledání:")
    for q in ["Kolik mám dní dovolené?", "Co když onemocním?", "Kde můžu zaparkovat?"]:
        print(q, "->", search(q, vectors))

