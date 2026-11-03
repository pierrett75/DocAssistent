import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

from docassist.config import EMBEDDING_MODEL

load_dotenv()
client = OpenAI()


def get_embeddings(texts: list[str]) -> np.ndarray:
    """Vrátí matici embeddingů, jeden řádek = jedna věta."""
    client_response = client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
    #  print(client_response) pro debug, abych viděl, co se vrací   
    data = sorted(client_response.data, key=lambda item: item.index)
    embeddings = [item.embedding for item in data]
    return np.array(embeddings)