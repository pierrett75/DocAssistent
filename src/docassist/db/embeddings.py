import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

from docassist.llm import embedding_model_name, get_openai_client

load_dotenv()
client = get_openai_client()


def get_embeddings(texts: list[str]) -> np.ndarray:
    """Vrátí matici embeddingů, jeden řádek = jedna věta."""
    client_response = client.embeddings.create(model=embedding_model_name(), input=texts)
    #  print(client_response) pro debug, abych viděl, co se vrací   
    data = sorted(client_response.data, key=lambda item: item.index)
    embeddings = [item.embedding for item in data]
    return np.array(embeddings)