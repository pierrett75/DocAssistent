"""Jedno místo pro všechna nastavení. Ostatní moduly si odsud berou hodnoty,
takže když chceš změnit TOP_K nebo model, měníš to jen tady."""

from pathlib import Path

from dotenv import load_dotenv

# Cesta ke kořeni projektu, spočítaná z umístění tohoto souboru:
#   .../DocAssistent/src/docassist/config.py
#   parents[0] = docassist, parents[1] = src, parents[2] = DocAssistent
PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")

# Cesty
DATA_DIR = PROJECT_ROOT / "data"
CHROMA_DIR = DATA_DIR / "chroma_db"
COLLECTION_NAME = "docassist"

# Chunking (den 4)
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150

# Modely (dny 3 a 6)
EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"

# Retrieval a RAG (dny 5 a 6)
TOP_K = 8
MIN_SIMILARITY = 0.5

PDF_PATH = DATA_DIR / "smernice_dovolena.pdf"

# kolik tokenů historie posílám modelu
MAX_HISTORY_TOKENS = 4000   

CACHE_DIR = DATA_DIR / "cache"

RECURSION_LIMIT = 10

MAX_UPLOAD_MB = 10