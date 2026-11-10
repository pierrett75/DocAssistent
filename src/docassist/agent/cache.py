import hashlib
import json
from collections.abc import Callable

from docassist.config import CACHE_DIR, PDF_PATH


def pdf_fingerprint() -> str:
    """Prvních 12 znaků SHA-256 z obsahu PDF. Jiné PDF = jiný otisk."""
    return hashlib.sha256(PDF_PATH.read_bytes()).hexdigest()[:12]


def short_hash(text: str) -> str:
    """Krátký hash textu, aby šel použít v názvu souboru (bez mezer a diakritiky)."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:8]


def cached(name: str, compute: Callable[[], str]) -> str:
    """Vrátí výsledek z cache. Když v cache není, zavolá compute(), uloží a vrátí.

    name    = co ukládám, např. "policy" nebo "summary_ab12cd34"
    compute = funkce BEZ argumentů, která výsledek spočítá (zavolá LLM)
    """
   
    path = CACHE_DIR / f"{name}_{pdf_fingerprint()}.json"
    if path.exists():
        print(f"  [cache HIT {name}]")
        return json.loads(path.read_text(encoding="utf-8"))
    else:
        print(f"  [cache MISS {name}]")
        result = compute()
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
        return result   