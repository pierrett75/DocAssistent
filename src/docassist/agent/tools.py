import json

from langchain_core.tools import tool
from openai import OpenAI

from docassist.agent.cache import cached, short_hash
from docassist.agent.prompts import SUMMARY_PROMPT
from docassist.config import CHAT_MODEL, MIN_SIMILARITY, TOP_K, PDF_PATH
from docassist.db.vector_store import get_collection, retrieve
from docassist.extract.company_rule import extract_policy
from docassist.ingest.pdf_loader import load_pdf
from docassist.rag.pipeline import format_context

client = OpenAI()

def load_full_text() -> str:
    pages = load_pdf(PDF_PATH)
    return "\n\n".join(f"[strana {p['page']}]\n{p['text']}" for p in pages)

@tool
def search_documents(query: str) -> str:
    """Vyhledá v interních směrnicích pasáže k jednomu pravidlu, postupu nebo podmínce.
    Použij i pro otázku na jedno číslo (např. kolik mám sick days) a i když si myslíš,
    že odpověď v dokumentech není. Vrací pasáže s čísly [n] a stranou.
    query: dotaz česky, slovy, jaká mohou být v dokumentu."""
    results = retrieve(get_collection(), query, k=TOP_K)
    if not results or results[0]["similarity"] < MIN_SIMILARITY:
        return "Nic relevantního nenalezeno."      
    return format_context(results)                



@tool
def extract_vacation_policy() -> str:
    """Vytáhne ze směrnice o dovolené všechna čísla najednou a dopočítá dny navíc.
    Použij JEN pro přehled více čísel nebo pro výpočet, např.
    kolik dní dovolené mám navíc nad zákonné minimum' nebo 'dej mi přehled čísel o dovolené'.
    Na otázku na jedno pravidlo nebo podmínku (např. sick days, převod dovolené) použij search_documents."""

    def compute() -> str:                      # funkce uvnitř funkce: cache ji zavolá jen při MISS
            data_policy = extract_policy(load_full_text())
            data = data_policy.model_dump()
            data["extra_vacation_days"] = data_policy.extra_vacation_days()
            return json.dumps(data, ensure_ascii=False)

    return cached("policy", compute)

@tool
def summarize_document(topic: str) -> str:
    """Shrne směrnici o dovolené a sick days do přehledných odrážek se stranami.
    Použij, když uživatel chce shrnutí, přehled pravidel nebo "o čem směrnice je".
    topic: téma shrnutí (např. "sick days", "převod dovolené"), nebo "celý dokument"."""

    def compute() -> str:
            response = client.chat.completions.create(
                model=CHAT_MODEL,
                temperature=0,
                messages=[
                    {"role": "system", "content": SUMMARY_PROMPT},
                    {"role": "user", "content": f"<dokument>\n{load_full_text()}\n</dokument>\n\nTéma: {topic}"},
                ],
            )
            return response.choices[0].message.content

    return cached(f"summary_{short_hash(topic.lower().strip())}", compute).lower().strip()

TOOLS = [search_documents, extract_vacation_policy, summarize_document]