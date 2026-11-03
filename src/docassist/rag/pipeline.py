"""Celý RAG: retrieval → kontext → LLM → odpověď s citací (den 6)."""

from openai import OpenAI

from docassist.config import CHAT_MODEL, MIN_SIMILARITY, TOP_K
from docassist.db.vector_store import get_collection, retrieve
from docassist.rag.prompts import FALLBACK, SYSTEM_PROMPT

client = OpenAI()


def format_context(results: list[dict]) -> str:
    blocks = []
    for i, result in enumerate(results, start=1):
        blocks.append(f"[{i}] ({result['source']}, strana {result['page']})\n{result['text']}")
    return "\n\n".join(blocks)


def build_messages(question: str, context: str) -> list[dict]:
    user_content = f"""<kontext>
{context}
</kontext>

<otazka>
{question}
</otazka>"""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def answer(question: str) -> dict:
    """Vrátí {"answer": text, "sources": seznam citací}."""
    collection = get_collection()
    results = retrieve(collection, question, k=TOP_K)       

    if not results or results[0]["similarity"] < MIN_SIMILARITY:
        return {"answer": FALLBACK, "sources": []}

    context = format_context(results)
    messages = build_messages(question, context)
    response = client.chat.completions.create(
        model=CHAT_MODEL, messages=messages, temperature=0
    )
    answer_text = response.choices[0].message.content

    sources = [
        f"[{i}] {r['source']}, strana {r['page']}" for i, r in enumerate(results, start=1)
    ]
    if answer_text.strip() == FALLBACK:                     
        sources = []

    return {"answer": answer_text, "sources": sources}