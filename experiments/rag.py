from dotenv import load_dotenv
from openai import OpenAI

from retrieval import get_collection, retrieve 

load_dotenv()
client = OpenAI()

CHAT_MODEL = "gpt-4o-mini"   
TOP_K = 8
MIN_SIMILARITY = 0.5         # pod touto hranicí bereme výsledek jako "nic relevantního" 
FALLBACK = "V dostupných dokumentech jsem odpověď nenašla."

SYSTEM_PROMPT = """Jsi asistent, který odpovídá na otázky zaměstnanců podle firemních dokumentů.

Pravidla:
- Odpovídej POUZE z textu v sekci <kontext>. Nepoužívej vlastní znalosti.
- Za každé tvrzení dej odkaz na zdroj v hranatých závorkách, např. [1] nebo [2].
- Pokud odpověď v kontextu není, odpověz přesně: "V dostupných dokumentech jsem odpověď nenašla."
- Odpovídej česky, stručně, maximálně 4 věty.
- Pokud kontext obsahuje obecné pravidlo, které se na otázku vztahuje, odpověz podle něj a řekni, že jde o obecné pravidlo.
- Pokud kontext obsahuje pravidlo, které se na situaci z otázky vztahuje, odpověz podle něj, i když tam situace není popsaná doslova.
- Větu "V dostupných dokumentech jsem odpověď nenašla." použij JEN tehdy, když kontext s otázkou vůbec nesouvisí.
"""


def format_context(results: list[dict]) -> str:
    """Z výsledků retrievalu udělá jeden text s očíslovanými zdroji.

    Výstup má vypadat takto:
        [1] (dokument.pdf, strana 2)
        Text prvního chunku...

        [2] (dokument.pdf, strana 4)
        Text druhého chunku...
    """
    blocks = []  
    
    for i, result in enumerate(results, start=1):  
        block = f"[{i}] ({result['source']}, strana {result['page']})\n{result['text']}"
        blocks.append(block)
    return "\n\n".join(blocks)

def build_messages(question: str, context: str) -> list[dict]:
    """Složí zprávy pro chat API: system prompt + uživatelská zpráva s kontextem."""

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"<kontext>\n{context}\n</kontext>\n\n<otazka>\n{question}\n</otazka>"}
    ]


def answer(question: str) -> dict:
    """Celý RAG: vrátí {"answer": text odpovědi, "sources": seznam citací}."""
    collection = get_collection()
    results = retrieve(collection, question, k=TOP_K)

   
    if not results or results[0]["similarity"] < MIN_SIMILARITY:
        return {"answer": FALLBACK, "sources": []}
    
    context = format_context(results)
    messages = build_messages(question, context)

    
    response = client.chat.completions.create(
        model=CHAT_MODEL, messages=messages, temperature=0)  
    
    answer_text = response.choices[0].message.content
    
    sources = [f"[{i}] {result['source']}, strana {result['page']}" for i, result in enumerate(results, start=1)]

    if answer_text == FALLBACK:
                sources = []
                
    return {"answer": answer_text, "sources": sources}
    


if __name__ == "__main__":
    questions = [
        "Kolik dní dovolené mají zaměstnanci?",
        "Do kdy musím požádat o převedení 5. týdne dovolené?",
        "Kolik mám sick days?",
        "Můžu si vzít sick day v pátek před dovolenou?",
        "Kolik dní dovolené navíc mají zaměstnanci pobočky v Brně?",
        "Kolik bylo podáno odvolání?",                 # v dokumentu NENÍ → má přijít FALLBACK
        "Jaké je heslo na firemní wifi?",              # v dokumentu NENÍ
    ]

    collection = get_collection()
    for r in retrieve(collection, "Kolik dní dovolené mají zaměstnanci?", k=10):
        print(f"strana {r['page']}  {r['similarity']:.2f}  {r['text'][:80]}")

    for q in questions:
        result = answer(q)
        print(f"Otázka: {q}")
        print(f"Odpověď: {result['answer']}")
        for source in result["sources"]:
            print(f"   {source}")
        print()

    # TODO 4: Ověř odpovědi! Otevři PDF a u každé odpovědi zkontroluj:
    #         - Je pravdivá podle dokumentu?
    #         - Sedí citace [n] na stranu, kde to v PDF opravdu je?
    #         - Nespletl si model dovolenou se sick days (vzpomeň si na den 1)?
    #         Výsledek si zapiš jako tabulku: otázka | správně ano/ne | poznámka.
    #         Tohle je zárodek evaluační sady z týdne 3.

    # Otázky k zamyšlení (odpověz si do poznámek):
    # a) Které dvě ochrany proti halucinaci v tomto skriptu máš? Kde každá zasáhne?
    # b) Co se stane, když správná odpověď leží ve 4. nejpodobnějším chunku a TOP_K = 3?
    # c) Proč kontext číslujeme [1], [2] a nepíšeme rovnou "strana 2" do odpovědi?
    # d) Kdo zaručí, že model citaci [n] nevymyslel? (Nápověda: zatím nikdo, to je týden 3.)
