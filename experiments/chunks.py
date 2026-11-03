from pathlib import Path

from pypdf import PdfReader

PDF_PATH = Path("C:\\Users\\pierr\\project\\DocAssistent\\data\\smernice_dovolena.pdf")  

CHUNK_SIZE = 800     # kolik znaků má jeden chunk
CHUNK_OVERLAP = 150  # kolik znaků se opakuje mezi sousedními chunky


def load_pdf(path: Path) -> list[dict]:
    """Načte PDF a vrátí seznam stran.

    Výstup vypadá takto:
    [
        {"text": "Text první strany...", "page": 1, "source": "dokument.pdf"},
        {"text": "Text druhé strany...", "page": 2, "source": "dokument.pdf"},
    ]
    """
    reader = PdfReader(path)
    pages = []
   
    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text()
        if not text or not text.strip():
            continue
        pages.append({"text": text, "page": page_number, "source": path.name})
    return pages


def chunk_text(text: str, size: int, overlap: int) -> list[str]:
    """Rozseká text na kousky délky `size`, sousední kousky se překrývají o `overlap` znaků.

    Příklad: text = "ABCDEFGHIJ", size = 4, overlap = 1
        krok (step) = size - overlap = 3
        chunk 1: text[0:4]  = "ABCD"
        chunk 2: text[3:7]  = "DEFG"     ← "D" se opakuje (overlap)
        chunk 3: text[6:10] = "GHIJ"     ← "G" se opakuje
        chunk 4: text[9:13] = "J"        ← zbytek, který už je celý v chunku 3
    Ten poslední drobek je běžný vedlejší efekt. Až to budeš mít hotové,
    zkus vymyslet, jak ho nevytvářet (bonus, není povinné).
    """
    
    step = size - overlap
    if step <= 0:
        raise ValueError("Overlap musí být menší než size, jinak by krok byl 0 nebo záporný.")
    
    return [text[i:i + size] for i in range(0, len(text)-overlap, step)]


def chunk_pages(pages: list[dict], size: int, overlap: int) -> list[dict]:
    """Rozseká každou stranu zvlášť a ke každému chunku přidá metadata.

    Výstup:
    [
        {"text": "...", "page": 1, "source": "dokument.pdf", "chunk_id": 0},
        {"text": "...", "page": 1, "source": "dokument.pdf", "chunk_id": 1},
        {"text": "...", "page": 2, "source": "dokument.pdf", "chunk_id": 2},
    ]
    Proč po stranách: díky tomu víme, z které strany chunk je, a odpověď
    pak může citovat "dokument.pdf, strana 2".
    """
    chunks = []

    for page in pages: 
        page_chunks = chunk_text(page["text"], size, overlap)
        for chunk in page_chunks:
            chunks.append({
                "text": chunk,
                "page": page["page"],
                "source": page["source"],
                "chunk_id": len(chunks) # len(chunks) funguje proto, že Python spočítá hodnotu slovníku dřív, než ho append přidá do seznamu. První chunk tak dostane 0 (seznam je prázdný), druhý 1 a tak dál.
            })
    return chunks


def print_stats(chunks: list[dict]) -> None:
    """Vypíše základní přehled o chuncích."""
   
    lengths = [len(chunk["text"]) for chunk in chunks]

    print(f"Počet chunků: {len(chunks)}")
    print(f"Průměrná délka: {sum(lengths) / len(lengths):.0f} znaků")
    print(f"Nejkratší chunk: {min(lengths)} znaků")
    
    for chunk in chunks:
        print(f"Chunk {chunk['chunk_id']}: {len(chunk['text'])} znaků, strana {chunk['page']}, zdroj {chunk['source']}")


if __name__ == "__main__":
    pages = load_pdf(PDF_PATH)
    print(f"Načteno stran s textem: {len(pages)}")
    print(f"Ukázka strany 1 (prvních 300 znaků):\n{pages[0]['text'][:300]}\n")

    chunks = chunk_pages(pages, CHUNK_SIZE, CHUNK_OVERLAP)
    print_stats(chunks)

    print("\n--- Chunk 0 ---")
    print(chunks[0])
    print("\n--- Chunk 1 ---")
    print(chunks[1])
    # Podívej se na konec chunku 0 a začátek chunku 1: měl by tam být stejný text (overlap).

    # TODO 5: Experiment. Pro velikosti 300, 800 a 1500 (overlap nech 150)
    #         vypiš, kolik chunků vznikne. Výsledek si zapiš do poznámek.

    # Otázky k zamyšlení (odpověz si do poznámek):
    # a) Podívej se, kde chunky začínají a končí. Co se stane se slovem nebo větou na hranici?
    # b) K čemu je overlap? Co by se stalo s větou, která leží přesně na hranici, bez něj?
    # c) Malé chunky vs. velké chunky: co je lepší pro přesné vyhledávání
    #    a co pro to, aby LLM dostal dost kontextu?
    # d) Proč je číslo strany v metadatech a ne v textu chunku?
    chunks2 =[] 
    for size in [300, 800, 1500]:
        chunks2.append(chunk_pages(pages, size, 150))
    print(f"Počet chunků pro velikosti 300, 800, 1500: {[len(c) for c in chunks2]}")