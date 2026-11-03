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
