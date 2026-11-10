def chunk_text(text: str, size: int, overlap: int) -> list[str]:
    """Rozseká text na kousky (chunky) o dané velikosti a překryvu."""
    step = size - overlap
    if step <= 0:
            raise ValueError("Overlap musí být menší než size, jinak by krok byl 0 nebo záporný.")
    
    end_range = max(len(text) - overlap, 1)
   
    return [text[i:i + size] for i in range(0, end_range, step)]
    


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
