import pytest

from docassist.ingest.chunking import chunk_pages, chunk_text
from experiments import chunks


def test_chunk_text_priklad_z_docstringu():
    # Arrange
    text = "ABCDEFGHIJ"
    # Act
    chunks = chunk_text(text, size=4, overlap=1)
    # Assert
    assert chunks == ["ABCD", "DEFG", "GHIJ"]   


def test_chunk_text_odmitne_overlap_vetsi_nez_size():
    
    with pytest.raises(ValueError):
        chunk_text("cokoli", size=4, overlap=4)


# parametrize = stejný test pro víc vstupů. Pytest z toho udělá 3 samostatné testy.
@pytest.mark.parametrize("size, overlap", [(10, 2), (50, 10), (800, 150)])
def test_chunk_text_sousedni_chunky_se_prekryvaji(size, overlap):
    text = "Zaměstnanec má nárok na 25 dní dovolené. " * 100     # dost dlouhý text
    chunks = chunk_text(text, size, overlap)

    assert all(len(c) <= size for c in chunks)
    for prvni, druhy in zip(chunks, chunks[1:]):
        assert prvni[-overlap:] == druhy[:overlap]


def test_chunk_text_kratky_text_da_jeden_chunk():
    
    chunks = chunk_text("Příloha 1", size=800, overlap=150)
    assert chunks == ["Příloha 1"]


def test_chunk_pages_cisluje_chunky_pres_vsechny_strany():
    pages = [
        {"text": "A" * 25, "page": 1, "source": "smernice.pdf"},
        {"text": "B" * 25, "page": 2, "source": "smernice.pdf"},
    ]
    chunks = chunk_pages(pages, size=10, overlap=0)

   
    assert [c["chunk_id"] for c in chunks] == list(range(len(chunks)))
    for c in chunks[3:]:                 
       assert c["page"] == 2
       assert set(c["text"]) == {"B"}    

    for c in chunks:
        assert c["source"] == "smernice.pdf"
