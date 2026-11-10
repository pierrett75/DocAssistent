"""
Týden 3, den 1 (den 15): REST API a FastAPI, endpointy /ask a /upload
Cvičení: z DocAssistu uděláš webovou službu. Agent z týdne 2 se nemění,
jen ho "obalíš" HTTP rozhraním, aby ho mohla volat jiná aplikace (web, Teams bot, ...).

Instalace:
    pip install fastapi uvicorn python-multipart
    (a přidej je do dependencies v pyproject.toml)
    python-multipart je potřeba kvůli nahrávání souborů (UploadFile).

Kam:      src/docassist/api/__init__.py   (prázdný)
          src/docassist/api/main.py       ← tento soubor
Spuštění (z kořene projektu):
          uvicorn docassist.api.main:app --reload
          → otevři v prohlížeči http://127.0.0.1:8000/docs  (Swagger: API si tam proklikáš a vyzkoušíš)
          --reload = po uložení souboru se server sám restartuje

"""




# ---------------------------------------------------------------------------
# Co vyzkoušet v http://127.0.0.1:8000/docs (tlačítko "Try it out"):
#
#   1. GET /health                                         → 200 {"status": "ok"}
#   2. POST /ask {"question": "Kolik mám sick days?"}       → 200, opiš si thread_id
#   3. POST /ask {"question": "A když nastoupím v srpnu?", "thread_id": "<to z bodu 2>"}
#                                                          → paměť funguje i přes API?
#   4. POST /ask {"question": ""}                          → jaký status kód? kdo ho vrátil?
#   5. POST /upload s .txt nebo .docx souborem              → 400
#   6. POST /upload s nějakým jiným PDF (třeba ze dne 4)    → 200, kolik chunků?
#      Pak se na jeho obsah zeptej přes /ask. Najde to search_documents?
#   7. POST /upload se STEJNÝM PDF podruhé                  → co se stane? (nápověda: ID chunků v Chroma)
#
# ---------------------------------------------------------------------------

#
# Otázky k zamyšlení (odpověz si do poznámek):
#   a) Proč je /ask POST, a ne GET? A proč je /health GET?
#   b) Co znamenají kódy 200, 400, 422 a 500? Který z nich vrací FastAPI samo? 500 = chyba serveru, 422 = validace vstupu, 400 = špatný požadavek, 200 = OK
#   c) Proč se agent vytváří jednou nahoře, a ne uvnitř ask()? protože pak by se paměť konverzací po každé otázce zahodila
#   d) Proč thread_id posílá klient? Kde by se v reálné aplikaci vzal? např. z cookies nebo authorization headeru
#   e) Nástroje extract a summarize pořád čtou jen PDF_PATH (směrnici o dovolené). 
#      Co to znamená pro nahrané dokumenty? Jak bys to v budoucnu řešila? omohl by parametr document u shrnutí. Extrakce má schéma jen pro dovolenou, takže jiný typ dokumentu potřebuje jiné schéma. Fulltext má ale smysl jako doplněk: hybrid search (BM25 + embeddingy) najde i přesné výrazy jako čísla předpisů.
# ---------------------------------------------------------------------------