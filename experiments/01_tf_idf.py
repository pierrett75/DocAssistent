"""
Exercise 1: TF-IDF from scratch
================================

Goal: implement TF-IDF scoring yourself and use it to rank documents.

Formulas used in this exercise:
    TF(term, doc)   = count of term in doc / number of words in doc
    IDF(term)       = log(N / df)
                      N  = number of documents in the corpus
                      df = number of documents that contain the term
    TF-IDF          = TF * IDF
    score(query, d) = sum of TF-IDF(term, d) for every term in the query

How to work:
    1. Fill in every function marked with TODO (replace `raise NotImplementedError`).
    2. Run:  python 01_tf_idf.py
    3. The self-checks at the bottom tell you what is still wrong.

Only the standard library is needed.
"""

import math
from collections import Counter

CORPUS = [
    "the cat sat on the mat",
    "the dog sat on the log",
    "cats and dogs are friends",
    "saffron is the most expensive spice in the world",
    "the cat chased the dog around the garden",
]


def tokenize(text: str) -> list[str]:
    """Given helper: lowercase and split on whitespace."""
    return text.lower().split()


DOCS = [tokenize(d) for d in CORPUS]


# ---------------------------------------------------------------------------
# TODO 1: term frequency
# ---------------------------------------------------------------------------
def term_frequency(term: str, doc: list[str]) -> float:
    """Return how often `term` appears in `doc`, divided by the length of `doc`.

    Example: term_frequency("the", ["the", "cat", "the"]) -> 2/3
    Hint: Counter or list.count() both work.
    """
    raise NotImplementedError


# ---------------------------------------------------------------------------
# TODO 2: document frequency and IDF
# ---------------------------------------------------------------------------
def document_frequency(term: str, docs: list[list[str]]) -> int:
    """Return the number of documents that contain `term` at least once."""
    raise NotImplementedError


def inverse_document_frequency(term: str, docs: list[list[str]]) -> float:
    """Return log(N / df). If df is 0 (term never appears), return 0.0.

    Think: why do we need to handle df == 0 separately?
    """
    raise NotImplementedError


# ---------------------------------------------------------------------------
# TODO 3: TF-IDF for one term in one document
# ---------------------------------------------------------------------------
def tf_idf(term: str, doc: list[str], docs: list[list[str]]) -> float:
    """Return TF * IDF. Reuse your functions from TODO 1 and 2."""
    raise NotImplementedError


# ---------------------------------------------------------------------------
# TODO 4: search
# ---------------------------------------------------------------------------
def search(query: str, docs: list[list[str]], top_k: int = 3) -> list[tuple[int, float]]:
    """Score every document against the query and return the best ones.

    Return a list of (doc_index, score) tuples, sorted by score (highest first),
    containing at most `top_k` items. Skip documents with score 0.
    """
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Self-checks (do not edit)
# ---------------------------------------------------------------------------
def _close(a, b, tol=1e-6):
    return abs(a - b) < tol


def run_checks():
    checks = [
        ("TF of 'the' in doc 0", lambda: _close(term_frequency("the", DOCS[0]), 2 / 6)),
        ("TF of missing word", lambda: term_frequency("zebra", DOCS[0]) == 0),
        ("DF of 'the'", lambda: document_frequency("the", DOCS) == 4),
        ("DF of 'saffron'", lambda: document_frequency("saffron", DOCS) == 1),
        ("IDF of 'the'", lambda: _close(inverse_document_frequency("the", DOCS), math.log(5 / 4))),
        ("IDF of 'saffron'", lambda: _close(inverse_document_frequency("saffron", DOCS), math.log(5))),
        ("IDF of missing word", lambda: inverse_document_frequency("zebra", DOCS) == 0.0),
        ("rare word beats common word",
         lambda: tf_idf("saffron", DOCS[3], DOCS) > tf_idf("the", DOCS[3], DOCS)),
        ("search 'cat' finds doc 0 first", lambda: search("cat", DOCS)[0][0] == 0),
        ("search 'expensive spice' -> doc 3", lambda: search("expensive spice", DOCS)[0][0] == 3),
        ("search respects top_k", lambda: len(search("the", DOCS, top_k=2)) <= 2),
        ("search skips zero scores", lambda: search("zebra", DOCS) == []),
    ]
    passed = 0
    for name, check in checks:
        try:
            ok = check()
        except NotImplementedError:
            ok, name = False, name + "  (not implemented yet)"
        except Exception as e:  # noqa: BLE001
            ok, name = False, f"{name}  (error: {e!r})"
        print(("PASS " if ok else "FAIL ") + name)
        passed += ok
    print(f"\n{passed}/{len(checks)} checks passed")


if __name__ == "__main__":
    run_checks()

    # ---------------------------------------------------------------------
    # THINK ABOUT IT (no code needed, answer in a comment):
    # 1. Search for "the cat". Which word contributes more to the score, and why?
    # 2. Why does doc 2 ("cats and dogs are friends") NOT match the query "cat"?
    #    What would you need to fix that? (Hint: stemming / lemmatization.)
    # ---------------------------------------------------------------------
