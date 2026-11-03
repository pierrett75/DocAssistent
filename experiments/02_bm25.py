"""
Exercise 2: BM25 from scratch
=============================

Goal: implement BM25 and see how it fixes two weaknesses of TF-IDF:
    - term-frequency saturation (repeating a word gives diminishing returns)
    - document-length normalization (long docs don't win just by being long)

BM25 formula for one query term t in document d:

                         tf * (k1 + 1)
    score(t, d) = IDF(t) * ---------------------------------------
                         tf + k1 * (1 - b + b * len(d) / avgdl)

    IDF(t) = log( (N - df + 0.5) / (df + 0.5) + 1 )     # Lucene/Elasticsearch variant

    tf     = raw count of t in d (NOT divided by length - BM25 handles length itself)
    avgdl  = average document length in the corpus
    k1     = controls saturation        (typical 1.2 - 2.0)
    b      = controls length normalization (0 = none, 1 = full, typical 0.75)

    score(query, d) = sum of score(t, d) over query terms

How to work:
    1. Fill in every TODO.
    2. Run:  python 02_bm25.py
    3. Then do the experiments at the bottom.

Only the standard library is needed.
"""

import math
from collections import Counter

CORPUS = [
    "python is a programming language",
    "python python python python python python python python",
    "the python snake is a large reptile found in asia and africa it eats small animals",
    "java is also a programming language used in enterprise software",
    "learn python programming with simple examples",
]


def tokenize(text: str) -> list[str]:
    return text.lower().split()


class BM25:
    def __init__(self, corpus: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.docs = [tokenize(d) for d in corpus]
        self.N = len(self.docs)

        # -------------------------------------------------------------------
        # TODO 1: precompute corpus statistics
        # -------------------------------------------------------------------
        # self.doc_lens : list with the length (word count) of every document
        # self.avgdl    : average document length
        # self.tf       : list of Counter objects, one per document (word -> count)
        # self.df       : Counter with, for each word, the number of docs containing it
        raise NotImplementedError

    # -----------------------------------------------------------------------
    # TODO 2: IDF
    # -----------------------------------------------------------------------
    def idf(self, term: str) -> float:
        """Return log((N - df + 0.5) / (df + 0.5) + 1).

        Note: for a term that appears nowhere, df = 0 and that's fine here.
        """
        raise NotImplementedError

    # -----------------------------------------------------------------------
    # TODO 3: score of one term in one document
    # -----------------------------------------------------------------------
    def term_score(self, term: str, doc_index: int) -> float:
        """Apply the BM25 formula from the top of the file. Return 0.0 if tf == 0."""
        raise NotImplementedError

    # -----------------------------------------------------------------------
    # TODO 4: score a whole query, then rank
    # -----------------------------------------------------------------------
    def score(self, query: str, doc_index: int) -> float:
        """Sum term_score over all tokens in the query."""
        raise NotImplementedError

    def search(self, query: str, top_k: int = 3) -> list[tuple[int, float]]:
        """Return up to top_k (doc_index, score) tuples, best first, skipping score 0."""
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Self-checks (do not edit)
# ---------------------------------------------------------------------------
def _close(a, b, tol=1e-4):
    return abs(a - b) < tol


def run_checks():
    def bm():
        return BM25(CORPUS)

    checks = [
        ("avgdl", lambda: _close(bm().avgdl, (5 + 8 + 16 + 10 + 6) / 5)),
        ("df of 'python'", lambda: bm().df["python"] == 4),
        ("tf of 'python' in doc 1", lambda: bm().tf[1]["python"] == 8),
        ("idf of 'python'", lambda: _close(bm().idf("python"), math.log(1.5 / 4.5 + 1))),
        ("idf of 'java'", lambda: _close(bm().idf("java"), math.log(4.5 / 1.5 + 1))),
        ("rare term has higher idf", lambda: bm().idf("java") > bm().idf("python")),
        ("term_score is 0 for missing term", lambda: bm().term_score("java", 0) == 0.0),
        ("term_score known value",
         lambda: _close(bm().term_score("java", 3), 1.3203, tol=1e-3)),
        ("saturation: 8x 'python' scores < 8x a single 'python'",
         lambda: bm().term_score("python", 1) < 8 * bm().term_score("python", 0)),
        ("search 'java' -> doc 3", lambda: bm().search("java")[0][0] == 3),
        ("search 'python programming' -> doc 0 first",
         lambda: bm().search("python programming")[0][0] == 0),
        ("search skips zero scores", lambda: bm().search("zebra") == []),
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
    # EXPERIMENTS (once all checks pass)
    # ---------------------------------------------------------------------
    # TODO 5: Saturation. Print term_score("python", i) for every doc.
    #         Doc 1 repeats "python" 8 times. How much higher does it score than doc 0?
    #         Now try k1=0.1 and k1=10. What happens to the gap, and why?
    #
    # TODO 6: Length normalization. Compare the score of "python" in doc 2 (long)
    #         with b=0.0 and b=1.0. Which setting punishes long documents?
    #
    # TODO 7: Compare with your TF-IDF from exercise 1 on this CORPUS.
    #         Which document does TF-IDF rank first for "python"? Is that a good result?
    #
    # BONUS:  pip install rank_bm25 and compare your scores with BM25Okapi.
    #         They will differ slightly - find out why by reading its idf formula.
    # ---------------------------------------------------------------------
