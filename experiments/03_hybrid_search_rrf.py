"""
Exercise 3: Hybrid search with Reciprocal Rank Fusion (RRF)
===========================================================

Goal: combine a keyword retriever (BM25) with a semantic retriever (embeddings)
using Reciprocal Rank Fusion - the standard trick in production RAG.

Why fuse rankings instead of scores?
    BM25 scores might be 0-15, cosine similarities 0-1. They are not comparable.
    RRF ignores the raw scores and only uses each document's RANK in each list:

        RRF(d) = sum over retrievers r of   1 / (k + rank_r(d))

        rank starts at 1 (best). k is usually 60. A document missing from a list
        contributes nothing from that list.

To keep this exercise dependency-free, the two retrievers' result lists are
given below (as if they came from BM25 and a vector database).

How to work:
    1. Fill in every TODO.
    2. Run:  python 03_hybrid_search_rrf.py
"""

DOCS = {
    "d1": "Error ERR-4012: connection timeout when calling the payments API.",
    "d2": "How to fix network timeouts in our payment service.",
    "d3": "Troubleshooting slow responses from external services.",
    "d4": "ERR-4012 appears in logs after the nightly batch job.",
    "d5": "Guide to retry policies for unreliable HTTP calls.",
    "d6": "Office holiday schedule for December.",
}

QUERY = "ERR-4012 payment timeout fix"

# BM25 is great at the exact code "ERR-4012", but misses paraphrases.
BM25_RESULTS = ["d4", "d1", "d6", "d2"]

# Embeddings understand meaning ("slow responses" ~ "timeout"), but ignore the code.
DENSE_RESULTS = ["d2", "d1", "d3", "d5"]


# ===========================================================================
# PART A: where do DENSE_RESULTS come from? Cosine similarity!
# ===========================================================================
# Real embedding models output 384-3072 numbers per text. Here we use tiny,
# hand-made 5-number "embeddings" so you can reason about them. Each position
# roughly stands for a topic:
#
#       [payments, timeout/slowness, network/services, error codes, office/HR]
#
QUERY_EMBEDDING = [0.6, 0.7, 0.3, 0.1, 0.0]

DOC_EMBEDDINGS = {
    "d1": [0.5, 0.6, 0.2, 0.6, 0.0],
    "d2": [0.6, 0.7, 0.35, 0.0, 0.0],
    "d3": [0.1, 0.7, 0.6, 0.0, 0.0],
    "d4": [0.0, 0.1, 0.2, 0.9, 0.0],
    "d5": [0.0, 0.4, 0.7, 0.0, 0.0],
    "d6": [0.0, 0.0, 0.0, 0.0, 1.0],
}


# ---------------------------------------------------------------------------
# TODO A1: dot product and vector length
# ---------------------------------------------------------------------------
def dot(a: list[float], b: list[float]) -> float:
    """Multiply matching positions and add them up: a[0]*b[0] + a[1]*b[1] + ..."""
    raise NotImplementedError


def norm(a: list[float]) -> float:
    """Length of the vector: square root of (a[0]^2 + a[1]^2 + ...).
    Hint: you can reuse dot()."""
    raise NotImplementedError


# ---------------------------------------------------------------------------
# TODO A2: cosine similarity
# ---------------------------------------------------------------------------
def cosine_similarity(a: list[float], b: list[float]) -> float:
    """cos = dot(a, b) / (norm(a) * norm(b)).
    If either vector has length 0, return 0.0 (avoid dividing by zero)."""
    raise NotImplementedError


# ---------------------------------------------------------------------------
# TODO A3: dense search
# ---------------------------------------------------------------------------
def dense_search(query_vec: list[float], doc_vecs: dict[str, list[float]], top_k: int = 4) -> list[str]:
    """Return the ids of the top_k documents with the highest cosine similarity
    to the query, best first. Example output: ["d2", "d1", ...]"""
    raise NotImplementedError


# ===========================================================================
# PART B: fuse BM25 + dense results with RRF
# ===========================================================================


# ---------------------------------------------------------------------------
# TODO 1: rank lookup
# ---------------------------------------------------------------------------
def ranks(result_list: list[str]) -> dict[str, int]:
    """Turn ["d4", "d1"] into {"d4": 1, "d1": 2}. Ranks start at 1."""
    raise NotImplementedError


# ---------------------------------------------------------------------------
# TODO 2: RRF fusion
# ---------------------------------------------------------------------------
def reciprocal_rank_fusion(result_lists: list[list[str]], k: int = 60) -> list[tuple[str, float]]:
    """Fuse any number of ranked lists.

    Return a list of (doc_id, rrf_score) sorted by score, highest first.
    Tie-break by doc_id alphabetically so the result is deterministic.
    Hint: use your ranks() function and a dict to accumulate scores.
    """
    raise NotImplementedError


# ---------------------------------------------------------------------------
# TODO 3: weighted RRF
# ---------------------------------------------------------------------------
def weighted_rrf(result_lists: list[list[str]], weights: list[float], k: int = 60) -> list[tuple[str, float]]:
    """Same as RRF, but each retriever's contribution is multiplied by its weight.

    Use case: trust one retriever more, e.g. BM25 for queries with error codes or IDs.
    """
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Self-checks (do not edit)
# ---------------------------------------------------------------------------
def _close(a, b, tol=1e-9):
    return abs(a - b) < tol


def run_checks():
    def fused():
        return reciprocal_rank_fusion([BM25_RESULTS, DENSE_RESULTS])

    checks = [
        # Part A: cosine similarity
        ("dot product", lambda: dot([1, 2, 3], [4, 5, 6]) == 32),
        ("norm", lambda: _close(norm([3, 4]), 5.0)),
        ("cosine of identical vectors = 1", lambda: _close(cosine_similarity([1, 2], [1, 2]), 1.0)),
        ("cosine ignores length", lambda: _close(cosine_similarity([1, 2], [10, 20]), 1.0)),
        ("cosine of perpendicular vectors = 0", lambda: _close(cosine_similarity([1, 0], [0, 1]), 0.0)),
        ("cosine with zero vector = 0", lambda: cosine_similarity([0, 0], [1, 1]) == 0.0),
        ("cosine of d2 vs query", lambda: _close(cosine_similarity(QUERY_EMBEDDING, DOC_EMBEDDINGS["d2"]), 0.99358, tol=1e-4)),
        ("dense_search reproduces DENSE_RESULTS",
         lambda: dense_search(QUERY_EMBEDDING, DOC_EMBEDDINGS) == DENSE_RESULTS),
        ("dense_search respects top_k", lambda: len(dense_search(QUERY_EMBEDDING, DOC_EMBEDDINGS, top_k=2)) == 2),
        # Part B: RRF
        ("ranks start at 1", lambda: ranks(["a", "b"]) == {"a": 1, "b": 2}),
        ("ranks of empty list", lambda: ranks([]) == {}),
        ("d1 RRF score", lambda: _close(dict(fused())["d1"], 1 / 62 + 1 / 62)),
        ("d3 only in dense list", lambda: _close(dict(fused())["d3"], 1 / 63)),
        ("all retrieved docs present", lambda: {d for d, _ in fused()} == {"d1", "d2", "d3", "d4", "d5", "d6"}),
        ("d1 wins (high in BOTH lists)", lambda: fused()[0][0] == "d1"),
        ("sorted descending", lambda: [s for _, s in fused()] == sorted([s for _, s in fused()], reverse=True)),
        ("weights 1,1 == plain RRF",
         lambda: weighted_rrf([BM25_RESULTS, DENSE_RESULTS], [1, 1]) == fused()),
        ("heavy dense weight puts d2 on top",
         lambda: weighted_rrf([BM25_RESULTS, DENSE_RESULTS], [1, 3])[0][0] == "d2"),
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
    # THINK ABOUT IT:
    # 0. (Part A) Print the cosine similarity of every doc vs the query.
    #    - Why is d4 (which contains the exact code ERR-4012!) ranked so low?
    #    - d6 scores exactly 0. Look at the vectors: why?
    #    - Try cosine_similarity([1, 1], [100, 100]). Why is this "length doesn't
    #      matter" property useful for comparing a short query with a long document?
    # 1. d1 was only #2 in BM25 and #2 in dense, yet it wins overall. Why is that
    #    usually a good thing in RAG?
    # 2. Change k from 60 to 1. How does the ranking change? What does k control?
    # 3. d6 (holiday schedule) sneaks into the BM25 list - imagine it matched only
    #    a common word. Why can irrelevant docs still get a small BM25 score?
    #    What step after fusion would push it out? (Hint: reranker / cross-encoder.)
    #
    # BONUS (needs pip installs):
    #   - Replace BM25_RESULTS with your BM25 class from exercise 2 run on DOCS.
    #   - Replace DENSE_RESULTS with real embeddings, e.g.
    #     sentence-transformers "all-MiniLM-L6-v2" + cosine similarity.
    #   - Does the fused result change? Notice how tokenization of "ERR-4012."
    #     (with the dot!) affects BM25.
    # ---------------------------------------------------------------------
