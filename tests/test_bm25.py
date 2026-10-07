import math
import random

import pytest

from hybrid_retrieval import BM25Index, tokenize

DOCS = {
    "d1": "the cat sat on the mat",
    "d2": "the dog chased the cat",
    "d3": "dogs and cats are pets",
    "d4": "quantum computing uses qubits",
}


def brute_force(
    docs: dict[str, str], query: str, k1: float = 1.2, b: float = 0.75
) -> dict[str, float]:
    """Reference BM25 written directly from the formula, with no index."""
    tokens = {doc_id: tokenize(text) for doc_id, text in docs.items()}
    n = len(docs)
    avg_len = sum(len(t) for t in tokens.values()) / n
    out: dict[str, float] = {}
    for doc_id, toks in tokens.items():
        score = 0.0
        for term in set(tokenize(query)):
            tf = toks.count(term)
            if tf == 0:
                continue
            df = sum(1 for t in tokens.values() if term in t)
            idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
            score += idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * len(toks) / avg_len))
        out[doc_id] = score
    return out


def test_scores_match_brute_force() -> None:
    index = BM25Index(list(DOCS), list(DOCS.values()))
    for query in ["cat", "the cat", "dog cat pets", "qubits", "cat cat cat"]:
        expected = brute_force(DOCS, query)
        got = dict(zip(index.doc_ids, index.scores(query), strict=True))
        for doc_id in DOCS:
            assert got[doc_id] == pytest.approx(expected[doc_id], rel=1e-5, abs=1e-6)


def test_scores_match_brute_force_on_random_corpus() -> None:
    rng = random.Random(0)
    vocab = [f"w{i}" for i in range(50)]
    docs = {
        f"d{i}": " ".join(rng.choices(vocab, k=rng.randint(1, 40))) for i in range(200)
    }
    index = BM25Index(list(docs), list(docs.values()))
    for _ in range(20):
        query = " ".join(rng.choices(vocab, k=rng.randint(1, 5)))
        expected = brute_force(docs, query)
        got = dict(zip(index.doc_ids, index.scores(query), strict=True))
        for doc_id in docs:
            assert got[doc_id] == pytest.approx(expected[doc_id], rel=1e-4, abs=1e-5)


def test_search_orders_by_score_and_drops_non_matches() -> None:
    index = BM25Index(list(DOCS), list(DOCS.values()))
    results = index.search("cat", k=10)
    # "cat" is in d1 and d2 only. d2 is shorter (5 tokens against 6), so it scores higher.
    assert [doc_id for doc_id, _ in results] == ["d2", "d1"]
    assert results[0][1] > results[1][1] > 0


def test_search_respects_k() -> None:
    index = BM25Index(list(DOCS), list(DOCS.values()))
    assert len(index.search("the cat dog", k=1)) == 1


def test_unknown_term_and_empty_index() -> None:
    assert BM25Index(list(DOCS), list(DOCS.values())).search("zebra") == []
    assert BM25Index([], []).search("cat") == []


def test_mismatched_lengths_raise() -> None:
    with pytest.raises(ValueError):
        BM25Index(["d1"], [])
