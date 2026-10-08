import math
import random

import pytest

from hybrid_retrieval import BM25Index, tokenize
from hybrid_retrieval.bm25 import Method

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


def random_corpus(seed: int, num_docs: int) -> dict[str, str]:
    rng = random.Random(seed)
    # "the" is in almost every document, like a stopword. The other words are rare.
    vocab = [f"w{i}" for i in range(300)]
    return {
        f"d{i}": " ".join(["the"] * rng.randint(0, 3) + rng.choices(vocab, k=rng.randint(1, 30)))
        for i in range(num_docs)
    }


@pytest.mark.parametrize("method", ["daat", "maxscore"])
@pytest.mark.parametrize("k", [1, 10, 100])
def test_compiled_search_returns_the_exhaustive_top_k(method: Method, k: int) -> None:
    docs = random_corpus(seed=1, num_docs=2000)
    index = BM25Index(list(docs), list(docs.values()))
    rng = random.Random(2)
    for _ in range(50):
        query = " ".join(["the"] + rng.choices([f"w{i}" for i in range(300)], k=rng.randint(1, 6)))
        expected = index.search(query, k=k)
        got = index.search(query, k=k, method=method)
        # The k best scores are the same.
        assert [s for _, s in got] == pytest.approx([s for _, s in expected], rel=1e-5)
        # Each returned document really has the score it was returned with. Together with
        # the line above, this proves a correct top k. The document lists themselves can
        # differ only where two scores are equal to within float rounding, because the
        # methods add the same numbers in a different order.
        truth = dict(zip(index.doc_ids, index.scores(query), strict=True))
        for doc_id, score in got:
            assert score == pytest.approx(truth[doc_id], rel=1e-5)


def test_maxscore_scores_fewer_postings_than_exhaustive_daat() -> None:
    docs = random_corpus(seed=3, num_docs=5000)
    index = BM25Index(list(docs), list(docs.values()))
    query = "the w1 w2 w3"
    index.search(query, k=10, method="daat")
    exhaustive = index.last_postings_scored
    index.search(query, k=10, method="maxscore")
    pruned = index.last_postings_scored
    # The exhaustive walk reads every posting of every query term.
    assert exhaustive == index.num_postings(query)
    # "the" has a low upper bound, so MaxScore stops walking its long list.
    assert pruned < exhaustive / 2


def test_compiled_search_edge_cases() -> None:
    index = BM25Index(list(DOCS), list(DOCS.values()))
    for method in ("daat", "maxscore"):
        assert index.search("zebra", method=method) == []
        assert [d for d, _ in index.search("cat", k=10, method=method)] == ["d2", "d1"]
        assert len(index.search("the cat dog", k=1, method=method)) == 1
        assert BM25Index([], []).search("cat", method=method) == []


def test_ties_break_by_document_order_in_every_method() -> None:
    # Three identical documents have identical scores.
    index = BM25Index(["a", "b", "c"], ["cat dog", "cat dog", "cat dog"])
    for method in ("exhaustive", "daat", "maxscore"):
        assert [d for d, _ in index.search("cat dog", k=2, method=method)] == ["a", "b"]
