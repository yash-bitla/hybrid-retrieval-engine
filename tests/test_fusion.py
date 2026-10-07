import pytest

from hybrid_retrieval import HybridRetriever, rrf, weighted

LEXICAL = [("a", 12.0), ("b", 9.0), ("c", 3.0)]
DENSE = [("b", 0.9), ("d", 0.8), ("a", 0.5)]


def test_rrf_hand_computed() -> None:
    # With k = 60:
    # a: rank 1 and rank 3 -> 1/61 + 1/63 = 0.016393 + 0.015873 = 0.032266
    # b: rank 2 and rank 1 -> 1/62 + 1/61 = 0.016129 + 0.016393 = 0.032522
    # c: rank 3 only       -> 1/63                              = 0.015873
    # d: rank 2 only       -> 1/62                              = 0.016129
    fused = rrf([LEXICAL, DENSE], k=60)
    assert [doc_id for doc_id, _ in fused] == ["b", "a", "d", "c"]
    assert dict(fused)["b"] == pytest.approx(1 / 62 + 1 / 61)
    assert dict(fused)["a"] == pytest.approx(1 / 61 + 1 / 63)


def test_rrf_ignores_score_scale() -> None:
    scaled = [(doc_id, score * 1000) for doc_id, score in LEXICAL]
    assert rrf([scaled, DENSE]) == rrf([LEXICAL, DENSE])


def test_weighted_hand_computed() -> None:
    # Min-max per ranking:
    # lexical: a = (12-3)/9 = 1, b = (9-3)/9 = 0.6667, c = 0
    # dense:   b = (0.9-0.5)/0.4 = 1, d = (0.8-0.5)/0.4 = 0.75, a = 0
    # With weights 0.5 and 0.5:
    # a = 0.5, b = 0.3333 + 0.5 = 0.8333, c = 0, d = 0.375
    fused = weighted([LEXICAL, DENSE], [0.5, 0.5])
    assert [doc_id for doc_id, _ in fused] == ["b", "a", "d", "c"]
    assert dict(fused) == pytest.approx({"b": 0.83333, "a": 0.5, "d": 0.375, "c": 0.0}, abs=1e-5)


def test_weighted_with_one_weight_at_zero_follows_the_other_ranking() -> None:
    fused = weighted([LEXICAL, DENSE], [1.0, 0.0])
    assert [doc_id for doc_id, _ in fused][:3] == ["a", "b", "c"]


def test_weighted_handles_equal_scores_and_empty_rankings() -> None:
    assert weighted([[("a", 5.0), ("b", 5.0)], []], [1.0, 1.0]) == [("a", 1.0), ("b", 1.0)]
    with pytest.raises(ValueError):
        weighted([LEXICAL], [0.5, 0.5])


def test_ties_are_broken_by_doc_id() -> None:
    assert rrf([[("z", 1.0)], [("y", 1.0)]]) == [("y", 1 / 61), ("z", 1 / 61)]


class Fixed:
    def __init__(self, ranking: list[tuple[str, float]]) -> None:
        self.ranking = ranking
        self.seen_k = 0

    def search(self, query: str, k: int = 10) -> list[tuple[str, float]]:
        self.seen_k = k
        return self.ranking[:k]


def test_hybrid_retriever_fetches_depth_and_returns_k() -> None:
    lexical, dense = Fixed(LEXICAL), Fixed(DENSE)
    hybrid = HybridRetriever([lexical, dense], rrf, depth=50)
    results = hybrid.search("q", k=2)
    assert [doc_id for doc_id, _ in results] == ["b", "a"]
    # Each retriever is asked for `depth` candidates, not for k.
    assert lexical.seen_k == dense.seen_k == 50
