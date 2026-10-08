from collections.abc import Sequence

import pytest

from hybrid_retrieval.rerank import Reranker

TEXTS = {"a": "one", "b": "three", "c": "four", "d": "seventeen", "e": "x"}
FIRST_STAGE = [("a", 5.0), ("b", 4.0), ("c", 3.0), ("d", 2.0), ("e", 1.0)]


class Fixed:
    def __init__(self) -> None:
        self.seen_k = 0

    def search(self, query: str, k: int = 10) -> list[tuple[str, float]]:
        self.seen_k = k
        return FIRST_STAGE[:k]


class LengthScorer:
    """Scores a passage by its length, and records what it was asked to score."""

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def __call__(self, query: str, passages: Sequence[str]) -> Sequence[float]:
        self.calls.append(list(passages))
        return [float(len(p)) for p in passages]


def test_reranks_the_head_and_keeps_the_tail_in_first_stage_order() -> None:
    scorer = LengthScorer()
    reranker = Reranker(Fixed(), TEXTS, scorer, depth=3)
    results = reranker.search("q", k=5)
    # Head a, b, c has lengths 3, 5, 4, so it becomes b, c, a. The tail d, e is not scored,
    # so "d" (length 9, the longest passage) stays at rank 4.
    assert [doc_id for doc_id, _ in results] == ["b", "c", "a", "d", "e"]
    assert scorer.calls == [["one", "three", "four"]]


def test_scores_only_depth_candidates() -> None:
    scorer = LengthScorer()
    Reranker(Fixed(), TEXTS, scorer, depth=2).search("q", k=5)
    assert len(scorer.calls[0]) == 2


def test_fetches_depth_candidates_when_k_is_smaller() -> None:
    first_stage = Fixed()
    results = Reranker(first_stage, TEXTS, LengthScorer(), depth=4).search("q", k=1)
    # The best of the top 4 is "d" (length 9). The first stage is asked for 4, not for 1.
    assert results == [("d", 9.0)]
    assert first_stage.seen_k == 4


def test_equal_scores_keep_first_stage_order() -> None:
    reranker = Reranker(Fixed(), TEXTS, lambda q, ps: [1.0] * len(ps), depth=5)
    assert [doc_id for doc_id, _ in reranker.search("q", k=5)] == ["a", "b", "c", "d", "e"]


def test_depth_larger_than_candidates_and_no_candidates() -> None:
    assert len(Reranker(Fixed(), TEXTS, LengthScorer(), depth=50).search("q", k=10)) == 5

    class Empty:
        def search(self, query: str, k: int = 10) -> list[tuple[str, float]]:
            return []

    assert Reranker(Empty(), TEXTS, LengthScorer(), depth=3).search("q") == []


def test_depth_must_be_positive() -> None:
    with pytest.raises(ValueError):
        Reranker(Fixed(), TEXTS, LengthScorer(), depth=0)
