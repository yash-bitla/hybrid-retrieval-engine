import math

import pytest

from hybrid_retrieval import mrr, ndcg_at_k, recall_at_k

QRELS = {"a": 2, "b": 1, "c": 0}


def test_ndcg_perfect_ranking_is_one() -> None:
    assert ndcg_at_k(["a", "b", "x"], QRELS, k=10) == pytest.approx(1.0)


def test_ndcg_hand_computed() -> None:
    # Ranking: x (rel 0), b (rel 1), a (rel 2).
    # DCG  = 0/log2(2) + 1/log2(3) + 2/log2(4) = 0 + 0.63093 + 1.0 = 1.63093
    # IDCG = 2/log2(2) + 1/log2(3)             = 2 + 0.63093       = 2.63093
    # nDCG = 1.63093 / 2.63093 = 0.61991
    dcg = 1 / math.log2(3) + 2 / math.log2(4)
    idcg = 2 / math.log2(2) + 1 / math.log2(3)
    assert ndcg_at_k(["x", "b", "a"], QRELS, k=10) == pytest.approx(dcg / idcg)
    assert dcg / idcg == pytest.approx(0.61991, abs=1e-5)


def test_ndcg_cutoff() -> None:
    # Only "x" is inside k=1, so DCG = 0.
    assert ndcg_at_k(["x", "a"], QRELS, k=1) == 0.0


def test_mrr() -> None:
    # First relevant document is at rank 3, so MRR = 1/3. "c" has rel 0 and does not count.
    assert mrr(["c", "x", "b", "a"], QRELS) == pytest.approx(1 / 3)
    assert mrr(["c", "x"], QRELS) == 0.0


def test_recall() -> None:
    # Relevant set is {a, b}. Top 2 holds only "a", so recall = 1/2.
    assert recall_at_k(["a", "x", "b"], QRELS, k=2) == pytest.approx(0.5)
    assert recall_at_k(["a", "x", "b"], QRELS, k=3) == pytest.approx(1.0)


def test_no_relevant_documents() -> None:
    assert ndcg_at_k(["a"], {"a": 0}) == 0.0
    assert recall_at_k(["a"], {}) == 0.0
