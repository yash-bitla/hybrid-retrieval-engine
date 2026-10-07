import pytest

from hybrid_retrieval.beir import Dataset
from hybrid_retrieval.evaluate import bootstrap_ci, evaluate, to_markdown


class FixedRetriever:
    """Returns a preset ranking for each query text."""

    def __init__(self, rankings: dict[str, list[str]]) -> None:
        self.rankings = rankings

    def search(self, query: str, k: int = 10) -> list[tuple[str, float]]:
        return [(doc_id, 1.0) for doc_id in self.rankings[query][:k]]


DATASET = Dataset(
    name="tiny",
    corpus={"a": "", "b": "", "x": ""},
    queries={"q1": "first", "q2": "second"},
    qrels={"q1": {"a": 1}, "q2": {"b": 1}},
)


def test_evaluate_averages_over_queries() -> None:
    # q1: relevant doc at rank 1 -> nDCG 1,              MRR 1,   recall 1
    # q2: relevant doc at rank 2 -> nDCG 1/log2(3)=0.631, MRR 0.5, recall 1
    # Means: nDCG (1 + 0.63093) / 2 = 0.81546, MRR (1 + 0.5) / 2 = 0.75, recall 1
    retriever = FixedRetriever({"first": ["a", "x"], "second": ["x", "b"]})
    report = evaluate(retriever, DATASET, name="fixed")
    assert report.num_queries == 2
    assert report.ndcg_at_10.mean == pytest.approx(0.81546, abs=1e-5)
    assert report.mrr_at_10.mean == pytest.approx(0.75)
    assert report.recall_at_100.mean == pytest.approx(1.0)
    assert report.latency_p95_ms >= report.latency_p50_ms >= 0.0


def test_bootstrap_interval_contains_the_mean() -> None:
    est = bootstrap_ci([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    assert est.mean == pytest.approx(0.5)
    assert est.low < est.mean < est.high


def test_bootstrap_of_constant_sample_has_zero_width() -> None:
    est = bootstrap_ci([0.7] * 20)
    assert est.low == pytest.approx(0.7)
    assert est.high == pytest.approx(0.7)


def test_bootstrap_interval_narrows_with_more_queries() -> None:
    small = bootstrap_ci([0.0, 1.0] * 10)
    large = bootstrap_ci([0.0, 1.0] * 1000, resamples=2000)
    assert (large.high - large.low) < (small.high - small.low)


def test_bootstrap_is_deterministic_and_rejects_empty() -> None:
    values = [0.1, 0.9, 0.4, 0.6]
    assert bootstrap_ci(values) == bootstrap_ci(values)
    with pytest.raises(ValueError):
        bootstrap_ci([])


def test_markdown_has_one_row_per_report() -> None:
    retriever = FixedRetriever({"first": ["a"], "second": ["b"]})
    table = to_markdown([evaluate(retriever, DATASET, name="fixed")])
    assert len(table.splitlines()) == 3
    assert "| fixed | tiny | 2 | 1.000 [1.000, 1.000]" in table
