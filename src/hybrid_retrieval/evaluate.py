import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
import numpy.typing as npt

from hybrid_retrieval.beir import Dataset
from hybrid_retrieval.metrics import mrr, ndcg_at_k, recall_at_k


class Retriever(Protocol):
    def search(self, query: str, k: int = 10) -> list[tuple[str, float]]: ...


@dataclass(frozen=True)
class Estimate:
    """A mean over queries with a bootstrap confidence interval."""

    mean: float
    low: float
    high: float


@dataclass(frozen=True)
class Report:
    retriever: str
    dataset: str
    num_queries: int
    ndcg_at_10: Estimate
    mrr_at_10: Estimate
    recall_at_100: Estimate
    latency_p50_ms: float
    latency_p95_ms: float
    # Per-query nDCG@10 in dataset order, kept so two reports can be compared query by query.
    ndcg_per_query: tuple[float, ...] = field(default=(), repr=False)


def bootstrap_ci(
    values: Sequence[float] | npt.NDArray[np.float64],
    resamples: int = 10_000,
    confidence: float = 0.95,
    seed: int = 0,
) -> Estimate:
    """Percentile bootstrap over queries: resample with replacement, take the mean each time."""
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        raise ValueError("cannot bootstrap an empty sample")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(resamples, arr.size))
    means = arr[idx].mean(axis=1)
    tail = (1.0 - confidence) / 2.0
    low, high = np.quantile(means, [tail, 1.0 - tail])
    return Estimate(mean=float(arr.mean()), low=float(low), high=float(high))


def evaluate(retriever: Retriever, dataset: Dataset, name: str, depth: int = 100) -> Report:
    """Run every query once at `depth` and score the same ranking with all three metrics."""
    ndcgs: list[float] = []
    mrrs: list[float] = []
    recalls: list[float] = []
    latencies: list[float] = []
    for query_id, text in dataset.queries.items():
        start = time.perf_counter()
        results = retriever.search(text, k=depth)
        latencies.append((time.perf_counter() - start) * 1000.0)

        ranked = [doc_id for doc_id, _ in results]
        qrels = dataset.qrels[query_id]
        ndcgs.append(ndcg_at_k(ranked, qrels, k=10))
        mrrs.append(mrr(ranked, qrels, k=10))
        recalls.append(recall_at_k(ranked, qrels, k=100))

    p50, p95 = np.percentile(latencies, [50, 95])
    return Report(
        retriever=name,
        dataset=dataset.name,
        num_queries=len(ndcgs),
        ndcg_at_10=bootstrap_ci(ndcgs),
        mrr_at_10=bootstrap_ci(mrrs),
        recall_at_100=bootstrap_ci(recalls),
        latency_p50_ms=float(p50),
        latency_p95_ms=float(p95),
        ndcg_per_query=tuple(ndcgs),
    )


def paired_bootstrap(
    candidate: Sequence[float],
    baseline: Sequence[float],
    resamples: int = 10_000,
    confidence: float = 0.95,
    seed: int = 0,
) -> Estimate:
    """Mean per-query difference (candidate - baseline) with a bootstrap interval.

    Both systems are scored on the same queries, so the difference is resampled per query.
    This removes the query-difficulty noise that makes two separate intervals overlap.
    If the interval excludes 0, the difference is significant at the given confidence.
    """
    if len(candidate) != len(baseline):
        raise ValueError("both systems must be scored on the same queries")
    diffs = np.asarray(candidate, dtype=np.float64) - np.asarray(baseline, dtype=np.float64)
    return bootstrap_ci(diffs, resamples=resamples, confidence=confidence, seed=seed)


def _cell(e: Estimate) -> str:
    return f"{e.mean:.3f} [{e.low:.3f}, {e.high:.3f}]"


def to_markdown(reports: Sequence[Report]) -> str:
    lines = [
        "| Retriever | Dataset | Queries | nDCG@10 | MRR@10 | Recall@100 | P50 ms | P95 ms |",
        "|---|---|---:|---|---|---|---:|---:|",
    ]
    for r in reports:
        lines.append(
            f"| {r.retriever} | {r.dataset} | {r.num_queries} | {_cell(r.ndcg_at_10)} "
            f"| {_cell(r.mrr_at_10)} | {_cell(r.recall_at_100)} "
            f"| {r.latency_p50_ms:.2f} | {r.latency_p95_ms:.2f} |"
        )
    return "\n".join(lines)
