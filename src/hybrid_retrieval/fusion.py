from collections import defaultdict
from collections.abc import Callable, Sequence

from hybrid_retrieval.evaluate import Retriever

Ranking = Sequence[tuple[str, float]]
Fuse = Callable[[Sequence[Ranking]], list[tuple[str, float]]]


def _sorted(scores: dict[str, float]) -> list[tuple[str, float]]:
    # Sort by score descending, then by doc id so ties are deterministic.
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))


def rrf(rankings: Sequence[Ranking], k: int = 60) -> list[tuple[str, float]]:
    """Reciprocal rank fusion: score(d) = sum over rankings of 1 / (k + rank of d).

    It uses ranks only, so the two retrievers' score scales do not need to match.
    """
    scores: dict[str, float] = defaultdict(float)
    for ranking in rankings:
        for rank, (doc_id, _) in enumerate(ranking, start=1):
            scores[doc_id] += 1.0 / (k + rank)
    return _sorted(scores)


def weighted(rankings: Sequence[Ranking], weights: Sequence[float]) -> list[tuple[str, float]]:
    """Weighted sum of min-max normalized scores.

    Each ranking's scores are rescaled to [0, 1] over its own candidates. A document
    that a ranking does not contain scores 0 there.
    """
    if len(rankings) != len(weights):
        raise ValueError("rankings and weights must have the same length")
    scores: dict[str, float] = defaultdict(float)
    for ranking, weight in zip(rankings, weights, strict=True):
        if not ranking:
            continue
        values = [score for _, score in ranking]
        low, span = min(values), max(values) - min(values)
        for doc_id, score in ranking:
            # When every score is equal, each candidate gets the full weight.
            scores[doc_id] += weight * ((score - low) / span if span else 1.0)
    return _sorted(scores)


class HybridRetriever:
    """Takes `depth` candidates from each retriever and fuses them into one ranking."""

    def __init__(self, retrievers: Sequence[Retriever], fuse: Fuse, depth: int = 100) -> None:
        self.retrievers = list(retrievers)
        self.fuse = fuse
        self.depth = depth

    def search(self, query: str, k: int = 10) -> list[tuple[str, float]]:
        rankings = [r.search(query, k=self.depth) for r in self.retrievers]
        return self.fuse(rankings)[:k]
