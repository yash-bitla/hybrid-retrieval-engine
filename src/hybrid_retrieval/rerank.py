from collections.abc import Callable, Mapping, Sequence

from hybrid_retrieval.evaluate import Retriever

# (query, passages) -> one relevance score per passage, higher is better
Scorer = Callable[[str, Sequence[str]], Sequence[float]]


class Reranker:
    """Second stage: rescore the top `depth` first-stage candidates with a slower model.

    Candidates below `depth` keep their first-stage order and stay under the reranked
    block, so recall at a large k is the same as the first stage's. The reranked block
    carries the scorer's scores and the tail carries first-stage scores. The two scales
    are not comparable, so use the order and not the score values.
    """

    def __init__(
        self, retriever: Retriever, texts: Mapping[str, str], score: Scorer, depth: int
    ) -> None:
        if depth < 1:
            raise ValueError("depth must be at least 1")
        self.retriever = retriever
        self.texts = texts
        self.score = score
        self.depth = depth

    def search(self, query: str, k: int = 10) -> list[tuple[str, float]]:
        candidates = self.retriever.search(query, k=max(k, self.depth))
        head, tail = candidates[: self.depth], candidates[self.depth :]
        if not head:
            return []
        scores = self.score(query, [self.texts[doc_id] for doc_id, _ in head])
        # Python's sort is stable, so equal scores keep their first-stage order.
        reranked = sorted(
            zip((doc_id for doc_id, _ in head), map(float, scores), strict=True),
            key=lambda item: -item[1],
        )
        return (reranked + list(tail))[:k]


def cross_encoder(model_name: str, batch_size: int = 32) -> Scorer:
    """A scorer backed by a sentence-transformers cross-encoder.

    Needs the optional `dense` dependencies. A cross-encoder reads the query and the
    passage together, so it cannot precompute anything per document: the cost is one
    model forward pass per candidate, on every query.
    """
    from sentence_transformers import CrossEncoder

    model = CrossEncoder(model_name)

    def score(query: str, passages: Sequence[str]) -> Sequence[float]:
        scores: Sequence[float] = model.predict(
            [(query, passage) for passage in passages], batch_size=batch_size
        ).tolist()
        return scores

    return score
