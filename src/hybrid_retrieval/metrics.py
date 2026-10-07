import math
from collections.abc import Mapping, Sequence


def ndcg_at_k(ranked: Sequence[str], qrels: Mapping[str, int], k: int = 10) -> float:
    """nDCG@k with linear gain: DCG = sum of rel_i / log2(i + 1), i counted from 1."""
    dcg = sum(qrels.get(doc, 0) / math.log2(i + 2) for i, doc in enumerate(ranked[:k]))
    ideal = sorted((rel for rel in qrels.values() if rel > 0), reverse=True)[:k]
    idcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(ideal))
    return dcg / idcg if idcg else 0.0


def mrr(ranked: Sequence[str], qrels: Mapping[str, int], k: int = 10) -> float:
    """Reciprocal rank of the first relevant document in the top k, or 0."""
    for i, doc in enumerate(ranked[:k]):
        if qrels.get(doc, 0) > 0:
            return 1.0 / (i + 1)
    return 0.0


def recall_at_k(ranked: Sequence[str], qrels: Mapping[str, int], k: int = 100) -> float:
    """Share of the relevant documents that appear in the top k."""
    relevant = {doc for doc, rel in qrels.items() if rel > 0}
    if not relevant:
        return 0.0
    return len(relevant.intersection(ranked[:k])) / len(relevant)
