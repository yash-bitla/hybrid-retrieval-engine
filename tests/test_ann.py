from collections.abc import Sequence

import numpy as np
import pytest

from hybrid_retrieval import DenseIndex
from hybrid_retrieval.dense import Embeddings

pytest.importorskip("faiss")

from hybrid_retrieval.ann import HnswIndex  # noqa: E402

RNG = np.random.default_rng(0)
DOCS: Embeddings = RNG.normal(size=(500, 16)).astype(np.float32)
QUERIES: Embeddings = RNG.normal(size=(20, 16)).astype(np.float32)
IDS = [f"d{i}" for i in range(len(DOCS))]


def encode(texts: Sequence[str]) -> Embeddings:
    # The "text" of a test query is its row number in QUERIES.
    return QUERIES[[int(t) for t in texts]]


def test_high_ef_search_matches_exact_search() -> None:
    exact = DenseIndex(IDS, DOCS, encode)
    approx = HnswIndex(IDS, DOCS, encode, m=16, ef_search=500)
    for q in range(len(QUERIES)):
        want = exact.search(str(q), k=10)
        got = approx.search(str(q), k=10)
        # With a candidate list as large as the index, the graph walk checks everything.
        assert [d for d, _ in got] == [d for d, _ in want]
        assert [s for _, s in got] == pytest.approx([s for _, s in want], abs=1e-5)


def test_recall_does_not_drop_when_ef_search_grows() -> None:
    exact = DenseIndex(IDS, DOCS, encode)
    approx = HnswIndex(IDS, DOCS, encode, m=4, ef_search=1)

    def recall() -> float:
        hits = 0
        for q in range(len(QUERIES)):
            want = {d for d, _ in exact.search(str(q), k=10)}
            hits += len(want & {d for d, _ in approx.search(str(q), k=10)})
        return hits / (10 * len(QUERIES))

    low = recall()
    approx.ef_search = 500
    assert approx.ef_search == 500
    assert recall() >= low
    assert recall() == pytest.approx(1.0)


def test_empty_index_and_mismatched_lengths() -> None:
    empty = HnswIndex([], np.zeros((0, 16), dtype=np.float32), encode)
    assert empty.search("0") == []
    with pytest.raises(ValueError):
        HnswIndex(["d1"], DOCS, encode)
