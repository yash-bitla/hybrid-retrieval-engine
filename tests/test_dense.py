from collections.abc import Sequence

import numpy as np
import pytest

from hybrid_retrieval import DenseIndex
from hybrid_retrieval.dense import Embeddings

# A 2-dimensional toy space: the first axis is "cats", the second is "dogs".
VECTORS = {
    "cats": [1.0, 0.0],
    "dogs": [0.0, 1.0],
    "pets": [1.0, 1.0],
}


def encode(texts: Sequence[str]) -> Embeddings:
    return np.asarray([VECTORS[t] for t in texts], dtype=np.float32)


def make_index() -> DenseIndex:
    # d3 points the same way as d1 but is 10 times longer.
    embeddings = np.asarray([[1.0, 0.0], [0.0, 1.0], [10.0, 0.0], [3.0, 4.0]], dtype=np.float32)
    return DenseIndex(["d1", "d2", "d3", "d4"], embeddings, encode)


def test_search_ranks_by_cosine_similarity() -> None:
    results = make_index().search("dogs", k=4)
    # cos(dogs, d2) = 1. cos(dogs, d4) = 4 / 5 = 0.8. d1 and d3 are orthogonal: 0.
    assert [doc_id for doc_id, _ in results] == ["d2", "d4", "d1", "d3"]
    assert [score for _, score in results] == pytest.approx([1.0, 0.8, 0.0, 0.0])


def test_vector_length_does_not_change_the_score() -> None:
    results = dict(make_index().search("cats", k=4))
    # d3 = 10 * d1, so both have cosine 1 with "cats". The tie is broken by doc index.
    assert results["d1"] == pytest.approx(results["d3"]) == pytest.approx(1.0)
    assert [doc_id for doc_id, _ in make_index().search("cats", k=2)] == ["d1", "d3"]


def test_query_is_normalized() -> None:
    # "pets" = (1, 1) has length sqrt(2). cos(pets, d4) = (3 + 4) / (5 * sqrt(2)) = 0.98995
    top = make_index().search("pets", k=1)[0]
    assert top[0] == "d4"
    assert top[1] == pytest.approx(0.98995, abs=1e-5)


def test_k_larger_than_index_and_empty_index() -> None:
    assert len(make_index().search("cats", k=99)) == 4
    empty = DenseIndex([], np.zeros((0, 2), dtype=np.float32), encode)
    assert empty.search("cats") == []


def test_mismatched_lengths_raise() -> None:
    with pytest.raises(ValueError):
        DenseIndex(["d1"], np.zeros((2, 2), dtype=np.float32), encode)
