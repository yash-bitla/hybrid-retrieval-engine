from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

from hybrid_retrieval.dense import Embeddings, Encoder, _normalize


class HnswIndex:
    """Approximate cosine-similarity search with an HNSW graph (FAISS).

    HNSW links each vector to about `m` near neighbors, and a query walks that graph
    toward closer vectors. `ef_search` is the size of the candidate list the walk keeps:
    a larger value checks more vectors, so recall and latency both go up. Unlike
    `DenseIndex`, the result can miss a true nearest neighbor.

    Needs the optional `ann` dependencies.
    """

    def __init__(
        self,
        doc_ids: Sequence[str],
        embeddings: Embeddings,
        encode_query: Encoder,
        m: int = 32,
        ef_construction: int = 200,
        ef_search: int = 64,
    ) -> None:
        import faiss

        if len(doc_ids) != len(embeddings):
            raise ValueError("doc_ids and embeddings must have the same length")
        self.doc_ids = list(doc_ids)
        self._encode_query = encode_query
        vectors = _normalize(np.asarray(embeddings, dtype=np.float32))
        # On unit vectors the inner product is the cosine similarity.
        self._index = faiss.IndexHNSWFlat(vectors.shape[1], m, faiss.METRIC_INNER_PRODUCT)
        self._index.hnsw.efConstruction = ef_construction
        self._index.add(vectors)
        self.ef_search = ef_search

    @property
    def ef_search(self) -> int:
        return int(self._index.hnsw.efSearch)

    @ef_search.setter
    def ef_search(self, value: int) -> None:
        self._index.hnsw.efSearch = value

    def __len__(self) -> int:
        return len(self.doc_ids)

    def search_vectors(
        self, queries: Embeddings, k: int = 10
    ) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.float32]]:
        """Document indexes and similarities for already-encoded queries, one row per query."""
        sims, idx = self._index.search(_normalize(queries), min(k, len(self.doc_ids)))
        return idx, sims

    def search(self, query: str, k: int = 10) -> list[tuple[str, float]]:
        """Top-k (doc_id, cosine similarity) pairs, best first."""
        if not self.doc_ids:
            return []
        idx, sims = self.search_vectors(self._encode_query([query]), k)
        # FAISS pads with -1 when the graph walk finds fewer than k vectors.
        return [(self.doc_ids[i], float(s)) for i, s in zip(idx[0], sims[0], strict=True) if i >= 0]
