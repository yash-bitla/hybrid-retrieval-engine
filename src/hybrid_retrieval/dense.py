from collections.abc import Callable, Sequence

import numpy as np
import numpy.typing as npt

Embeddings = npt.NDArray[np.float32]
Encoder = Callable[[Sequence[str]], Embeddings]


def _normalize(x: Embeddings) -> Embeddings:
    norms = np.linalg.norm(x, axis=-1, keepdims=True)
    normalized: Embeddings = (x / np.maximum(norms, 1e-12)).astype(np.float32)
    return normalized


class DenseIndex:
    """Exact cosine-similarity search over document embeddings.

    Every query is compared with every document (one matrix-vector product).
    That is exact, so it is the reference that an approximate index is measured against.
    """

    def __init__(
        self, doc_ids: Sequence[str], embeddings: Embeddings, encode_query: Encoder
    ) -> None:
        if len(doc_ids) != len(embeddings):
            raise ValueError("doc_ids and embeddings must have the same length")
        self.doc_ids = list(doc_ids)
        self._embeddings = _normalize(np.asarray(embeddings, dtype=np.float32))
        self._encode_query = encode_query

    def __len__(self) -> int:
        return len(self.doc_ids)

    def search(self, query: str, k: int = 10) -> list[tuple[str, float]]:
        """Top-k (doc_id, cosine similarity) pairs, best first."""
        k = min(k, len(self.doc_ids))
        if k == 0:
            return []
        q = _normalize(self._encode_query([query]))[0]
        sims = self._embeddings @ q
        top = np.argpartition(-sims, k - 1)[:k]
        # Sort by similarity descending, then by doc index so ties are deterministic.
        top = top[np.lexsort((top, -sims[top]))]
        return [(self.doc_ids[i], float(sims[i])) for i in top]


def sentence_transformer(
    model_name: str, query_prefix: str = "", batch_size: int = 64
) -> tuple[Encoder, Encoder]:
    """Document and query encoders backed by a sentence-transformers model.

    Needs the optional `dense` dependencies. Some models (the BGE family) expect an
    instruction in front of each query but not in front of documents.
    """
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)

    def encode(texts: Sequence[str], prefix: str) -> Embeddings:
        out: Embeddings = model.encode(
            [prefix + t for t in texts], batch_size=batch_size, convert_to_numpy=True
        ).astype(np.float32)
        return out

    return (lambda texts: encode(texts, "")), (lambda texts: encode(texts, query_prefix))
