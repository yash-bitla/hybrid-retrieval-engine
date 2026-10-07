from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from hybrid_retrieval.tokenizer import tokenize


@dataclass(frozen=True)
class Postings:
    """One term's posting list: parallel arrays sorted by doc index."""

    docs: npt.NDArray[np.int32]
    tfs: npt.NDArray[np.float32]


class BM25Index:
    """Inverted index with BM25 scoring (Lucene idf variant, never negative).

    score(q, d) = sum over query terms t of
        idf(t) * tf * (k1 + 1) / (tf + k1 * (1 - b + b * len(d) / avg_len))
    idf(t) = ln(1 + (N - df + 0.5) / (df + 0.5))
    """

    def __init__(
        self,
        doc_ids: Sequence[str],
        texts: Sequence[str],
        k1: float = 1.2,
        b: float = 0.75,
    ) -> None:
        if len(doc_ids) != len(texts):
            raise ValueError("doc_ids and texts must have the same length")
        self.doc_ids = list(doc_ids)
        self.k1 = k1
        self.b = b

        n = len(texts)
        doc_lens = np.zeros(n, dtype=np.float32)
        raw: dict[str, tuple[list[int], list[int]]] = {}
        for i, text in enumerate(texts):
            counts = Counter(tokenize(text))
            doc_lens[i] = sum(counts.values())
            for term, tf in counts.items():
                docs, tfs = raw.setdefault(term, ([], []))
                docs.append(i)
                tfs.append(tf)

        avg_len = float(doc_lens.mean()) if n else 0.0
        # The length normalizer depends only on the document, so compute it once.
        self._norm = k1 * (1.0 - b + b * doc_lens / avg_len) if avg_len else doc_lens
        self._postings = {
            term: Postings(np.asarray(docs, dtype=np.int32), np.asarray(tfs, dtype=np.float32))
            for term, (docs, tfs) in raw.items()
        }
        self._idf = {
            term: float(np.log(1.0 + (n - len(p.docs) + 0.5) / (len(p.docs) + 0.5)))
            for term, p in self._postings.items()
        }

    def __len__(self) -> int:
        return len(self.doc_ids)

    def scores(self, query: str) -> npt.NDArray[np.float32]:
        """BM25 score of every document for the query (term-at-a-time)."""
        acc = np.zeros(len(self.doc_ids), dtype=np.float32)
        for term in set(tokenize(query)):
            postings = self._postings.get(term)
            if postings is None:
                continue
            tf = postings.tfs
            acc[postings.docs] += (
                self._idf[term] * tf * (self.k1 + 1.0) / (tf + self._norm[postings.docs])
            )
        return acc

    def search(self, query: str, k: int = 10) -> list[tuple[str, float]]:
        """Top-k (doc_id, score) pairs, best first. Documents that match no term are left out."""
        acc = self.scores(query)
        k = min(k, int(np.count_nonzero(acc)))
        if k == 0:
            return []
        top = np.argpartition(-acc, k - 1)[:k]
        # Sort by score descending, then by doc index so ties are deterministic.
        top = top[np.lexsort((top, -acc[top]))]
        return [(self.doc_ids[i], float(acc[i])) for i in top]
