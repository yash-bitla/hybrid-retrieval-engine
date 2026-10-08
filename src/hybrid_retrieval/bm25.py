from collections import Counter
from collections.abc import Sequence
from typing import Literal

import numpy as np
import numpy.typing as npt

from hybrid_retrieval.pruning import daat_topk
from hybrid_retrieval.tokenizer import tokenize

Method = Literal["exhaustive", "daat", "maxscore"]


class BM25Index:
    """Inverted index with BM25 scoring (Lucene idf variant, never negative).

    score(q, d) = sum over query terms t of
        idf(t) * tf * (k1 + 1) / (tf + k1 * (1 - b + b * len(d) / avg_len))
    idf(t) = ln(1 + (N - df + 0.5) / (df + 0.5))

    Everything in that formula is known at build time, so the index stores one finished
    score contribution (an "impact") per posting. A query only adds impacts together.

    Postings are stored as three flat arrays. Term t owns the slice
    offsets[t]:offsets[t + 1] of `doc_ids` (sorted ascending) and of `impacts`.
    `max_impact[t]` is the largest impact in that slice: the most that term t can add
    to any document's score. MaxScore pruning uses it to skip documents.
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

        self._term_ids = {term: t for t, term in enumerate(raw)}
        dfs = np.asarray([len(docs) for docs, _ in raw.values()], dtype=np.int64)
        self._offsets = np.concatenate(([0], np.cumsum(dfs))).astype(np.int64)
        self._postings = np.fromiter(
            (d for docs, _ in raw.values() for d in docs), dtype=np.int32, count=int(dfs.sum())
        )
        term_freqs = np.fromiter(
            (f for _, freqs in raw.values() for f in freqs), dtype=np.float32, count=int(dfs.sum())
        )

        avg_len = float(doc_lens.mean()) if n else 0.0
        norm = k1 * (1.0 - b + b * doc_lens / avg_len) if avg_len else doc_lens
        idf = np.log(1.0 + (n - dfs + 0.5) / (dfs + 0.5))
        self._impacts: npt.NDArray[np.float32] = (
            np.repeat(idf, dfs) * term_freqs * (k1 + 1.0) / (term_freqs + norm[self._postings])
        ).astype(np.float32)
        self._max_impact: npt.NDArray[np.float32] = (
            np.maximum.reduceat(self._impacts, self._offsets[:-1])
            if len(dfs)
            else np.zeros(0, dtype=np.float32)
        )
        # Postings read by the last "daat" or "maxscore" search, for measuring pruning.
        self.last_postings_scored = 0

    def __len__(self) -> int:
        return len(self.doc_ids)

    def _terms(self, query: str) -> npt.NDArray[np.int64]:
        """Ids of the distinct query terms that the index contains."""
        ids = {self._term_ids[t] for t in tokenize(query) if t in self._term_ids}
        return np.fromiter(ids, dtype=np.int64, count=len(ids))

    def num_postings(self, query: str) -> int:
        """Total posting-list length of the query's terms: the work an exhaustive search does."""
        terms = self._terms(query)
        return int((self._offsets[terms + 1] - self._offsets[terms]).sum())

    def scores(self, query: str) -> npt.NDArray[np.float32]:
        """BM25 score of every document for the query (term-at-a-time)."""
        acc = np.zeros(len(self.doc_ids), dtype=np.float32)
        for t in self._terms(query):
            start, end = self._offsets[t], self._offsets[t + 1]
            acc[self._postings[start:end]] += self._impacts[start:end]
        return acc

    def search(
        self, query: str, k: int = 10, method: Method = "exhaustive"
    ) -> list[tuple[str, float]]:
        """Top-k (doc_id, score) pairs, best first. Documents that match no term are left out.

        "exhaustive" scores every posting with NumPy, one term at a time.
        "daat" scores every posting in compiled code, one document at a time.
        "maxscore" is "daat" plus MaxScore pruning: it returns the same top k but skips
        documents that cannot reach it.
        """
        if method == "exhaustive":
            acc = self.scores(query)
            k = min(k, int(np.count_nonzero(acc)))
            if k == 0:
                return []
            top = np.argpartition(-acc, k - 1)[:k]
            # Sort by score descending, then by doc index so ties are deterministic.
            top = top[np.lexsort((top, -acc[top]))]
            return [(self.doc_ids[i], float(acc[i])) for i in top]

        terms = self._terms(query)
        if k < 1 or len(terms) == 0:
            self.last_postings_scored = 0
            return []
        docs, scores, self.last_postings_scored = daat_topk(
            self._offsets, self._postings, self._impacts, self._max_impact,
            terms, k, method == "maxscore",
        )
        return [(self.doc_ids[d], float(s)) for d, s in zip(docs, scores, strict=True)]
