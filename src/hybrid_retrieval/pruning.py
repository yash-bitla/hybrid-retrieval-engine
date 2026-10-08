import numpy as np
import numpy.typing as npt
from numba import njit


@njit(cache=True)  # type: ignore[untyped-decorator]
def daat_topk(
    offsets: npt.NDArray[np.int64],
    postings: npt.NDArray[np.int32],
    impacts: npt.NDArray[np.float32],
    max_impact: npt.NDArray[np.float32],
    terms: npt.NDArray[np.int64],
    k: int,
    prune: bool,
) -> tuple[npt.NDArray[np.int32], npt.NDArray[np.float64], int]:
    """Document-at-a-time top-k over the posting lists of `terms`, with optional MaxScore.

    Returns (doc indexes, scores, postings scored), best first.

    MaxScore (Turtle and Flood, 1995). Sort the query terms by `max_impact`, smallest
    first, and keep the running sum `prefix`. Let theta be the k-th best score so far.
    While prefix[i] <= theta, a document that appears only in terms 0..i cannot beat
    theta, so those terms are "non-essential": the main loop stops walking their lists.
    It walks only the essential lists to find candidates, then looks each candidate up
    in the non-essential lists, and stops the lookup early when the candidate cannot
    reach theta even with every remaining upper bound.

    The result is the same top k as an exhaustive search. A skipped document scores at
    most theta, and every document already in the top k has a smaller index, so the
    skipped one also loses a tie.
    """
    order = np.argsort(max_impact[terms])
    terms = terms[order]
    m = len(terms)
    prefix = np.cumsum(max_impact[terms].astype(np.float64))
    cur = offsets[terms].copy()
    end = offsets[terms + 1]

    top_scores = np.full(k, -np.inf)
    top_docs = np.full(k, -1, dtype=np.int32)
    size = 0
    worst = 0  # position of the entry that the next better document replaces
    theta = -np.inf
    first = 0  # terms[first:] are essential
    scored = 0
    no_doc = np.iinfo(np.int32).max

    while True:
        doc = no_doc
        for i in range(first, m):
            if cur[i] < end[i] and postings[cur[i]] < doc:
                doc = postings[cur[i]]
        if doc == no_doc:
            break

        score = 0.0
        for i in range(first, m):
            if cur[i] < end[i] and postings[cur[i]] == doc:
                score += impacts[cur[i]]
                cur[i] += 1
                scored += 1

        alive = True
        i = first - 1
        while i >= 0:
            if score + prefix[i] <= theta:
                alive = False
                break
            # Seek list i forward to the first posting >= doc (binary search).
            lo, hi = cur[i], end[i]
            while lo < hi:
                mid = (lo + hi) // 2
                if postings[mid] < doc:
                    lo = mid + 1
                else:
                    hi = mid
            cur[i] = lo
            if lo < end[i] and postings[lo] == doc:
                score += impacts[lo]
                scored += 1
            i -= 1

        if not alive or score <= theta:
            continue
        if size < k:
            top_scores[size], top_docs[size] = score, doc
            size += 1
        else:
            top_scores[worst], top_docs[worst] = score, doc
        if size == k:
            # The worst entry has the lowest score. Among equal scores it is the one
            # with the largest doc index, to match the exhaustive tie order.
            worst = 0
            for j in range(1, k):
                if top_scores[j] < top_scores[worst] or (
                    top_scores[j] == top_scores[worst] and top_docs[j] > top_docs[worst]
                ):
                    worst = j
            theta = top_scores[worst]
            if prune:
                while first < m and prefix[first] <= theta:
                    first += 1

    # Best first: score descending, then doc index ascending.
    idx = np.argsort(top_docs[:size])
    idx = idx[np.argsort(-top_scores[:size][idx], kind="mergesort")]
    return top_docs[:size][idx], top_scores[:size][idx], scored
