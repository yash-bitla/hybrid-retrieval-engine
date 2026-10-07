# Hybrid Retrieval Engine

A search engine built in stages (BM25, dense retrieval, fusion, reranking) with an evaluation harness that measures what each stage adds in ranking quality and what it costs in latency.

**Status:** milestone 2 of 5. BM25 is the only retriever so far.

## Results

[SciFact](https://github.com/beir-cellar/beir) test split: 5,183 documents, 300 queries. Brackets are 95% bootstrap confidence intervals over queries (10,000 resamples). Latency is per query, single thread, on an Apple Silicon laptop.

| Retriever | nDCG@10 | MRR@10 | Recall@100 | P50 ms | P95 ms |
|---|---|---|---|---:|---:|
| bm25 | 0.660 [0.615, 0.705] | 0.627 [0.578, 0.675] | 0.886 [0.848, 0.920] | 0.19 | 0.32 |

The BEIR paper reports 0.665 nDCG@10 for BM25 on SciFact, so the index agrees with the reference to within half a point. Its Recall@100 is 0.908 against 0.886 here. The likely cause is that this tokenizer has no stemming, but that is not yet tested.

With 300 queries the interval on nDCG@10 is about ±0.045. A later stage must beat BM25 by more than that, or be tested with a paired comparison, before the README calls it better.

Raw numbers are in [`benchmarks/results/`](benchmarks/results). Reproduce them with:

```bash
python -m benchmarks.run --dataset scifact
```

## Milestones

| # | Milestone | State |
|---|---|---|
| 1 | BM25 inverted index and ranking metrics (nDCG@k, MRR, Recall@k) | Done |
| 2 | BEIR loader and evaluation runner with bootstrap confidence intervals | Done |
| 3 | Dense retrieval and fusion (RRF against weighted scores) | Next |
| 4 | Cross-encoder reranking: nDCG gain against P95 latency per rerank depth | |
| 5 | Top-k pruning (WAND or MaxScore) and ANN parameter trade-offs | |

## How it works

- **`BM25Index`** is an inverted index with term-at-a-time scoring. Posting lists are parallel NumPy arrays. The tests check every score against a brute-force implementation of the formula.
- **`evaluate`** runs each query once at depth 100 and scores that one ranking with nDCG@10, MRR@10 and Recall@100. Any object with a `search(query, k)` method can be evaluated, so later retrievers use the same harness.
- **`bootstrap_ci`** resamples queries with replacement and reports the percentile interval of the mean.

```python
from hybrid_retrieval import BM25Index

index = BM25Index(["d1", "d2"], ["the cat sat on the mat", "the dog chased the cat"])
index.search("cat", k=10)  # [("d2", ...), ("d1", ...)]
```

## Development

```bash
python3.12 -m venv .venv && source .venv/bin/activate && pip install -e ".[dev]"
pytest && ruff check . && mypy
```
