# Hybrid Retrieval Engine

A search engine built in stages (BM25, dense retrieval, fusion, reranking) with an evaluation harness that measures what each stage adds in ranking quality and what it costs in latency.

**Status:** milestone 1 of 5. No benchmark results yet.

## Milestones

| # | Milestone | State |
|---|---|---|
| 1 | BM25 inverted index and ranking metrics (nDCG@k, MRR, Recall@k) | Done |
| 2 | BEIR loader and evaluation runner with bootstrap confidence intervals | Next |
| 3 | Dense retrieval and fusion (RRF against weighted scores) | |
| 4 | Cross-encoder reranking: nDCG gain against P95 latency per rerank depth | |
| 5 | Top-k pruning (WAND or MaxScore) and ANN parameter trade-offs | |

## What exists

- `BM25Index`: an inverted index with term-at-a-time scoring. The tests check every score against a brute-force implementation of the formula.
- `ndcg_at_k`, `mrr`, `recall_at_k`: the tests check each against hand-computed values.

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
