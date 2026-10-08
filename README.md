# Hybrid Retrieval Engine

A search engine built in stages (BM25, dense retrieval, fusion, reranking) with an evaluation harness that measures what each stage adds in ranking quality and what it costs in latency.

**Status:** milestone 4 of 5.

## Results

[SciFact](https://github.com/beir-cellar/beir) test split: 5,183 documents, 300 queries. Brackets are 95% bootstrap confidence intervals over queries (10,000 resamples). Latency is per query, single thread, on an Apple Silicon laptop, and it varies by a few milliseconds between runs.

| Retriever | nDCG@10 | MRR@10 | Recall@100 | P50 ms | P95 ms |
|---|---|---|---|---:|---:|
| bm25 | 0.660 [0.615, 0.705] | 0.627 [0.578, 0.675] | 0.886 [0.848, 0.920] | 0.16 | 0.30 |
| dense (`bge-small-en-v1.5`) | 0.713 [0.669, 0.756] | 0.682 [0.634, 0.729] | 0.942 [0.913, 0.967] | 6.94 | 12.76 |
| hybrid, RRF | 0.703 [0.660, 0.747] | 0.672 [0.625, 0.718] | 0.965 [0.943, 0.983] | 8.43 | 14.93 |
| hybrid, weighted (alpha = 0.7) | 0.731 [0.688, 0.771] | 0.698 [0.652, 0.744] | 0.968 [0.948, 0.987] | 7.14 | 8.51 |

The intervals in that table overlap, so it cannot say which retriever is better. Both systems answer the same 300 queries, so the right test is on the per-query difference (paired bootstrap):

| Candidate | Baseline | nDCG@10 difference | Significant at 95% |
|---|---|---|---|
| dense | bm25 | +0.052 [+0.016, +0.089] | yes |
| hybrid, RRF | bm25 | +0.043 [+0.020, +0.066] | yes |
| hybrid, RRF | dense | -0.009 [-0.036, +0.017] | no |
| hybrid, weighted | bm25 | +0.070 [+0.039, +0.103] | yes |
| hybrid, weighted | dense | +0.018 [+0.004, +0.032] | yes |
| hybrid, weighted | hybrid, RRF | +0.027 [+0.007, +0.048] | yes |

**What the numbers say.**

- **Hybrid is not automatically better.** RRF adds BM25 to a stronger dense retriever and gets no gain in nDCG@10 over dense alone (-0.009, not significant). Equal-weight rank fusion lets the weaker retriever pull good dense results down.
- **A tuned weight is better than RRF.** Weighted fusion with 70% dense and 30% BM25 is better than dense by 0.018 and better than RRF by 0.027. Both differences are significant, and both are small.
- **Fusion helps recall more than ranking.** Recall@100 goes from 0.942 (dense) to 0.965 and 0.968. The two retrievers find different documents. That matters for milestone 4, because a reranker can only reorder what the first stage found.
- **The cost is the query encoder.** BM25 answers in 0.16 ms. Each dense or hybrid query takes about 7 ms, and almost all of that is the embedding model. The fusion step itself is negligible.

**The fusion weight is tuned without the test set.** Alpha was chosen from 11 values on the 809 training queries, which gave 0.7. The test queries were used once, for the tables above.

**Checks against published numbers.** The BEIR paper reports 0.665 nDCG@10 for BM25 on SciFact, and the BGE model card reports 0.713 for `bge-small-en-v1.5`. Both agree with this harness.

### Reranking

A cross-encoder rescores the top candidates of the best first stage (the weighted hybrid). Two models, four rerank depths, the same 300 queries. The gain column is a paired bootstrap against the first stage.

![nDCG@10 against P95 latency for two rerankers at four depths. Both lines fall as depth grows, and only one point is above the first-stage line.](benchmarks/results/scifact-rerank.png)

| Reranker | Depth | nDCG@10 | Gain over first stage | Significant | P50 ms | P95 ms |
|---|---:|---|---|---|---:|---:|
| none (first stage) | 0 | 0.731 [0.688, 0.771] | | | 11.1 | 12.5 |
| `ms-marco-MiniLM-L-6-v2` (23M) | 10 | 0.711 [0.668, 0.752] | -0.020 [-0.042, +0.002] | no | 62.2 | 77.1 |
| | 20 | 0.704 [0.660, 0.745] | -0.027 [-0.053, -0.001] | yes | 130.4 | 138.6 |
| | 50 | 0.699 [0.656, 0.742] | -0.032 [-0.060, -0.004] | yes | 260.6 | 293.4 |
| | 100 | 0.690 [0.646, 0.733] | -0.041 [-0.070, -0.013] | yes | 442.8 | 503.0 |
| `bge-reranker-base` (278M) | 10 | 0.734 [0.692, 0.774] | +0.003 [-0.018, +0.023] | no | 312.1 | 313.7 |
| | 20 | 0.729 [0.687, 0.768] | -0.002 [-0.026, +0.021] | no | 610.4 | 630.7 |
| | 50 | 0.720 [0.678, 0.760] | -0.010 [-0.036, +0.015] | no | 1368.0 | 1506.6 |
| | 100 | 0.710 [0.667, 0.751] | -0.021 [-0.049, +0.006] | no | 2634.9 | 2900.9 |

**What the numbers say.**

- **Neither reranker is worth its cost on this dataset.** The small model makes the ranking worse, and three of its four losses are significant. The large model never differs significantly from the first stage, and its best point (+0.003 at depth 10) costs 312 ms against 11 ms.
- **A deeper rerank is worse, for both models.** The first stage has 96.8% of the relevant documents in its top 100, so better documents are there to be promoted. The rerankers promote the wrong ones: the more candidates they see, the more good first-stage results they displace.
- **Latency grows linearly with depth.** A cross-encoder runs one forward pass per candidate and can precompute nothing, so depth 100 costs about 10 times depth 10.
- **A likely cause is domain mismatch, and it is not tested here.** Both rerankers are general-purpose models, and SciFact queries are scientific claims. The first-stage embedding model and the tuned fusion weight may already capture what these rerankers know.

The practical conclusion is that a reranker is a hypothesis to test on your own data, not a default. On this dataset the correct engineering decision is to ship the first stage alone.

Reranker latency was measured on the laptop GPU (Apple MPS). The first stage measured 11 ms here and 7 ms in the table above, which is run-to-run variation.

Raw numbers, including the full alpha sweep, are in [`benchmarks/results/`](benchmarks/results). Reproduce them with:

```bash
pip install -e ".[dev,dense]"
python -m benchmarks.run --dataset scifact
python -m benchmarks.rerank --dataset scifact   # about 35 minutes on a laptop GPU
pip install -e ".[bench]" && python -m benchmarks.plot --dataset scifact
```

## Milestones

| # | Milestone | State |
|---|---|---|
| 1 | BM25 inverted index and ranking metrics (nDCG@k, MRR, Recall@k) | Done |
| 2 | BEIR loader and evaluation runner with bootstrap confidence intervals | Done |
| 3 | Dense retrieval and fusion (RRF against weighted scores) | Done |
| 4 | Cross-encoder reranking: nDCG gain against P95 latency per rerank depth | Done |
| 5 | Top-k pruning (WAND or MaxScore) and ANN parameter trade-offs | Next |

## How it works

- **`BM25Index`** is an inverted index with term-at-a-time scoring. Posting lists are parallel NumPy arrays. The tests check every score against a brute-force implementation of the formula.
- **`evaluate`** runs each query once at depth 100 and scores that one ranking with nDCG@10, MRR@10 and Recall@100. Any object with a `search(query, k)` method can be evaluated, so later retrievers use the same harness.
- **`DenseIndex`** is exact cosine search: one matrix-vector product over all document embeddings. It is the reference that an approximate index will be measured against in milestone 5.
- **`rrf`** and **`weighted`** fuse rankings. RRF uses ranks only. `weighted` rescales each retriever's scores to [0, 1] and takes a weighted sum.
- **`Reranker`** rescores the top `depth` candidates of any retriever with an injected scoring function. Candidates below `depth` keep their first-stage order, so recall at a large k does not change.
- **`bootstrap_ci`** resamples queries with replacement and reports the percentile interval of the mean. **`paired_bootstrap`** does the same for the per-query difference between two retrievers.

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
