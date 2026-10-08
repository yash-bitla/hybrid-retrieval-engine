| Retriever | Dataset | Queries | nDCG@10 | MRR@10 | Recall@100 | P50 ms | P95 ms |
|---|---|---:|---|---|---|---:|---:|
| bm25 | fiqa | 648 | 0.236 [0.214, 0.259] | 0.294 [0.264, 0.324] | 0.509 [0.478, 0.539] | 0.98 | 1.54 |
| dense | fiqa | 648 | 0.403 [0.375, 0.432] | 0.488 [0.454, 0.522] | 0.696 [0.668, 0.724] | 8.59 | 9.12 |
| hybrid-rrf | fiqa | 648 | 0.349 [0.323, 0.375] | 0.421 [0.389, 0.454] | 0.685 [0.656, 0.713] | 9.81 | 10.73 |
| hybrid-weighted | fiqa | 648 | 0.412 [0.384, 0.441] | 0.491 [0.458, 0.525] | 0.696 [0.668, 0.724] | 9.72 | 10.71 |

| Candidate | Baseline | nDCG@10 difference | Significant at 95% |
|---|---|---|---|
| dense | bm25 | +0.168 [+0.143, +0.192] | yes |
| hybrid-rrf | bm25 | +0.113 [+0.098, +0.129] | yes |
| hybrid-rrf | dense | -0.054 [-0.073, -0.036] | yes |
| hybrid-weighted | bm25 | +0.176 [+0.153, +0.199] | yes |
| hybrid-weighted | dense | +0.009 [+0.002, +0.015] | yes |
| hybrid-weighted | hybrid-rrf | +0.063 [+0.047, +0.079] | yes |
