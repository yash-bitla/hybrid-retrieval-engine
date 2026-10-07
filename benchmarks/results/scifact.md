| Retriever | Dataset | Queries | nDCG@10 | MRR@10 | Recall@100 | P50 ms | P95 ms |
|---|---|---:|---|---|---|---:|---:|
| bm25 | scifact | 300 | 0.660 [0.615, 0.705] | 0.627 [0.578, 0.675] | 0.886 [0.848, 0.920] | 0.16 | 0.30 |
| dense | scifact | 300 | 0.713 [0.669, 0.756] | 0.682 [0.634, 0.729] | 0.942 [0.913, 0.967] | 6.94 | 12.76 |
| hybrid-rrf | scifact | 300 | 0.703 [0.660, 0.747] | 0.672 [0.625, 0.718] | 0.965 [0.943, 0.983] | 8.43 | 14.93 |
| hybrid-weighted | scifact | 300 | 0.731 [0.688, 0.771] | 0.698 [0.652, 0.744] | 0.968 [0.948, 0.987] | 7.14 | 8.51 |

| Candidate | Baseline | nDCG@10 difference | Significant at 95% |
|---|---|---|---|
| dense | bm25 | +0.052 [+0.016, +0.089] | yes |
| hybrid-rrf | bm25 | +0.043 [+0.020, +0.066] | yes |
| hybrid-rrf | dense | -0.009 [-0.036, +0.017] | no |
| hybrid-weighted | bm25 | +0.070 [+0.039, +0.103] | yes |
| hybrid-weighted | dense | +0.018 [+0.004, +0.032] | yes |
| hybrid-weighted | hybrid-rrf | +0.027 [+0.007, +0.048] | yes |
