| Dataset | Documents | k | Method | Mean ms | P50 ms | P95 ms | Postings read |
|---|---:|---:|---|---:|---:|---:|---:|
| scifact | 5,183 | 10 | NumPy, term-at-a-time | 0.10 | 0.10 | 0.15 | 100% |
| scifact | 5,183 | 10 | compiled, document-at-a-time | 0.15 | 0.13 | 0.27 | 100% |
| scifact | 5,183 | 10 | compiled + MaxScore | 0.05 | 0.04 | 0.13 | 9% |
| scifact | 5,183 | 100 | NumPy, term-at-a-time | 0.11 | 0.11 | 0.16 | 100% |
| scifact | 5,183 | 100 | compiled, document-at-a-time | 0.19 | 0.17 | 0.31 | 100% |
| scifact | 5,183 | 100 | compiled + MaxScore | 0.14 | 0.13 | 0.25 | 23% |
| fiqa | 57,638 | 10 | NumPy, term-at-a-time | 0.75 | 0.76 | 1.12 | 100% |
| fiqa | 57,638 | 10 | compiled, document-at-a-time | 1.48 | 1.42 | 2.89 | 100% |
| fiqa | 57,638 | 10 | compiled + MaxScore | 0.34 | 0.26 | 0.89 | 7% |
| fiqa | 57,638 | 100 | NumPy, term-at-a-time | 0.79 | 0.79 | 1.16 | 100% |
| fiqa | 57,638 | 100 | compiled, document-at-a-time | 1.56 | 1.50 | 2.96 | 100% |
| fiqa | 57,638 | 100 | compiled + MaxScore | 0.70 | 0.59 | 1.57 | 13% |
| quora | 522,931 | 10 | NumPy, term-at-a-time | 4.32 | 4.30 | 6.23 | 100% |
| quora | 522,931 | 10 | compiled, document-at-a-time | 7.32 | 6.61 | 14.95 | 100% |
| quora | 522,931 | 10 | compiled + MaxScore | 0.82 | 0.45 | 2.65 | 5% |
| quora | 522,931 | 100 | NumPy, term-at-a-time | 4.35 | 4.32 | 6.25 | 100% |
| quora | 522,931 | 100 | compiled, document-at-a-time | 7.43 | 6.72 | 15.05 | 100% |
| quora | 522,931 | 100 | compiled + MaxScore | 2.08 | 1.46 | 5.66 | 11% |
