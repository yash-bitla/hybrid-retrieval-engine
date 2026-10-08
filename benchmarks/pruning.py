"""Measure BM25 query latency for three search methods as the corpus grows.

    python -m benchmarks.pruning --datasets scifact fiqa quora
"""

import argparse
import json
import platform
import time
from itertools import islice
from pathlib import Path

import numpy as np

from benchmarks.run import ROOT
from hybrid_retrieval import BM25Index
from hybrid_retrieval.beir import download, load
from hybrid_retrieval.bm25 import Method

METHODS: dict[str, Method] = {
    "NumPy, term-at-a-time": "exhaustive",
    "compiled, document-at-a-time": "daat",
    "compiled + MaxScore": "maxscore",
}
KS = [10, 100]
MAX_QUERIES = 1000


def measure(index: BM25Index, queries: list[str], method: Method, k: int) -> dict[str, float]:
    latencies, scored, total = [], 0, 0
    for query in queries:
        start = time.perf_counter()
        index.search(query, k=k, method=method)
        latencies.append((time.perf_counter() - start) * 1000.0)
        if method != "exhaustive":
            scored += index.last_postings_scored
            total += index.num_postings(query)
    p50, p95 = np.percentile(latencies, [50, 95])
    return {
        "mean_ms": float(np.mean(latencies)),
        "p50_ms": float(p50),
        "p95_ms": float(p95),
        # Share of the query terms' postings that were read. 1.0 means no pruning.
        "postings_scored": scored / total if total else 1.0,
    }


def same_top_k(index: BM25Index, queries: list[str], k: int) -> float:
    """Share of queries where MaxScore returns the same k scores as the exhaustive search."""
    same = 0
    for query in queries:
        expected = [s for _, s in index.search(query, k=k)]
        got = [s for _, s in index.search(query, k=k, method="maxscore")]
        same += len(got) == len(expected) and bool(np.allclose(got, expected, rtol=1e-5))
    return same / len(queries)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["scifact", "fiqa", "quora"])
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "benchmarks" / "results")
    args = parser.parse_args()

    lines = [
        "| Dataset | Documents | k | Method | Mean ms | P50 ms | P95 ms | Postings read |",
        "|---|---:|---:|---|---:|---:|---:|---:|",
    ]
    results = []
    for name in args.datasets:
        dataset = load(download(name, args.data_dir))
        queries = list(islice(dataset.queries.values(), MAX_QUERIES))
        start = time.perf_counter()
        index = BM25Index(list(dataset.corpus), list(dataset.corpus.values()))
        build_s = time.perf_counter() - start
        index.search(queries[0], k=10, method="maxscore")  # compile before timing
        print(f"{name}: {len(index)} documents, {len(queries)} queries, built in {build_s:.1f} s")

        for k in KS:
            agreement = same_top_k(index, queries, k)
            for label, method in METHODS.items():
                row = measure(index, queries, method, k)
                results.append(
                    {"dataset": name, "documents": len(index), "queries": len(queries), "k": k,
                     "method": label, "same_top_k_as_exhaustive": agreement, **row}
                )
                read = "100%" if method == "exhaustive" else f"{row['postings_scored']:.0%}"
                lines.append(
                    f"| {name} | {len(index):,} | {k} | {label} | {row['mean_ms']:.2f} "
                    f"| {row['p50_ms']:.2f} | {row['p95_ms']:.2f} | {read} |"
                )
                print(lines[-1], f"same top k: {agreement:.1%}", flush=True)

    table = "\n".join(lines)
    payload = {
        "machine": f"{platform.system()} {platform.machine()}, Python {platform.python_version()}",
        "rows": results,
    }
    (args.out_dir / "pruning.json").write_text(json.dumps(payload, indent=2) + "\n")
    (args.out_dir / "pruning.md").write_text(table + "\n")


if __name__ == "__main__":
    main()
