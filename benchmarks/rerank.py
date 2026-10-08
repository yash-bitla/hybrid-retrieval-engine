"""Measure what cross-encoder reranking adds, and what it costs, at each rerank depth.

The first stage is the weighted hybrid from `benchmarks.run`, with the alpha that run saved.

    pip install -e ".[dev,dense]"
    python -m benchmarks.run --dataset scifact
    python -m benchmarks.rerank --dataset scifact
"""

import argparse
import gc
import json
import platform
from dataclasses import asdict
from functools import partial
from pathlib import Path

import torch

from benchmarks.run import MODEL, QUERY_PREFIX, ROOT, embed_corpus
from hybrid_retrieval import BM25Index, DenseIndex, HybridRetriever, weighted
from hybrid_retrieval.beir import download, load
from hybrid_retrieval.dense import sentence_transformer
from hybrid_retrieval.evaluate import Report, evaluate, paired_bootstrap
from hybrid_retrieval.rerank import Reranker, cross_encoder

RERANKERS = {
    "MiniLM-L6 (23M)": "cross-encoder/ms-marco-MiniLM-L-6-v2",
    "bge-reranker-base (278M)": "BAAI/bge-reranker-base",
}
DEPTHS = [10, 20, 50, 100]
FIRST_STAGE = "hybrid-weighted"


def to_row(label: str, depth: int, report: Report, baseline: Report) -> dict[str, object]:
    diff = paired_bootstrap(report.ndcg_per_query, baseline.ndcg_per_query)
    return {
        "reranker": label,
        "depth": depth,
        "ndcg_at_10": asdict(report.ndcg_at_10),
        "ndcg_gain": asdict(diff),
        "significant": diff.low > 0.0 or diff.high < 0.0,
        "latency_p50_ms": report.latency_p50_ms,
        "latency_p95_ms": report.latency_p95_ms,
    }


def to_markdown(baseline: Report, rows: list[dict[str, object]]) -> str:
    b = baseline.ndcg_at_10
    lines = [
        "| Reranker | Depth | nDCG@10 | Gain over first stage | Significant | P50 ms | P95 ms |",
        "|---|---:|---|---|---|---:|---:|",
        f"| none (first stage) | 0 | {b.mean:.3f} [{b.low:.3f}, {b.high:.3f}] | | "
        f"| {baseline.latency_p50_ms:.1f} | {baseline.latency_p95_ms:.1f} |",
    ]
    for row in rows:
        n, g = row["ndcg_at_10"], row["ndcg_gain"]
        assert isinstance(n, dict) and isinstance(g, dict)
        lines.append(
            f"| {row['reranker']} | {row['depth']} "
            f"| {n['mean']:.3f} [{n['low']:.3f}, {n['high']:.3f}] "
            f"| {g['mean']:+.3f} [{g['low']:+.3f}, {g['high']:+.3f}] "
            f"| {'yes' if row['significant'] else 'no'} "
            f"| {row['latency_p50_ms']:.1f} | {row['latency_p95_ms']:.1f} |"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="scifact")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "benchmarks" / "results")
    args = parser.parse_args()

    path = download(args.dataset, args.data_dir)
    test = load(path, "test")
    # The fusion weight was tuned on the training queries by benchmarks.run.
    alpha = float(json.loads((args.out_dir / f"{test.name}.json").read_text())["alpha"])

    bm25 = BM25Index(list(test.corpus), list(test.corpus.values()))
    encode_docs, encode_query = sentence_transformer(MODEL, query_prefix=QUERY_PREFIX)
    cache = path / f"embeddings-{MODEL.replace('/', '--')}.npy"
    embeddings, _ = embed_corpus(test, encode_docs, cache)
    dense = DenseIndex(list(test.corpus), embeddings, encode_query)
    first_stage = HybridRetriever(
        [bm25, dense], partial(weighted, weights=[1.0 - alpha, alpha]), depth=100
    )
    first_stage.search("warm up", k=1)
    baseline = evaluate(first_stage, test, name=FIRST_STAGE)

    rows = []
    for label, model_name in RERANKERS.items():
        score = cross_encoder(model_name)
        score("warm up", ["warm up"])  # load the model onto the device before timing
        for depth in DEPTHS:
            reranker = Reranker(first_stage, test.corpus, score, depth=depth)
            report = evaluate(reranker, test, name=f"{label} @ {depth}")
            rows.append(to_row(label, depth, report, baseline))
            print(rows[-1], flush=True)
        # Release this model before the next one loads. With two cross-encoders on the
        # GPU at once, the second one slowed down until the run made no progress.
        del score, reranker
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    table = to_markdown(baseline, rows)
    print(table)
    payload = {
        "dataset": test.name,
        "machine": f"{platform.system()} {platform.machine()}, Python {platform.python_version()}",
        "first_stage": {
            "retriever": FIRST_STAGE,
            "alpha": alpha,
            "ndcg_at_10": asdict(baseline.ndcg_at_10),
            "latency_p50_ms": baseline.latency_p50_ms,
            "latency_p95_ms": baseline.latency_p95_ms,
        },
        "rerankers": RERANKERS,
        "rows": rows,
    }
    (args.out_dir / f"{test.name}-rerank.json").write_text(json.dumps(payload, indent=2) + "\n")
    (args.out_dir / f"{test.name}-rerank.md").write_text(table + "\n")


if __name__ == "__main__":
    main()
