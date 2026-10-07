"""Evaluate every retriever on a BEIR dataset and write the results.

    pip install -e ".[dev,dense]"
    python -m benchmarks.run --dataset scifact
"""

import argparse
import json
import platform
import time
from dataclasses import asdict
from functools import partial
from pathlib import Path

import numpy as np

from hybrid_retrieval import BM25Index, DenseIndex, HybridRetriever, ndcg_at_k, rrf, weighted
from hybrid_retrieval.beir import Dataset, download, load
from hybrid_retrieval.dense import Embeddings, Encoder, sentence_transformer
from hybrid_retrieval.evaluate import Report, Retriever, evaluate, paired_bootstrap, to_markdown

ROOT = Path(__file__).resolve().parent.parent
MODEL = "BAAI/bge-small-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
ALPHAS = [round(0.1 * i, 1) for i in range(11)]


def embed_corpus(dataset: Dataset, encode: Encoder, cache: Path) -> tuple[Embeddings, float]:
    """Document embeddings and the time to compute them. Cached on disk after the first run."""
    if cache.exists():
        meta = json.loads(cache.with_suffix(".json").read_text())
        return np.load(cache), float(meta["encode_s"])
    start = time.perf_counter()
    embeddings = encode(list(dataset.corpus.values()))
    encode_s = time.perf_counter() - start
    np.save(cache, embeddings)
    cache.with_suffix(".json").write_text(json.dumps({"model": MODEL, "encode_s": encode_s}))
    return embeddings, encode_s


def tune_alpha(
    lexical: Retriever, dense: Retriever, train: Dataset, depth: int
) -> dict[str, float]:
    """Mean nDCG@10 on the training queries for each dense weight alpha.

    Each retriever runs once per query. Only the fusion is repeated per alpha.
    """
    totals = dict.fromkeys(ALPHAS, 0.0)
    for query_id, text in train.queries.items():
        candidates = [lexical.search(text, k=depth), dense.search(text, k=depth)]
        for alpha in ALPHAS:
            ranked = [doc_id for doc_id, _ in weighted(candidates, [1.0 - alpha, alpha])]
            totals[alpha] += ndcg_at_k(ranked, train.qrels[query_id], k=10)
    return {str(alpha): total / len(train.queries) for alpha, total in totals.items()}


def comparison_table(
    reports: list[Report], pairs: list[tuple[str, str]]
) -> tuple[str, list[dict[str, object]]]:
    by_name = {r.retriever: r for r in reports}
    lines = [
        "| Candidate | Baseline | nDCG@10 difference | Significant at 95% |",
        "|---|---|---|---|",
    ]
    rows: list[dict[str, object]] = []
    for candidate, baseline in pairs:
        diff = paired_bootstrap(
            by_name[candidate].ndcg_per_query, by_name[baseline].ndcg_per_query
        )
        significant = diff.low > 0.0 or diff.high < 0.0
        lines.append(
            f"| {candidate} | {baseline} | {diff.mean:+.3f} [{diff.low:+.3f}, {diff.high:+.3f}] "
            f"| {'yes' if significant else 'no'} |"
        )
        row = {"candidate": candidate, "baseline": baseline, "significant": significant}
        rows.append({**row, **asdict(diff)})
    return "\n".join(lines), rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="scifact")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "benchmarks" / "results")
    parser.add_argument("--depth", type=int, default=100, help="candidates per retriever")
    args = parser.parse_args()

    path = download(args.dataset, args.data_dir)
    test, train = load(path, "test"), load(path, "train")
    print(f"{test.name}: {len(test.corpus)} documents, {len(test.queries)} test queries, "
          f"{len(train.queries)} train queries")

    start = time.perf_counter()
    bm25 = BM25Index(list(test.corpus), list(test.corpus.values()))
    bm25_build_s = time.perf_counter() - start

    encode_docs, encode_query = sentence_transformer(MODEL, query_prefix=QUERY_PREFIX)
    cache = path / f"embeddings-{MODEL.replace('/', '--')}.npy"
    embeddings, encode_s = embed_corpus(test, encode_docs, cache)
    dense = DenseIndex(list(test.corpus), embeddings, encode_query)
    dense.search("warm up", k=1)  # load the model onto the device before timing queries

    # The fusion weight is chosen on the training queries only. The test queries are
    # used once, for the final table.
    sweep = tune_alpha(bm25, dense, train, args.depth)
    alpha = float(max(sweep, key=lambda a: sweep[a]))
    print(f"alpha sweep on train: {sweep}\nbest alpha: {alpha}")

    retrievers: dict[str, Retriever] = {
        "bm25": bm25,
        "dense": dense,
        "hybrid-rrf": HybridRetriever([bm25, dense], rrf, depth=args.depth),
        "hybrid-weighted": HybridRetriever(
            [bm25, dense], partial(weighted, weights=[1.0 - alpha, alpha]), depth=args.depth
        ),
    }
    reports = [evaluate(r, test, name=name, depth=args.depth) for name, r in retrievers.items()]
    table = to_markdown(reports)
    comparisons, comparison_rows = comparison_table(
        reports,
        [
            ("dense", "bm25"),
            ("hybrid-rrf", "bm25"),
            ("hybrid-rrf", "dense"),
            ("hybrid-weighted", "bm25"),
            ("hybrid-weighted", "dense"),
            ("hybrid-weighted", "hybrid-rrf"),
        ],
    )
    print(table, comparisons, sep="\n\n")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset": test.name,
        "num_documents": len(test.corpus),
        "machine": f"{platform.system()} {platform.machine()}, Python {platform.python_version()}",
        "dense_model": MODEL,
        "depth": args.depth,
        "index_build_s": {"bm25": round(bm25_build_s, 3), "dense": round(encode_s, 3)},
        "alpha_sweep_train_ndcg_at_10": sweep,
        "alpha": alpha,
        "reports": [
            {k: v for k, v in asdict(r).items() if k != "ndcg_per_query"} for r in reports
        ],
        "comparisons": comparison_rows,
    }
    (args.out_dir / f"{test.name}.json").write_text(json.dumps(payload, indent=2) + "\n")
    (args.out_dir / f"{test.name}.md").write_text(f"{table}\n\n{comparisons}\n")


if __name__ == "__main__":
    main()
