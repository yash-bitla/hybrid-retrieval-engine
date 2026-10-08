"""Compare exact dense search with HNSW at several graph and search settings.

Run `benchmarks.run` for the dataset first: it caches the document embeddings.

    pip install -e ".[dev,dense,ann]"
    python -m benchmarks.ann --dataset fiqa
"""

import argparse
import json
import platform
import time
from collections.abc import Callable
from functools import partial
from pathlib import Path

import numpy as np

from benchmarks.run import MODEL, QUERY_PREFIX, ROOT
from hybrid_retrieval import ndcg_at_k
from hybrid_retrieval.beir import download, load
from hybrid_retrieval.dense import Embeddings

MS = [16, 32]
EF_SEARCHES = [8, 16, 32, 64, 128, 256]
K = 10


def query_embeddings(queries: list[str], cache: Path) -> tuple[Embeddings, float]:
    """Query embeddings and the mean time to encode one query. Cached on disk."""
    meta = cache.with_suffix(".json")
    if cache.exists():
        return np.load(cache), float(json.loads(meta.read_text())["encode_ms"])
    from hybrid_retrieval.dense import sentence_transformer

    _, encode_query = sentence_transformer(MODEL, query_prefix=QUERY_PREFIX)
    encode_query(["warm up"])
    start = time.perf_counter()
    embeddings = np.concatenate([encode_query([q]) for q in queries])
    encode_ms = (time.perf_counter() - start) * 1000.0 / len(queries)
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
    np.save(cache, embeddings)
    meta.write_text(json.dumps({"model": MODEL, "encode_ms": encode_ms}))
    return embeddings, encode_ms


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="fiqa")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "benchmarks" / "results")
    args = parser.parse_args()

    path = download(args.dataset, args.data_dir)
    dataset = load(path)
    doc_ids = list(dataset.corpus)
    tag = MODEL.replace("/", "--")
    queries, encode_ms = query_embeddings(
        list(dataset.queries.values()), path / f"query-embeddings-{tag}.npy"
    )
    docs = np.load(path / f"embeddings-{tag}.npy")
    docs /= np.linalg.norm(docs, axis=1, keepdims=True)

    # FAISS is imported after the encoder has run. Both it and PyTorch bring an OpenMP
    # runtime, and loading the two together can crash the process on macOS.
    import faiss

    from hybrid_retrieval.ann import HnswIndex

    faiss.omp_set_num_threads(1)
    qrels = [dataset.qrels[q] for q in dataset.queries]

    def ndcg(rows: np.ndarray) -> float:
        ranked = ([doc_ids[i] for i in row if i >= 0] for row in rows)
        return float(np.mean([ndcg_at_k(r, q, k=K) for r, q in zip(ranked, qrels, strict=True)]))

    def timed(
        search: Callable[[Embeddings], np.ndarray], repeats: int = 3
    ) -> tuple[np.ndarray, float, float]:
        """Results plus P50 and P95 per-query latency. Best of `repeats` passes per query."""
        rows, best = [], np.full(len(queries), np.inf)
        for r in range(repeats):
            for i, q in enumerate(queries):
                start = time.perf_counter()
                row = search(q)
                best[i] = min(best[i], (time.perf_counter() - start) * 1000.0)
                if r == 0:
                    rows.append(row)
        p50, p95 = np.percentile(best, [50, 95])
        return np.asarray(rows), float(p50), float(p95)

    def exact_search(q: Embeddings) -> np.ndarray:
        sims = docs @ q
        top = np.argpartition(-sims, K - 1)[:K]
        return top[np.argsort(-sims[top])]

    def hnsw_search(index: HnswIndex, q: Embeddings) -> np.ndarray:
        row: np.ndarray = index.search_vectors(q[None], K)[0][0]
        return row

    exact_rows, p50, p95 = timed(exact_search)
    exact_sets = [set(row) for row in exact_rows]
    rows = [{
        "index": "exact", "m": None, "ef_search": None, "recall_at_10": 1.0,
        "ndcg_at_10": ndcg(exact_rows), "p50_ms": p50, "p95_ms": p95,
        "build_s": 0.0, "size_mb": docs.nbytes / 1e6,
    }]

    for m in MS:
        start = time.perf_counter()
        index = HnswIndex(doc_ids, docs, lambda texts: queries[:0], m=m)
        build_s = time.perf_counter() - start
        size_mb = faiss.serialize_index(index._index).nbytes / 1e6
        for ef in EF_SEARCHES:
            index.ef_search = ef
            ann_rows, p50, p95 = timed(partial(hnsw_search, index))
            recall = float(np.mean(
                [len(want & set(got)) / K for want, got in zip(exact_sets, ann_rows, strict=True)]
            ))
            rows.append({
                "index": "hnsw", "m": m, "ef_search": ef, "recall_at_10": recall,
                "ndcg_at_10": ndcg(ann_rows), "p50_ms": p50, "p95_ms": p95,
                "build_s": build_s, "size_mb": size_mb,
            })
            print(rows[-1], flush=True)

    lines = [
        "| Index | M | efSearch | Recall@10 against exact | nDCG@10 | P50 ms | P95 ms "
        "| Build s | Size MB |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['index']} | {r['m'] or ''} | {r['ef_search'] or ''} "
            f"| {r['recall_at_10']:.3f} | {r['ndcg_at_10']:.3f} | {r['p50_ms']:.3f} "
            f"| {r['p95_ms']:.3f} | {r['build_s']:.1f} | {r['size_mb']:.0f} |"
        )
    table = "\n".join(lines)
    print(table)
    payload = {
        "dataset": dataset.name,
        "num_documents": len(doc_ids),
        "num_queries": len(queries),
        "dense_model": MODEL,
        "machine": f"{platform.system()} {platform.machine()}, Python {platform.python_version()}",
        "query_encode_ms": encode_ms,
        "rows": rows,
    }
    (args.out_dir / f"{dataset.name}-ann.json").write_text(json.dumps(payload, indent=2) + "\n")
    (args.out_dir / f"{dataset.name}-ann.md").write_text(table + "\n")


if __name__ == "__main__":
    main()
