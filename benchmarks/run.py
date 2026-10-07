"""Evaluate every retriever on a BEIR dataset and write the results.

    python -m benchmarks.run --dataset scifact
"""

import argparse
import json
import platform
import time
from dataclasses import asdict
from pathlib import Path

from hybrid_retrieval import BM25Index
from hybrid_retrieval.beir import download, load
from hybrid_retrieval.evaluate import evaluate, to_markdown

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="scifact")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "benchmarks" / "results")
    args = parser.parse_args()

    dataset = load(download(args.dataset, args.data_dir))
    print(f"{dataset.name}: {len(dataset.corpus)} documents, {len(dataset.queries)} queries")

    start = time.perf_counter()
    index = BM25Index(list(dataset.corpus), list(dataset.corpus.values()))
    build_s = time.perf_counter() - start
    print(f"bm25 index built in {build_s:.2f} s")

    reports = [evaluate(index, dataset, name="bm25")]
    table = to_markdown(reports)
    print(table)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset": dataset.name,
        "num_documents": len(dataset.corpus),
        "machine": f"{platform.system()} {platform.machine()}, Python {platform.python_version()}",
        "index_build_s": {"bm25": round(build_s, 3)},
        "reports": [asdict(r) for r in reports],
    }
    (args.out_dir / f"{dataset.name}.json").write_text(json.dumps(payload, indent=2) + "\n")
    (args.out_dir / f"{dataset.name}.md").write_text(table + "\n")


if __name__ == "__main__":
    main()
