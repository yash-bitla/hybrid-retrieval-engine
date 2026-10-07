import csv
import json
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

BEIR_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{name}.zip"


@dataclass(frozen=True)
class Dataset:
    name: str
    corpus: dict[str, str]
    queries: dict[str, str]
    # qrels[query_id][doc_id] = graded relevance
    qrels: dict[str, dict[str, int]]


def download(name: str, data_dir: Path) -> Path:
    """Download and unpack a BEIR dataset, unless it is already on disk."""
    target = data_dir / name
    if target.exists():
        return target
    data_dir.mkdir(parents=True, exist_ok=True)
    archive = data_dir / f"{name}.zip"
    urllib.request.urlretrieve(BEIR_URL.format(name=name), archive)
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(data_dir)
    archive.unlink()
    return target


def load(path: Path, split: str = "test") -> Dataset:
    """Load a dataset in BEIR layout: corpus.jsonl, queries.jsonl, qrels/<split>.tsv.

    A document's text is its title and body joined, which is the BEIR convention.
    Only queries that have at least one judgment in the split are kept.
    """
    corpus: dict[str, str] = {}
    with (path / "corpus.jsonl").open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            corpus[row["_id"]] = f"{row.get('title', '')} {row['text']}".strip()

    qrels: dict[str, dict[str, int]] = {}
    with (path / "qrels" / f"{split}.tsv").open(encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            qrels.setdefault(row["query-id"], {})[row["corpus-id"]] = int(row["score"])

    queries: dict[str, str] = {}
    with (path / "queries.jsonl").open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row["_id"] in qrels:
                queries[row["_id"]] = row["text"]

    return Dataset(name=path.name, corpus=corpus, queries=queries, qrels=qrels)
