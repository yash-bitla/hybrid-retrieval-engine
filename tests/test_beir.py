import json
from pathlib import Path

from hybrid_retrieval.beir import download, load


def write_dataset(root: Path) -> Path:
    path = root / "tiny"
    (path / "qrels").mkdir(parents=True)
    corpus = [
        {"_id": "d1", "title": "Cats", "text": "cats are pets"},
        {"_id": "d2", "title": "", "text": "dogs chase cats"},
    ]
    queries = [
        {"_id": "q1", "text": "what are cats"},
        {"_id": "q2", "text": "a query with no judgment in the test split"},
    ]
    (path / "corpus.jsonl").write_text("\n".join(json.dumps(r) for r in corpus) + "\n")
    (path / "queries.jsonl").write_text("\n".join(json.dumps(r) for r in queries) + "\n")
    (path / "qrels" / "test.tsv").write_text(
        "query-id\tcorpus-id\tscore\nq1\td1\t2\nq1\td2\t1\n"
    )
    return path


def test_load_parses_corpus_queries_and_qrels(tmp_path: Path) -> None:
    dataset = load(write_dataset(tmp_path))
    assert dataset.name == "tiny"
    # Title and text are joined. An empty title leaves no leading space.
    assert dataset.corpus == {"d1": "Cats cats are pets", "d2": "dogs chase cats"}
    assert dataset.qrels == {"q1": {"d1": 2, "d2": 1}}


def test_load_drops_queries_without_judgments(tmp_path: Path) -> None:
    dataset = load(write_dataset(tmp_path))
    assert dataset.queries == {"q1": "what are cats"}


def test_download_skips_when_dataset_is_on_disk(tmp_path: Path) -> None:
    path = write_dataset(tmp_path)
    # No network call happens, because the target directory exists.
    assert download("tiny", tmp_path) == path
