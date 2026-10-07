"""Plot ranking quality against latency for each reranker and depth.

    pip install -e ".[bench]"
    python -m benchmarks.plot --dataset scifact
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker

from benchmarks.run import ROOT

SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SERIES = ["#2a78d6", "#eb6834"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="scifact")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "benchmarks" / "results")
    args = parser.parse_args()

    data = json.loads((args.out_dir / f"{args.dataset}-rerank.json").read_text())
    first = data["first_stage"]

    fig, ax = plt.subplots(figsize=(8, 4.8), dpi=160, facecolor=SURFACE)
    ax.set_facecolor(SURFACE)

    ax.axhline(first["ndcg_at_10"]["mean"], color=MUTED, linewidth=1, linestyle=(0, (4, 3)))
    ax.annotate(
        f"first stage, no reranker: {first['ndcg_at_10']['mean']:.3f} "
        f"at {first['latency_p95_ms']:.0f} ms",
        xy=(0, first["ndcg_at_10"]["mean"]),
        xycoords=("axes fraction", "data"),
        xytext=(0, 5),
        textcoords="offset points",
        ha="left",
        va="bottom",
        fontsize=9,
        color=MUTED,
    )

    for color, label in zip(SERIES, data["rerankers"], strict=True):
        rows = [r for r in data["rows"] if r["reranker"] == label]
        xs = [r["latency_p95_ms"] for r in rows]
        ys = [r["ndcg_at_10"]["mean"] for r in rows]
        ax.plot(
            xs, ys, color=color, linewidth=2, marker="o", markersize=8,
            markeredgecolor=SURFACE, markeredgewidth=2, label=label,
        )
        for r, x, y in zip(rows, xs, ys, strict=True):
            ax.annotate(
                f"top {r['depth']}", (x, y), xytext=(7, 7), textcoords="offset points",
                ha="left", fontsize=9, color=INK,
                bbox={"facecolor": SURFACE, "edgecolor": "none", "pad": 1},
            )

    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.set_xlabel("P95 latency per query (ms, log scale)", color=MUTED, fontsize=10)
    ax.set_ylabel("nDCG@10", color=MUTED, fontsize=10)
    ax.set_title(
        f"Reranking on {data['dataset']}: ranking quality against latency, by rerank depth",
        loc="left", fontsize=12, color=INK, pad=12,
    )
    ax.grid(True, which="major", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, labelsize=9, length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.margins(x=0.08, y=0.18)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK, loc="lower left")

    fig.tight_layout()
    fig.savefig(args.out_dir / f"{args.dataset}-rerank.png", facecolor=SURFACE)


if __name__ == "__main__":
    main()
