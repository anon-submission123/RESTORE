"""Summarize Base, Symmetric, and RESTORE results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from training_engine import (
    DATASET_PATHS,
    DEFAULT_ARTIFACTS_ROOT,
    EMBEDDING_FILENAMES,
    METRICS,
    METHODS,
    REPOSITORY_ROOT,
    SEEDS,
    run_root,
)


DEFAULT_OUTPUT_DIRECTORY = REPOSITORY_ROOT / "results"


def load_rows(artifacts_root: Path) -> list[dict]:
    rows = []
    for upstream in EMBEDDING_FILENAMES:
        for dataset in DATASET_PATHS:
            for seed in SEEDS:
                for method in METHODS:
                    path = run_root(
                        artifacts_root,
                        dataset,
                        upstream,
                        seed,
                        method,
                    ) / "test_metrics.json"
                    value = json.loads(path.read_text())
                    rows.append(
                        {
                            "upstream": upstream,
                            "dataset": dataset,
                            "seed": seed,
                            "method": method,
                            **{name: float(value[name]) for name in METRICS},
                        }
                    )
    return rows


def aggregate(rows: list[dict]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    output = []
    for (upstream, dataset, method), part in frame.groupby(
        ["upstream", "dataset", "method"],
        sort=False,
    ):
        row = {
            "upstream": upstream,
            "dataset": dataset,
            "method": method,
            "seeds": len(part),
        }
        for metric in METRICS:
            row[f"{metric}_mean"] = float(part[metric].mean())
            row[f"{metric}_std"] = float(part[metric].std(ddof=1))
        output.append(row)
    order = {name: index for index, name in enumerate(METHODS)}
    table = pd.DataFrame(output)
    table["_method_order"] = table["method"].map(order)
    return table.sort_values(
        ["upstream", "dataset", "_method_order"]
    ).drop(columns="_method_order")


def write_outputs(table: pd.DataFrame, output_directory: Path) -> None:
    output_directory.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_directory / "summary_results.csv", index=False)
    lines = [
        "# Summary Results",
        "",
        "Values are mean ± sample standard deviation over seeds 2024, 2025, and 2026.",
        "",
        "| Upstream | Dataset | Method | Recall@10 | NDCG@10 | Recall@20 | NDCG@20 |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for _, row in table.iterrows():
        values = [
            f"{row[f'{metric}_mean']:.6f} ± {row[f'{metric}_std']:.6f}"
            for metric in METRICS
        ]
        lines.append(
            f"| {row.upstream} | {row.dataset} | {row.method} | "
            + " | ".join(values)
            + " |"
        )
    lines.append("")
    (output_directory / "summary_results.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize the three-method results.")
    parser.add_argument("--artifacts-root", type=Path, default=DEFAULT_ARTIFACTS_ROOT)
    parser.add_argument("--output-directory", type=Path, default=DEFAULT_OUTPUT_DIRECTORY)
    args = parser.parse_args()
    table = aggregate(load_rows(args.artifacts_root))
    write_outputs(table, args.output_directory)
    print(f"CSV={args.output_directory / 'summary_results.csv'}")
    print(f"MARKDOWN={args.output_directory / 'summary_results.md'}")


if __name__ == "__main__":
    main()
