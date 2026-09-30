#!/usr/bin/env python3
"""Create a deterministic, difficulty-stratified cold-start SFT subset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def _allocate_samples(counts: pd.Series, num_samples: int) -> pd.Series:
    """Allocate samples proportionally with largest-remainder rounding."""
    exact = counts.astype(float) * num_samples / int(counts.sum())
    allocated = exact.astype(int)
    remaining = num_samples - int(allocated.sum())
    order = sorted(counts.index, key=lambda value: (-(exact[value] - allocated[value]), str(value)))
    for value in order[:remaining]:
        allocated[value] += 1
    return allocated


def create_subset(
    input_path: Path,
    output_path: Path,
    *,
    num_samples: int,
    seed: int,
    stratify_column: str,
) -> pd.DataFrame:
    dataframe = pd.read_parquet(input_path)
    if num_samples <= 0:
        raise ValueError(f"num_samples must be positive, got {num_samples}")
    if num_samples > len(dataframe):
        raise ValueError(f"cannot sample {num_samples} rows from a dataset with {len(dataframe)} rows")
    if stratify_column not in dataframe.columns:
        raise KeyError(f"stratify column {stratify_column!r} is not present in {input_path}")
    if dataframe[stratify_column].isna().any():
        raise ValueError(f"stratify column {stratify_column!r} contains null values")

    counts = dataframe[stratify_column].value_counts(sort=False).sort_index()
    allocations = _allocate_samples(counts, num_samples)
    if (allocations > counts).any():
        raise ValueError("a stratum was allocated more samples than it contains")

    parts = []
    for group_index, value in enumerate(counts.index):
        group = dataframe[dataframe[stratify_column] == value]
        parts.append(group.sample(n=int(allocations[value]), random_state=seed + group_index))

    subset = pd.concat(parts, axis=0).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    if len(subset) != num_samples:
        raise AssertionError(f"expected {num_samples} sampled rows, got {len(subset)}")
    if list(subset.columns) != list(dataframe.columns):
        raise AssertionError("the sampled dataset schema differs from the source schema")
    if "question_id" in dataframe.columns and dataframe["question_id"].is_unique:
        if not subset["question_id"].is_unique:
            raise AssertionError("question_id values are not unique in the sampled dataset")
        if not subset["question_id"].isin(dataframe["question_id"]).all():
            raise AssertionError("the sampled dataset contains question_id values absent from the source")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    subset.to_parquet(output_path, index=False)

    summary = {
        "input": str(input_path),
        "output": str(output_path),
        "source_rows": len(dataframe),
        "sampled_rows": len(subset),
        "seed": seed,
        "stratify_column": stratify_column,
        "source_distribution": {str(key): int(value) for key, value in counts.items()},
        "sampled_distribution": {
            str(key): int(value)
            for key, value in subset[stratify_column].value_counts().sort_index().items()
        },
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return subset


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=script_dir / "data/sft/train.parquet")
    parser.add_argument("--output", type=Path, default=script_dir / "data/sft/train_cold_start_200.parquet")
    parser.add_argument("--num-samples", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stratify-column", default="difficulty")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    create_subset(
        args.input,
        args.output,
        num_samples=args.num_samples,
        seed=args.seed,
        stratify_column=args.stratify_column,
    )


if __name__ == "__main__":
    main()
