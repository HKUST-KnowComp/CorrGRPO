#!/usr/bin/env python3
"""Build a deterministic SFT subset from the RLLA RL parquet files.

RLLA RL rows contain a ``prompt`` (system/user messages) and the supervised
target in ``extra_info.output``.  This script appends the target as an
assistant message and writes parquet files accepted by VERL's
``MultiTurnSFTDataset``.
"""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_SOURCE_DIR = Path(__file__).resolve().parents[1] / "data"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "data" / "sft_400"


def _normalize_messages(prompt: Any, *, row_position: int) -> list[dict[str, Any]]:
    """Convert a parquet prompt cell to a validated list of chat messages."""
    if isinstance(prompt, np.ndarray):
        prompt = prompt.tolist()
    elif isinstance(prompt, tuple):
        prompt = list(prompt)

    if not isinstance(prompt, list) or not prompt:
        raise ValueError(f"row {row_position}: prompt must be a non-empty message list")

    messages: list[dict[str, Any]] = []
    for message_position, raw_message in enumerate(prompt):
        if not isinstance(raw_message, Mapping):
            raise TypeError(
                f"row {row_position}, message {message_position}: expected a mapping, "
                f"got {type(raw_message).__name__}"
            )
        role = raw_message.get("role")
        content = raw_message.get("content")
        if not isinstance(role, str) or not role:
            raise ValueError(f"row {row_position}, message {message_position}: invalid role")
        if not isinstance(content, str):
            raise TypeError(f"row {row_position}, message {message_position}: content must be a string")
        messages.append(dict(raw_message))

    if messages[-1]["role"] == "assistant":
        raise ValueError(f"row {row_position}: prompt already ends with an assistant message")
    return messages


def _get_assistant_target(extra_info: Any, *, row_position: int) -> str:
    if not isinstance(extra_info, Mapping):
        raise TypeError(f"row {row_position}: extra_info must be a mapping")
    target = extra_info.get("output")
    if not isinstance(target, str) or not target:
        raise ValueError(f"row {row_position}: extra_info.output must be a non-empty string")
    return target


def convert_rows(source: pd.DataFrame, row_positions: Sequence[int], *, source_split: str) -> pd.DataFrame:
    """Convert selected RL rows into VERL multi-turn SFT rows."""
    required_columns = {"prompt", "extra_info"}
    missing_columns = required_columns.difference(source.columns)
    if missing_columns:
        raise ValueError(f"missing required columns: {sorted(missing_columns)}")

    records: list[dict[str, Any]] = []
    for row_position in row_positions:
        row = source.iloc[int(row_position)]
        messages = _normalize_messages(row["prompt"], row_position=int(row_position))
        target = _get_assistant_target(row["extra_info"], row_position=int(row_position))
        messages.append({"role": "assistant", "content": target})
        records.append(
            {
                "messages": messages,
                "source_split": source_split,
                "source_position": int(row_position),
            }
        )

    return pd.DataFrame.from_records(records, columns=["messages", "source_split", "source_position"])


def select_train_positions(dataset_size: int, train_size: int, seed: int) -> list[int]:
    if train_size <= 0:
        raise ValueError("train_size must be positive")
    if train_size > dataset_size:
        raise ValueError(f"requested {train_size} rows, but source train data has only {dataset_size}")
    rng = np.random.default_rng(seed)
    # Preserve source order in the written parquet; the SFT dataloader shuffles
    # training samples independently on every epoch.
    return sorted(int(position) for position in rng.choice(dataset_size, size=train_size, replace=False))


def _write_parquet_atomic(dataframe: pd.DataFrame, destination: Path) -> None:
    temporary = destination.with_name(f".{destination.name}.tmp")
    dataframe.to_parquet(temporary, index=False)
    os.replace(temporary, destination)


def build_dataset(
    source_train: Path,
    source_val: Path,
    output_dir: Path,
    *,
    train_size: int,
    seed: int,
    overwrite: bool,
) -> dict[str, Any]:
    source_train = source_train.expanduser().resolve()
    source_val = source_val.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    train_output = output_dir / "train.parquet"
    val_output = output_dir / "test.parquet"
    manifest_output = output_dir / "manifest.json"

    for source_path in (source_train, source_val):
        if not source_path.is_file():
            raise FileNotFoundError(f"source parquet not found: {source_path}")

    existing_outputs = [path for path in (train_output, val_output, manifest_output) if path.exists()]
    if existing_outputs and not overwrite:
        joined = ", ".join(str(path) for path in existing_outputs)
        raise FileExistsError(f"output already exists ({joined}); pass --overwrite to rebuild")

    train_source_df = pd.read_parquet(source_train)
    val_source_df = pd.read_parquet(source_val)
    selected_positions = select_train_positions(len(train_source_df), train_size, seed)

    train_sft_df = convert_rows(train_source_df, selected_positions, source_split="train")
    val_sft_df = convert_rows(val_source_df, range(len(val_source_df)), source_split="test")

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_parquet_atomic(train_sft_df, train_output)
    _write_parquet_atomic(val_sft_df, val_output)

    manifest = {
        "format": "verl_multiturn_sft",
        "source_train": str(source_train),
        "source_val": str(source_val),
        "source_train_rows": len(train_source_df),
        "source_val_rows": len(val_source_df),
        "sft_train_rows": len(train_sft_df),
        "sft_val_rows": len(val_sft_df),
        "seed": seed,
        "selected_train_positions": selected_positions,
        "target_field": "extra_info.output",
    }
    temporary_manifest = manifest_output.with_name(f".{manifest_output.name}.tmp")
    temporary_manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary_manifest, manifest_output)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-train", type=Path, default=DEFAULT_SOURCE_DIR / "train.parquet")
    parser.add_argument("--source-val", type=Path, default=DEFAULT_SOURCE_DIR / "test.parquet")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--train-size", type=int, default=400)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = build_dataset(
        args.source_train,
        args.source_val,
        args.output_dir,
        train_size=args.train_size,
        seed=args.seed,
        overwrite=args.overwrite,
    )
    print(
        f"Created RLLA SFT dataset: train={manifest['sft_train_rows']}, "
        f"validation={manifest['sft_val_rows']}, seed={manifest['seed']}"
    )
    print(f"Output directory: {args.output_dir.expanduser().resolve()}")


if __name__ == "__main__":
    main()
