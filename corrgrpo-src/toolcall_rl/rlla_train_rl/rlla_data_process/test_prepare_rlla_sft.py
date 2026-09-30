from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rlla_data_process.prepare_rlla_sft import convert_rows, select_train_positions


def _make_source(rows: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "prompt": [
                np.array(
                    [
                        {"role": "system", "content": "system instruction"},
                        {"role": "user", "content": f"question {index}"},
                    ],
                    dtype=object,
                )
                for index in range(rows)
            ],
            "extra_info": [{"output": f"answer {index}"} for index in range(rows)],
        }
    )


def test_selection_is_deterministic_unique_and_sorted():
    first = select_train_positions(dataset_size=20, train_size=8, seed=42)
    second = select_train_positions(dataset_size=20, train_size=8, seed=42)

    assert first == second
    assert first == sorted(first)
    assert len(first) == len(set(first)) == 8


def test_conversion_appends_assistant_target_without_mutating_prompt():
    source = _make_source(3)
    original_prompt_length = len(source.iloc[1]["prompt"])

    converted = convert_rows(source, [1], source_split="train")

    messages = converted.iloc[0]["messages"]
    assert [message["role"] for message in messages] == ["system", "user", "assistant"]
    assert messages[-1]["content"] == "answer 1"
    assert converted.iloc[0]["source_position"] == 1
    assert len(source.iloc[1]["prompt"]) == original_prompt_length


def test_selection_rejects_more_rows_than_source():
    with pytest.raises(ValueError, match="source train data has only 3"):
        select_train_positions(dataset_size=3, train_size=4, seed=42)


def test_conversion_rejects_missing_target():
    source = _make_source(1)
    source.at[0, "extra_info"] = {"output": ""}

    with pytest.raises(ValueError, match="extra_info.output"):
        convert_rows(source, [0], source_split="train")
