from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from _eval.work_item import WorkItem

def strip_code_fence(text: str) -> str:
    text = text.strip()
    # Consume only horizontal whitespace before the optional newline. Using
    # \s here would also eat indentation from the first code line.
    matches = re.findall(
        r"```(?:python|py)?[ \t]*(?:\r?\n)?(.*?)```", text, flags=re.I | re.S
    )
    return matches[-1].strip("\n") if matches else text

def normalize_mbpp(text: str) -> str:
    code = strip_code_fence(text).strip()
    if not code:
        raise ValueError("MBPP completion is empty")
    return code + "\n"

def load_mbpp(subset: str, base: Path, tokenizer: Any) -> list[WorkItem]:
    root = base / "mbpp" / "data" / "data"
    if subset in {"test", "full"}:
        path = root / "mbpp.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        if subset == "test":
            rows = [row for row in rows if 11 <= int(row["task_id"]) <= 510]
    else:
        path = root / "sanitized-mbpp.json"
        rows = json.loads(path.read_text())
    system = (
        "You are an expert Python programmer solving MBPP tasks. Return only a "
        "complete Python solution, including all required imports and definitions. "
        "Do not use Markdown fences or add explanations."
    )
    items: list[WorkItem] = []
    for row in rows:
        description = str(row["text"] if subset in {"test", "full"} else row["prompt"])
        tests = "\n".join(str(test) for test in row["test_list"])
        user = f"Task:\n{description}\n\nYour solution must satisfy:\n{tests}"
        prompt = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        items.append(WorkItem(item_id=str(row["task_id"]), prompt=prompt))
    return items
