from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from _eval.work_item import WorkItem
import pyarrow.parquet as pq

def strip_code_fence(text: str) -> str:
    text = text.strip()
    # Consume only horizontal whitespace before the optional newline. Using
    # \s here would also eat indentation from the first code line.
    matches = re.findall(
        r"```(?:python|py)?[ \t]*(?:\r?\n)?(.*?)```", text, flags=re.I | re.S
    )
    return matches[-1].strip("\n") if matches else text

def normalize_humaneval(text: str, prompt: str, entry_point: str) -> str:
    code = strip_code_fence(text)
    if code.startswith(prompt):
        code = code[len(prompt) :]

    lines = code.splitlines()
    definition = re.compile(rf"^\s*(?:async\s+)?def\s+{re.escape(entry_point)}\s*\(")
    def_index = next((i for i, line in enumerate(lines) if definition.search(line)), None)
    if def_index is not None:
        body = lines[def_index + 1 :]
        nonempty = [line for line in body if line.strip()]
        if nonempty:
            indent = min(len(line) - len(line.lstrip()) for line in nonempty)
            lines = [line[indent:] if line.strip() else "" for line in body]

    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    if not lines:
        raise ValueError("HumanEval completion is empty")
    if not lines[0].startswith((" ", "\t")):
        lines = [("    " + line) if line else line for line in lines]
    return "\n".join(lines) + "\n"

def load_humaneval(base: Path, tokenizer: Any) -> list[WorkItem]:
    files = sorted((base / "humaneval" / "data").rglob("*.parquet"))
    if not files:
        raise FileNotFoundError("HumanEval parquet snapshot not found")
    system = (
        "You are an expert Python programmer completing HumanEval functions. "
        "Return only the missing Python function body that can be appended verbatim "
        "to the prompt. Do not repeat the function signature or prompt. "
        "Do not use Markdown fences or explanations."
    )
    items: list[WorkItem] = []
    for path in files:
        table = pq.read_table(path, columns=["task_id", "prompt", "entry_point"])
        for row in table.to_pylist():
            source_prompt = str(row["prompt"])
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": source_prompt},
            ]
            rendered = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            items.append(
                WorkItem(
                    item_id=str(row["task_id"]),
                    prompt=rendered,
                    source_prompt=source_prompt,
                    entry_point=str(row["entry_point"]),
                )
            )
    return items
