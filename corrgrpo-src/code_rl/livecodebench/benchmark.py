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

def normalize_lcb(text: str) -> str:
    from lcb_runner.lm_styles import LMStyle
    from lcb_runner.utils.extraction_utils import extract_code

    code = extract_code(text, LMStyle.CodeQwenInstruct)
    if not code.strip():
        code = strip_code_fence(text)
    if not code.strip():
        raise ValueError("LiveCodeBench completion is empty")
    return code.strip() + "\n"

def load_lcb(version: str, base: Path) -> list[WorkItem]:
    previous_cwd = Path.cwd()
    os.chdir(base / "livecodebench" / "vendor")
    try:
        from lcb_runner.benchmarks.code_generation import load_code_generation_dataset
        from lcb_runner.lm_styles import LMStyle
        from lcb_runner.prompts.code_generation import format_prompt_generation

        benchmark = sorted(
            load_code_generation_dataset(version), key=lambda problem: problem.question_id
        )
        return [
            WorkItem(
                item_id=str(problem.question_id),
                prompt=format_prompt_generation(problem, LMStyle.CodeQwenInstruct),
            )
            for problem in benchmark
        ]
    finally:
        os.chdir(previous_cwd)
