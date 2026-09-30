#!/usr/bin/env python3
"""Prepare AgentDojo task manifests for RL rollouts and official evaluation.

AgentDojo is an interactive environment rather than a flat prompt dataset.  The
manifests written here therefore contain task IDs and metadata; a rollout
runner should use the IDs to instantiate the official environment and checkers.
No ground-truth tool traces are written to the training manifests.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

# Prefer the vendored official package snapshot when this script is run from
# the prepared GPU1 directory.  It falls back to the active environment when
# the snapshot has not been copied yet.
_ROOT = Path(__file__).resolve().parent
_VENDOR = _ROOT / "vendor"
if _VENDOR.is_dir():
    sys.path.insert(0, str(_VENDOR))

from agentdojo.attacks.attack_registry import ATTACKS, load_attack
from agentdojo.task_suite.load_suites import get_suites


DEFAULT_ATTACKS = ("important_instructions",)


class _AttackTarget:
    """Minimal target object for rendering official attack payloads."""

    name = "local"


def stable_split(key: str, eval_ratio: float) -> str:
    """Assign all cases for the same user task to one deterministic split."""

    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    value = int(digest[:8], 16) / 2**32
    return "eval" if value < eval_ratio else "train"


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("data"))
    parser.add_argument("--benchmark-version", default="v1.2.2")
    parser.add_argument(
        "--attacks",
        default=",".join(DEFAULT_ATTACKS),
        help="Comma-separated official attack names to include in security manifests.",
    )
    parser.add_argument(
        "--eval-ratio",
        type=float,
        default=0.2,
        help="Deterministic user-task holdout ratio. This is a research split, not an official split.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.eval_ratio < 1.0:
        raise ValueError("--eval-ratio must be in [0, 1).")

    attacks = tuple(name.strip() for name in args.attacks.split(",") if name.strip())
    unknown = sorted(set(attacks) - set(ATTACKS))
    if unknown:
        raise ValueError(f"Unknown AgentDojo attacks: {unknown}. Available: {sorted(ATTACKS)}")

    suites = get_suites(args.benchmark_version)
    clean_rows: list[dict[str, Any]] = []
    security_rows: list[dict[str, Any]] = []
    suite_summary: dict[str, Any] = {}

    for suite_name, suite in suites.items():
        user_tasks = list(suite.user_tasks.values())
        injection_tasks = list(suite.injection_tasks.values())
        attack_objects = {name: load_attack(name, suite, _AttackTarget()) for name in attacks}
        suite_summary[suite_name] = {
            "user_tasks": len(user_tasks),
            "injection_tasks": len(injection_tasks),
            "injection_vectors": sorted(suite.get_injection_vector_defaults()),
        }

        for user_task in user_tasks:
            split = stable_split(f"{suite_name}:{user_task.ID}", args.eval_ratio)
            clean_rows.append(
                {
                    "case_id": f"{suite_name}:{user_task.ID}:clean",
                    "split": split,
                    "case_type": "clean",
                    "benchmark_version": args.benchmark_version,
                    "suite": suite_name,
                    "user_task_id": user_task.ID,
                    "injection_task_id": None,
                    "attack": None,
                    "prompt": user_task.PROMPT,
                    "injection_goal": None,
                }
            )

            for injection_task in injection_tasks:
                for attack_name in attacks:
                    security_rows.append(
                        {
                            "case_id": f"{suite_name}:{user_task.ID}:{injection_task.ID}:{attack_name}",
                            "split": split,
                            "case_type": "security",
                            "benchmark_version": args.benchmark_version,
                            "suite": suite_name,
                            "user_task_id": user_task.ID,
                            "injection_task_id": injection_task.ID,
                            "attack": attack_name,
                            "prompt": user_task.PROMPT,
                            "injection_goal": injection_task.GOAL,
                            # These payloads must be inserted into the
                            # official environment/tool-output injection
                            # vectors, not concatenated to the user prompt.
                            "injections": attack_objects[attack_name].attack(user_task, injection_task),
                        }
                    )

    all_rows = clean_rows + security_rows
    output_dir = args.output_dir
    write_jsonl(output_dir / "clean_tasks.jsonl", clean_rows)
    write_jsonl(output_dir / "security_cases.jsonl", security_rows)
    write_jsonl(output_dir / "all_cases.jsonl", all_rows)
    write_jsonl(output_dir / "train_cases.jsonl", [row for row in all_rows if row["split"] == "train"])
    write_jsonl(output_dir / "eval_cases.jsonl", [row for row in all_rows if row["split"] == "eval"])

    metadata = {
        "benchmark_version": args.benchmark_version,
        "attacks": list(attacks),
        "eval_ratio": args.eval_ratio,
        "split_note": "The train/eval split is deterministic and user-task grouped; AgentDojo does not define this split officially.",
        "counts": {
            "clean": len(clean_rows),
            "security": len(security_rows),
            "all": len(all_rows),
            "train": sum(row["split"] == "train" for row in all_rows),
            "eval": sum(row["split"] == "eval" for row in all_rows),
        },
        "suites": suite_summary,
    }
    (output_dir / "manifest_metadata.json").parent.mkdir(parents=True, exist_ok=True)
    (output_dir / "manifest_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
