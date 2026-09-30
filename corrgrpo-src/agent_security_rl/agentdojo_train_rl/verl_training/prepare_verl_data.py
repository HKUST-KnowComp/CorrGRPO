#!/usr/bin/env python3
"""Convert AgentDojo manifests to VERL parquet and generate tool schemas."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import datasets
import yaml


ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / "vendor"
if VENDOR.is_dir():
    sys.path.insert(0, str(VENDOR))

from agentdojo.task_suite.load_suites import get_suites


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data" / "verl")
    parser.add_argument("--tool-config", type=Path, default=ROOT / "verl_training" / "tool_config.yaml")
    parser.add_argument("--benchmark-version", default="v1.2.2")
    parser.add_argument("--max-train-cases", type=int, default=None)
    parser.add_argument("--max-eval-cases", type=int, default=None)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_default_system_message() -> str:
    path = VENDOR / "agentdojo" / "data" / "system_messages.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8"))["default"]


def simplify_parameters_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Convert Pydantic JSON Schema to the subset accepted by VERL tools.

    VERL validates tool properties as ``type/description/enum`` only. Runtime
    argument validation is still performed by AgentDojo's original Pydantic
    model, so this conversion changes prompting compatibility, not execution
    semantics.
    """
    definitions = schema.get("$defs", {})

    def resolve_type(prop: dict[str, Any]) -> str | list[str]:
        if "$ref" in prop:
            name = str(prop["$ref"]).rsplit("/", 1)[-1]
            return resolve_type(definitions.get(name, {"type": "object"}))
        if "type" in prop:
            return prop["type"]
        if "anyOf" in prop:
            types = []
            for branch in prop["anyOf"]:
                branch_type = resolve_type(branch)
                branch_types = branch_type if isinstance(branch_type, list) else [branch_type]
                types.extend(value for value in branch_types if value != "null")
            unique_types = list(dict.fromkeys(types))
            return unique_types[0] if len(unique_types) == 1 else unique_types or "string"
        if "enum" in prop:
            values = prop["enum"]
            if values:
                value = values[0]
                if isinstance(value, bool):
                    return "boolean"
                if isinstance(value, int):
                    return "integer"
                if isinstance(value, float):
                    return "number"
            return "string"
        return "object"

    properties = {}
    for name, prop in schema.get("properties", {}).items():
        simplified = {"type": resolve_type(prop)}
        if prop.get("description"):
            simplified["description"] = prop["description"]
        if prop.get("enum") is not None:
            simplified["enum"] = prop["enum"]
        properties[name] = simplified
    return {
        "type": "object",
        "properties": properties,
        "required": schema.get("required", []),
    }


def write_tool_config(path: Path, benchmark_version: str) -> dict[str, list[str]]:
    suites = get_suites(benchmark_version)
    unique_tools: dict[str, dict[str, Any]] = {}
    suite_tools: dict[str, list[str]] = {}
    for suite_name, suite in suites.items():
        suite_tools[suite_name] = [function.name for function in suite.tools]
        for function in suite.tools:
            schema = {
                "type": "function",
                "function": {
                    "name": function.name,
                    "description": function.description,
                    "parameters": simplify_parameters_schema(function.parameters.model_json_schema()),
                },
            }
            previous = unique_tools.get(function.name)
            if previous is not None and previous != schema:
                raise ValueError(f"Conflicting schemas for shared tool {function.name!r}")
            unique_tools[function.name] = schema

    config = {
        "tools": [
            {
                "class_name": "verl_training.agentdojo_tool.AgentDojoTool",
                "config": {"type": "native", "function_name": name},
                "tool_schema": schema,
            }
            for name, schema in sorted(unique_tools.items())
        ]
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return suite_tools


def convert_rows(
    rows: list[dict[str, Any]],
    *,
    suite_tools: dict[str, list[str]],
    system_message: str,
    limit: int | None,
) -> list[dict[str, Any]]:
    if limit is not None:
        rows = rows[:limit]
    converted = []
    for index, row in enumerate(rows):
        case = {
            "case_id": row["case_id"],
            "case_type": row["case_type"],
            "benchmark_version": row["benchmark_version"],
            "suite": row["suite"],
            "user_task_id": row["user_task_id"],
            "injection_task_id": row.get("injection_task_id"),
            "attack": row.get("attack"),
            "injections": row.get("injections") or {},
        }
        case_json = json.dumps(case, ensure_ascii=False, sort_keys=True)
        converted.append(
            {
                "data_source": "agentdojo",
                "agent_name": "agentdojo_agent",
                "prompt": [
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": row["prompt"]},
                ],
                "ability": "tool_agent_safety",
                "reward_model": {"style": "rule", "ground_truth": case_json},
                "extra_info": {
                    "index": index,
                    "split": row["split"],
                    "case_id": row["case_id"],
                    "case_type": row["case_type"],
                    "suite": row["suite"],
                    "tool_selection": suite_tools[row["suite"]],
                    "agentdojo_case_json": case_json,
                },
            }
        )
    return converted


def write_parquet(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    datasets.Dataset.from_list(rows).to_parquet(str(path))


def main() -> None:
    args = parse_args()
    suite_tools = write_tool_config(args.tool_config, args.benchmark_version)
    system_message = load_default_system_message()
    train_rows = convert_rows(
        read_jsonl(args.manifest_dir / "train_cases.jsonl"),
        suite_tools=suite_tools,
        system_message=system_message,
        limit=args.max_train_cases,
    )
    eval_rows = convert_rows(
        read_jsonl(args.manifest_dir / "eval_cases.jsonl"),
        suite_tools=suite_tools,
        system_message=system_message,
        limit=args.max_eval_cases,
    )
    write_parquet(train_rows, args.output_dir / "train.parquet")
    write_parquet(eval_rows, args.output_dir / "test.parquet")
    summary = {
        "benchmark_version": args.benchmark_version,
        "train_cases": len(train_rows),
        "eval_cases": len(eval_rows),
        "tool_count": sum(1 for _ in yaml.safe_load(args.tool_config.read_text())["tools"]),
        "train_file": str((args.output_dir / "train.parquet").resolve()),
        "eval_file": str((args.output_dir / "test.parquet").resolve()),
        "tool_config": str(args.tool_config.resolve()),
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
