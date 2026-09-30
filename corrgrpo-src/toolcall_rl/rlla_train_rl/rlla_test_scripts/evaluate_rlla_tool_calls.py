"""Generate on RLLA test data and score outputs with reward_score/rlla.py."""

from __future__ import annotations

import argparse
import csv
import html
import io
import json
import re
from contextlib import redirect_stdout
from pathlib import Path
from statistics import fmean
from typing import Any

import pandas as pd
from transformers import AutoTokenizer

from verl.utils.reward_score import rlla


REWARD_RANGES = {
    "combined_reward_percent": ("score", -3.0, 4.0),
    "tool_call_reward_percent": ("accuracy_reward", -3.0, 3.0),
    "function_name_percent": (
        "function_name_reward",
        rlla.FUNCTION_NAME_REWARD_MIN_POSSIBLE,
        rlla.FUNCTION_NAME_REWARD_MAX_POSSIBLE,
    ),
    "parameter_name_percent": (
        "parameter_reward",
        rlla.PARAMETER_REWARD_MIN_POSSIBLE,
        rlla.PARAMETER_REWARD_MAX_POSSIBLE,
    ),
    "parameter_value_percent": (
        "values_reward",
        rlla.VALUES_REWARD_MIN_POSSIBLE,
        rlla.VALUES_REWARD_MAX_POSSIBLE,
    ),
    "format_percent": ("format_reward", 0.0, 1.0),
}

QWEN3_THINK_PREFILL = "<think> "
QWEN3_THINK_PREFILL_MODELS = frozenset(
    {
        "qwen3_grpo_4b_think",
        "qwen3-4b-think",
        "qwen3_grpo_cov_coeff_train_4b_think",
    }
)
MISSING_TOOL_CALL_CLOSE_REPAIR_MODELS = frozenset({"qwen25_7b_instruct"})
MISSING_TOOL_CALL_OPEN_REPAIR_MODELS = frozenset(
    {"qwen3_4b_think_rlla_sft_400"}
)


def reward_to_percent(value: float, minimum: float, maximum: float) -> float:
    """Linearly map one reward from its rlla.py range to [0, 100]."""
    percent = 100.0 * (float(value) - minimum) / (maximum - minimum)
    return min(100.0, max(0.0, percent))


def normalize_rewards(raw_rewards: dict[str, float]) -> dict[str, float]:
    return {
        output_name: reward_to_percent(raw_rewards[reward_name], minimum, maximum)
        for output_name, (reward_name, minimum, maximum) in REWARD_RANGES.items()
    }


def all_tool_fields_correct(raw_rewards: dict[str, float], has_tool_call: bool) -> bool:
    """True when function, parameter names, and parameter values all get full reward."""
    if not has_tool_call:
        return False
    return (
        raw_rewards["function_name_reward"]
        == rlla.FUNCTION_NAME_REWARD_MAX_POSSIBLE
        and raw_rewards["parameter_reward"] == rlla.PARAMETER_REWARD_MAX_POSSIBLE
        and raw_rewards["values_reward"] == rlla.VALUES_REWARD_MAX_POSSIBLE
    )


def score_response(
    response: str,
    ground_truth: str,
    model_name: str,
    response_length: int,
) -> dict[str, Any]:
    """Call rlla.compute_score and add percentage metrics plus exact correctness."""
    # rlla.py occasionally prints a debug sample. Keep evaluation output concise
    # without changing any part of its score calculation.
    with redirect_stdout(io.StringIO()):
        raw_rewards = rlla.compute_score(
            data_source="rlla",
            solution_str=response,
            ground_truth=ground_truth,
            extra_info={
                "experiment_name": model_name,
                "model_name": model_name,
                "response_length": response_length,
            },
        )

    has_tool_call = "<tool_call>" in ground_truth
    return {
        "raw_rewards": raw_rewards,
        "percent_rewards": normalize_rewards(raw_rewards),
        "has_tool_call": has_tool_call,
        "all_tool_fields_correct": all_tool_fields_correct(raw_rewards, has_tool_call),
    }


def as_messages(value: Any) -> list[dict[str, str]]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    return [dict(message) for message in value]


def add_qwen3_think_prefill(prompt: str, model_name: str) -> tuple[str, str]:
    """Return the prompt and assistant prefix used to activate Qwen3 thinking."""
    if model_name not in QWEN3_THINK_PREFILL_MODELS:
        return prompt, ""

    # Qwen3-Thinking's chat template already ends in ``<think>\n``.
    existing = re.search(r"<think>\s*$", prompt)
    if existing:
        return prompt, prompt[existing.start() :]

    return prompt + QWEN3_THINK_PREFILL, QWEN3_THINK_PREFILL


def repair_missing_tool_call_closing_tag(
    response: str, model_name: str, finish_reason: str | None
) -> tuple[str, bool]:
    """Close a tool-call block when a known model ends it directly with EOS."""
    if (
        model_name not in MISSING_TOOL_CALL_CLOSE_REPAIR_MODELS
        or finish_reason != "stop"
        or "<tool_call>" not in response
        or "</tool_call>" in response
    ):
        return response, False

    return response.rstrip() + "\n</tool_call>", True


def repair_missing_tool_call_opening_tag(
    response: str, model_name: str, finish_reason: str | None
) -> tuple[str, bool]:
    """Open a valid JSON tool-call block when a known model omitted the tag."""
    if (
        model_name not in MISSING_TOOL_CALL_OPEN_REPAIR_MODELS
        or finish_reason != "stop"
        or "<tool_call>" in response
        or "</tool_call>" not in response
    ):
        return response, False

    payload, remainder = response.split("</tool_call>", maxsplit=1)
    json_lines = [line.strip() for line in payload.strip().splitlines() if line.strip()]
    if not json_lines:
        return response, False
    try:
        tool_calls = [json.loads(line) for line in json_lines]
    except (json.JSONDecodeError, TypeError):
        return response, False
    if not all(
        isinstance(tool_call, dict)
        and isinstance(tool_call.get("name"), str)
        and isinstance(tool_call.get("parameters"), dict)
        for tool_call in tool_calls
    ):
        return response, False

    repaired = "<tool_call>\n" + payload.lstrip() + "</tool_call>" + remainder
    return repaired, True


def mean_metric(records: list[dict[str, Any]], metric: str) -> float:
    return fmean(record["percent_rewards"][metric] for record in records)


def build_summary(
    model_type: str,
    model_name: str,
    model_path: str,
    test_file: str,
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    tool_records = [record for record in records if record["has_tool_call"]]
    exact_count = sum(record["all_tool_fields_correct"] for record in tool_records)

    return {
        "model_type": model_type,
        "model_name": model_name,
        "qwen3_think_prefill": model_name in QWEN3_THINK_PREFILL_MODELS,
        "model_path": model_path,
        "test_file": test_file,
        "sample_count": len(records),
        "tool_call_sample_count": len(tool_records),
        "response_only_sample_count": len(records) - len(tool_records),
        "tool_call_closing_tag_repaired_count": sum(
            record.get("tool_call_closing_tag_repaired", False) for record in records
        ),
        "tool_call_opening_tag_repaired_count": sum(
            record.get("tool_call_opening_tag_repaired", False) for record in records
        ),
        "metrics_percent": {
            "combined_reward": mean_metric(tool_records, "combined_reward_percent"),
            "tool_call_reward": mean_metric(tool_records, "tool_call_reward_percent"),
            "function_name": mean_metric(tool_records, "function_name_percent"),
            "parameter_name": mean_metric(tool_records, "parameter_name_percent"),
            "parameter_value": mean_metric(tool_records, "parameter_value_percent"),
            "format": mean_metric(records, "format_percent"),
            "all_tool_fields_correct": 100.0 * exact_count / len(tool_records),
        },
        "all_tool_fields_correct_count": exact_count,
    }


def evaluate(args: argparse.Namespace) -> None:
    import multiprocessing

    # Importing VERL's reward module initializes PyTorch before vLLM starts its
    # engine process. Spawn is required; fork cannot re-initialize CUDA safely.
    multiprocessing.set_start_method("spawn", force=True)

    from vllm import LLM, SamplingParams

    dataframe = pd.read_parquet(args.test_file)
    if args.max_samples > 0:
        dataframe = dataframe.head(args.max_samples)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    prompts: list[str] = []
    assistant_prefills: list[str] = []
    ground_truths: list[str] = []
    source_positions: list[int] = []
    for _, row in dataframe.iterrows():
        messages = as_messages(row["messages"])
        ground_truths.append(messages[-1]["content"])
        source_positions.append(int(row["source_position"]))
        prompt = tokenizer.apply_chat_template(
            messages[:-1], tokenize=False, add_generation_prompt=True
        )
        prompt, assistant_prefill = add_qwen3_think_prefill(prompt, args.model_name)
        prompts.append(prompt)
        assistant_prefills.append(assistant_prefill)

    llm = LLM(
        model=args.model_path,
        tokenizer=args.model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        dtype="bfloat16",
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        enforce_eager=True,
        trust_remote_code=True,
        disable_log_stats=True,
    )
    outputs = llm.generate(
        prompts,
        SamplingParams(temperature=0.0, max_tokens=args.max_tokens, seed=42),
    )

    records: list[dict[str, Any]] = []
    for index, (output, ground_truth, source_position, assistant_prefill) in enumerate(
        zip(outputs, ground_truths, source_positions, assistant_prefills, strict=True)
    ):
        generated = output.outputs[0]
        response_token_ids = list(generated.token_ids)
        # Match verl/workers/reward_manager/naive.py exactly.
        continuation = tokenizer.decode(response_token_ids, skip_special_tokens=True)
        raw_response = assistant_prefill + continuation
        response, closing_tag_repaired = repair_missing_tool_call_closing_tag(
            raw_response, args.model_name, generated.finish_reason
        )
        response, opening_tag_repaired = repair_missing_tool_call_opening_tag(
            response, args.model_name, generated.finish_reason
        )
        scored = score_response(
            response=response,
            ground_truth=ground_truth,
            model_name=args.model_name,
            response_length=len(response_token_ids),
        )
        records.append(
            {
                "sample_index": index,
                "source_position": source_position,
                "response": response,
                "raw_response": raw_response,
                "generated_continuation": continuation,
                "assistant_prefill": assistant_prefill,
                "response_with_special_tokens": assistant_prefill
                + tokenizer.decode(response_token_ids, skip_special_tokens=False),
                "response_token_ids": response_token_ids,
                "finish_reason": generated.finish_reason,
                "stop_reason": generated.stop_reason,
                "tool_call_closing_tag_repaired": closing_tag_repaired,
                "tool_call_opening_tag_repaired": opening_tag_repaired,
                "ground_truth": ground_truth,
                "response_token_count": len(response_token_ids),
                **scored,
            }
        )

    summary = build_summary(
        args.model_type, args.model_name, args.model_path, args.test_file, records
    )
    model_output_dir = Path(args.output_dir) / args.model_name
    model_output_dir.mkdir(parents=True, exist_ok=True)
    with (model_output_dir / "predictions.jsonl").open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    (model_output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))


def ordered_summaries(result_dir: Path, model_list: Path) -> list[dict[str, Any]]:
    with model_list.open(encoding="utf-8", newline="") as stream:
        model_rows = list(csv.DictReader(stream, delimiter="|"))

    summaries = []
    for model_row in model_rows:
        path = result_dir / model_row["model_name"] / "summary.json"
        if path.is_file():
            summary = json.loads(path.read_text(encoding="utf-8"))
            summary["model_type"] = model_row["model_type"]
            summary["model_name"] = model_row["model_name"]
            summaries.append(summary)
    return summaries


def complete_experiment_groups(
    summaries: list[dict[str, Any]],
) -> list[list[dict[str, Any]]]:
    """Return adjacent model families containing all four requested stages."""
    groups: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    for summary in summaries:
        if summary["model_type"] == "base" and current:
            groups.append(current)
            current = []
        current.append(summary)
    if current:
        groups.append(current)

    expected = ["base", "sft", "grpo", "grpo_cov_coeff"]
    return [
        group
        for group in groups
        if [summary["model_type"] for summary in group] == expected
    ]


def render_complete_groups_table(groups: list[list[dict[str, Any]]]) -> list[str]:
    """Render complete groups as HTML so group boundaries can use double lines."""
    metric_columns = [
        ("Combined", "combined_reward"),
        ("Total reward", "tool_call_reward"),
        ("Function", "function_name"),
        ("Param name", "parameter_name"),
        ("Param value", "parameter_value"),
        ("Format", "format"),
        ("All correct", "all_tool_fields_correct"),
    ]
    lines = [
        "## Complete experiment groups",
        "",
        "Total reward is `accuracy_reward` from `rlla.py`. On each ",
        "`grpo_cov_coeff` row, the parenthesized delta is covariance-coefficient ",
        "minus GRPO, in percentage points.",
        "",
        "<table>",
        "<thead><tr>",
        "<th>Base family</th><th>Stage</th><th>Model</th>",
        "".join(f"<th>{label}</th>" for label, _ in metric_columns),
        "</tr></thead>",
        "<tbody>",
    ]
    for group_index, group in enumerate(groups):
        if group_index:
            lines.append(
                '<tr><td colspan="10" style="border-top:4px double; padding:0"></td></tr>'
            )

        grpo_metrics = group[2]["metrics_percent"]
        covariance_metrics = group[3]["metrics_percent"]
        deltas = {
            metric: covariance_metrics[metric] - grpo_metrics[metric]
            for _, metric in metric_columns
        }

        for row_index, summary in enumerate(group):
            metrics = summary["metrics_percent"]
            lines.append("<tr>")
            if row_index == 0:
                lines.append(
                    f'<td rowspan="4">{html.escape(summary["model_name"])}</td>'
                )
            lines.extend(
                [
                    f'<td>{html.escape(summary["model_type"])}</td>',
                    f'<td>{html.escape(summary["model_name"])}</td>',
                ]
            )
            for _, metric in metric_columns:
                value = f'{metrics[metric]:.2f}%'
                if metric == "all_tool_fields_correct":
                    value += (
                        f' ({summary["all_tool_fields_correct_count"]}/'
                        f'{summary["tool_call_sample_count"]})'
                    )
                if summary["model_type"] == "grpo_cov_coeff":
                    value += f' (Δ {deltas[metric]:+.2f} pp)'
                lines.append(f"<td>{value}</td>")
            lines.append("</tr>")
    lines.extend(["</tbody>", "</table>", ""])
    return lines


def summarize(args: argparse.Namespace) -> None:
    summaries = ordered_summaries(Path(args.summarize), Path(args.model_list))
    metric_columns = [
        ("combined_reward", "combined_reward"),
        ("total_reward", "tool_call_reward"),
        ("function_name", "function_name"),
        ("parameter_name", "parameter_name"),
        ("parameter_value", "parameter_value"),
        ("format", "format"),
        ("all_tool_fields_correct", "all_tool_fields_correct"),
    ]
    fieldnames = [
        "model_type",
        "model_name",
        "samples",
        "tool_call_samples",
        *(output_name for output_name, _ in metric_columns),
    ]

    csv_path = Path(args.summary_csv)
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for summary in summaries:
            writer.writerow(
                {
                    "model_type": summary["model_type"],
                    "model_name": summary["model_name"],
                    "samples": summary["sample_count"],
                    "tool_call_samples": summary["tool_call_sample_count"],
                    **{
                        output_name: f'{summary["metrics_percent"][metric]:.2f}'
                        for output_name, metric in metric_columns
                    },
                }
            )

    headers = [
        "Stage",
        "Model",
        "Combined",
        "Total reward",
        "Function",
        "Param name",
        "Param value",
        "Format",
        "All correct",
    ]
    lines = [
        "# RLLA tool-call evaluation",
        "",
        "All values are percentages. Total reward is `accuracy_reward` from `rlla.py`. ",
        "Tool metrics use only ground-truth tool-call samples.",
        "",
        *render_complete_groups_table(complete_experiment_groups(summaries)),
        "## All experiments",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---", "---"] + ["---:"] * (len(headers) - 2)) + " |",
    ]
    for summary in summaries:
        metrics = summary["metrics_percent"]
        lines.append(
            "| "
            + " | ".join(
                [
                    summary["model_type"],
                    summary["model_name"],
                    f'{metrics["combined_reward"]:.2f}%',
                    f'{metrics["tool_call_reward"]:.2f}%',
                    f'{metrics["function_name"]:.2f}%',
                    f'{metrics["parameter_name"]:.2f}%',
                    f'{metrics["parameter_value"]:.2f}%',
                    f'{metrics["format"]:.2f}%',
                    f'{metrics["all_tool_fields_correct"]:.2f}% '
                    f'({summary["all_tool_fields_correct_count"]}/'
                    f'{summary["tool_call_sample_count"]})',
                ]
            )
            + " |"
        )
    Path(args.summary_markdown).write_text("\n".join(lines) + "\n", encoding="utf-8")
    if args.summary_text:
        Path(args.summary_text).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-type")
    parser.add_argument("--model-name")
    parser.add_argument("--model-path")
    parser.add_argument("--test-file")
    parser.add_argument("--output-dir")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument("--summarize")
    parser.add_argument("--model-list")
    parser.add_argument("--summary-csv")
    parser.add_argument("--summary-markdown")
    parser.add_argument("--summary-text")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.summarize:
        summarize(args)
    else:
        evaluate(args)


if __name__ == "__main__":
    main()
