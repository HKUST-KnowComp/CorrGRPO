#!/usr/bin/env python3
"""Aggregate all AgentDojo val-reward metrics into one comparison table."""

from __future__ import annotations

import argparse
import csv
import io
import json
import random
import re
from pathlib import Path
from typing import Any

try:
    from verl_training.summarize_val_reward_eval import build_report, read_records, render_markdown as render_model_markdown
except ModuleNotFoundError:
    from summarize_val_reward_eval import build_report, read_records, render_markdown as render_model_markdown


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, default=None)
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--output-tex", type=Path, default=None)
    parser.add_argument("--output-subset-tex", type=Path, default=None)
    parser.add_argument(
        "--refresh-metrics",
        action="store_true",
        help="Recompute each metrics.json from its samples directory before aggregation.",
    )
    return parser.parse_args()


def value(metrics: dict[str, Any], name: str) -> float | None:
    item = metrics.get(name, {})
    result = item.get("value") if isinstance(item, dict) else None
    return float(result) if result is not None else None


def model_order(name: str) -> tuple[float, int, str]:
    lower = name.lower()
    match = re.search(r"(?:^|[_-])(\d+(?:\.\d+)?)b(?:[_-]|$)", lower)
    size = float(match.group(1)) if match else float("inf")
    method = 2 if "cov_coeff" in lower else 1 if "grpo" in lower else 0
    return size, method, lower


def experiment_group(name: str) -> str:
    """Group Base/GRPO/CovCoeff runs by model size."""
    match = re.search(r"(?:^|[_-])(\d+(?:\.\d+)?)b(?:[_-]|$)", name.lower())
    return f"{match.group(1)}B" if match else name


def refresh_metrics(input_dir: Path) -> None:
    """Rebuild per-model reports so derived metrics use current definitions."""
    metric_paths = sorted(input_dir.glob("*/metrics.json"), key=lambda item: model_order(item.parent.name))
    if not metric_paths:
        raise ValueError(f"No */metrics.json files found in {input_dir}")
    for path in metric_paths:
        old_report = json.loads(path.read_text(encoding="utf-8"))
        samples_dir = path.parent / "samples"
        report = build_report(read_records(samples_dir), model=old_report.get("model"))
        write_atomic(path, json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
        write_atomic(path.parent / "metrics.md", render_model_markdown(report))


def load_rows(input_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(input_dir.glob("*/metrics.json"), key=lambda item: model_order(item.parent.name)):
        report = json.loads(path.read_text(encoding="utf-8"))
        metrics = report.get("overall", {})
        utility_under_attack = value(metrics, "utility_under_attack")
        asr = value(metrics, "asr")
        joint_accuracy = value(metrics, "joint_accuracy")
        if utility_under_attack is None or asr is None or joint_accuracy is None:
            raise ValueError(
                f"Missing utility_under_attack/asr/joint_accuracy in {path}; "
                "rerun with --refresh-metrics"
            )
        rows.append(
            {
                "name": path.parent.name,
                "model": report.get("model"),
                "clean_utility": value(metrics, "clean_utility"),
                "utility_under_attack": utility_under_attack,
                "asr": asr,
                "joint_accuracy": joint_accuracy,
                "format_valid_rate": value(metrics, "format_valid_rate"),
                "reward_valid_rate": value(metrics, "reward_valid_rate"),
                "num_clean_cases": int(metrics.get("num_clean_cases", 0)),
                "num_attack_cases": int(metrics.get("num_attack_cases", 0)),
                "source": str(path),
            }
        )
    if not rows:
        raise ValueError(f"No */metrics.json files found in {input_dir}")
    return rows


def percent(number: float | None) -> str:
    return "—" if number is None else f"{100.0 * number:.2f}%"


def display_name(name: str) -> str:
    known = {
        "Qwen2.5-3B-Instruct": "Qwen2.5-3B",
        "qwen25_3b_instruct_agentdojo_grpo": "+GRPO",
        "qwen25_3b_instruct_agentdojo_grpo_cov_coeff": "+GRPO-CovCoeff",
        "Qwen2.5-7B-Instruct": "Qwen2.5-7B",
        "qwen25_7b_instruct_agentdojo_grpo": "+GRPO",
        "qwen25_7b_instruct_agentdojo_grpo_cov_coeff": "+GRPO-CovCoeff",
    }
    return known.get(name, name.replace("_", " "))


def method_name(name: str) -> str:
    lower = name.lower()
    if "cov_coeff" in lower:
        return "GRPO-CovCoeff"
    if "grpo" in lower:
        return "GRPO"
    return "Base"


def latex_escape(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in text)


def latex_percent(number: float | None, *, style: str | None = None) -> str:
    if number is None:
        return "--"
    result = f"{100.0 * number:.2f}"
    if style == "bold":
        return rf"\textbf{{{result}}}"
    if style == "underline":
        return rf"\underline{{{result}}}"
    return result


def rank_styles(
    rows: list[dict[str, Any]],
    key: str,
    *,
    lower_is_better: bool,
    tie_seed: str,
) -> dict[str, str]:
    """Assign exactly one bold winner and, when possible, one underlined runner-up."""
    valid_rows = [row for row in rows if row[key] is not None]
    if not valid_rows:
        return {}

    choose_value = min if lower_is_better else max
    best_value = choose_value(row[key] for row in valid_rows)
    best_rows = [row for row in valid_rows if abs(row[key] - best_value) < 1e-12]
    styles: dict[str, str] = {}

    if len(best_rows) > 1:
        candidates = sorted(best_rows, key=lambda row: model_order(row["name"]))
        bold_row, underline_row = random.Random(f"{tie_seed}:{key}:best").sample(candidates, 2)
        styles[bold_row["name"]] = "bold"
        styles[underline_row["name"]] = "underline"
        return styles

    styles[best_rows[0]["name"]] = "bold"
    remaining = [row for row in valid_rows if abs(row[key] - best_value) >= 1e-12]
    if not remaining:
        return styles

    second_value = choose_value(row[key] for row in remaining)
    second_rows = [row for row in remaining if abs(row[key] - second_value) < 1e-12]
    # For a tied runner-up, prefer GRPO-CovCoeff, then GRPO, then Base.
    underline_row = max(second_rows, key=lambda row: model_order(row["name"])[1])
    styles[underline_row["name"]] = "underline"
    return styles


def render_overall_latex(rows: list[dict[str, Any]]) -> str:
    metric_keys = (
        "clean_utility",
        "utility_under_attack",
        "asr",
        "joint_accuracy",
    )
    styles: dict[tuple[str, str, str], str] = {}
    for group in {experiment_group(row["name"]) for row in rows}:
        group_rows = [row for row in rows if experiment_group(row["name"]) == group]
        for key in metric_keys:
            for name, style in rank_styles(
                group_rows,
                key,
                lower_is_better=key == "asr",
                tie_seed=f"overall:{group}",
            ).items():
                styles[(group, key, name)] = style

    def cell(row: dict[str, Any], key: str) -> str:
        return latex_percent(
            row[key],
            style=styles.get((experiment_group(row["name"]), key, row["name"])),
        )

    lines = [
        r"% Requires: \usepackage{booktabs}",
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{5pt}",
        r"\caption{AgentDojo validation results. Joint accuracy is the attack-gated average of paired clean and attacked utility, computed per attack sample and then averaged. All values are percentages; within each model-size group, the best result is bold and the second-best is underlined. For a tie at the best value, one entry is bold and one is underlined.}",
        r"\label{tab:agentdojo-val-overall}",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Model & Clean Utility $\uparrow$ & Utility under Attack $\uparrow$ & ASR $\downarrow$ & Joint Accuracy $\uparrow$ \\",
        r"\midrule",
    ]
    previous_group = None
    for row in rows:
        group = experiment_group(row["name"])
        if previous_group is not None and group != previous_group:
            lines.append(r"\midrule")
        lines.append(
            " & ".join(
                [
                    latex_escape(display_name(row["name"])),
                    cell(row, "clean_utility"),
                    cell(row, "utility_under_attack"),
                    cell(row, "asr"),
                    cell(row, "joint_accuracy"),
                ]
            )
            + r" \\"
        )
        previous_group = group
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\vspace{2pt}",
            r"\parbox{0.98\linewidth}{\footnotesize Joint Accuracy $=\frac{1}{N}\sum_{i=1}^{N}(1-A_i)(U_i^{\mathrm{clean}}+U_i^{\mathrm{attack}})/2$, where the clean result is paired by suite and user-task ID. A fixed-seed random choice resolves best-value ties; tied runner-up values underline one entry, preferring GRPO-CovCoeff. Utility and attack-success components use VERL reward gating, and all cases remain in the denominator.}",
            r"\end{table*}",
        ]
    )
    return "\n".join(lines) + "\n"


def subset_rows(input_dir: Path) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    metric_paths = sorted(input_dir.glob("*/metrics.json"), key=lambda item: model_order(item.parent.name))
    for path in metric_paths:
        report = json.loads(path.read_text(encoding="utf-8"))
        groups = [
            ("Banking", report.get("by_suite", {}).get("banking")),
            ("Slack", report.get("by_suite", {}).get("slack")),
            ("Travel", report.get("by_suite", {}).get("travel")),
            ("Workspace", report.get("by_suite", {}).get("workspace")),
            ("Important Instructions", report.get("by_attack", {}).get("important_instructions")),
            ("Tool Knowledge", report.get("by_attack", {}).get("tool_knowledge")),
            ("Total", report.get("overall")),
        ]
        for subset, metrics in groups:
            if not isinstance(metrics, dict):
                continue
            utility_under_attack = value(metrics, "utility_under_attack")
            asr = value(metrics, "asr")
            output.append(
                {
                    "name": path.parent.name,
                    "subset": subset,
                    "clean_utility": value(metrics, "clean_utility"),
                    "utility_under_attack": utility_under_attack,
                    "asr": asr,
                    "joint_accuracy": value(metrics, "joint_accuracy"),
                    "format_valid_rate": value(metrics, "format_valid_rate"),
                    "num_clean_cases": int(metrics.get("num_clean_cases", 0)),
                    "num_attack_cases": int(metrics.get("num_attack_cases", 0)),
                }
            )
    return output


def render_subset_latex(rows: list[dict[str, Any]]) -> str:
    metric_keys = (
        "clean_utility",
        "utility_under_attack",
        "asr",
        "joint_accuracy",
    )
    styles: dict[tuple[str, str, str, str], str] = {}
    comparison_groups = {(experiment_group(row["name"]), row["subset"]) for row in rows}
    for group, subset in comparison_groups:
        group_rows = [
            row
            for row in rows
            if experiment_group(row["name"]) == group and row["subset"] == subset
        ]
        for key in metric_keys:
            for name, style in rank_styles(
                group_rows,
                key,
                lower_is_better=key == "asr",
                tie_seed=f"subset:{group}:{subset}",
            ).items():
                styles[(group, subset, key, name)] = style

    def cell(row: dict[str, Any], key: str) -> str:
        return latex_percent(
            row[key],
            style=styles.get(
                (experiment_group(row["name"]), row["subset"], key, row["name"])
            ),
        )

    subset_order = {
        "Banking": 0,
        "Slack": 1,
        "Travel": 2,
        "Workspace": 3,
        "Important Instructions": 4,
        "Tool Knowledge": 5,
        "Total": 6,
    }
    scale_order = sorted({experiment_group(row["name"]) for row in rows}, key=lambda scale: float(scale[:-1]))

    lines = [
        r"% Requires: \usepackage{booktabs,multirow,graphicx}",
        r"\begin{table*}[t]",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3.5pt}",
        r"\renewcommand{\arraystretch}{0.92}",
        r"\caption{AgentDojo results by suite, attack subset, and overall total. Within each model scale and subset, the best result is bold and the second-best is underlined for every metric. For a tie at the best value, one entry is bold and one is underlined. Joint accuracy follows Table~\ref{tab:agentdojo-val-overall}; all values are percentages.}",
        r"\label{tab:agentdojo-val-subsets}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{lllrrrrr}",
        r"\toprule",
        r"Scale & Subset & Method & Clean Utility $\uparrow$ & Utility under Attack $\uparrow$ & ASR $\downarrow$ & Joint Accuracy $\uparrow$ & $N_{\mathrm{clean}}/N_{\mathrm{attack}}$ \\",
        r"\midrule",
    ]
    for scale_index, scale in enumerate(scale_order):
        scale_rows = [row for row in rows if experiment_group(row["name"]) == scale]
        subsets = sorted({row["subset"] for row in scale_rows}, key=lambda name: subset_order.get(name, 999))
        scale_row_count = len(scale_rows)
        emitted_scale_rows = 0
        for subset_index, subset in enumerate(subsets):
            group = sorted(
                [row for row in scale_rows if row["subset"] == subset],
                key=lambda row: model_order(row["name"]),
            )
            for method_index, row in enumerate(group):
                scale_cell = rf"\multirow{{{scale_row_count}}}{{*}}{{{latex_escape(scale)}}}" if emitted_scale_rows == 0 else ""
                subset_cell = rf"\multirow{{{len(group)}}}{{*}}{{{latex_escape(subset)}}}" if method_index == 0 else ""
                lines.append(
                    " & ".join(
                        [
                            scale_cell,
                            subset_cell,
                            latex_escape(method_name(row["name"])),
                            cell(row, "clean_utility"),
                            cell(row, "utility_under_attack"),
                            cell(row, "asr"),
                            cell(row, "joint_accuracy"),
                            f"{row['num_clean_cases']}/{row['num_attack_cases']}",
                        ]
                    )
                    + r" \\"
                )
                emitted_scale_rows += 1
            if subset_index + 1 < len(subsets):
                rule = r"\cmidrule(lr){2-8}"
                lines.append(rule)
                if subsets[subset_index + 1] == "Total":
                    lines.append(r"\morecmidrules")
                    lines.append(rule)
        if scale_index + 1 < len(scale_order):
            lines.append(r"\midrule")
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}%",
            r"}",
            r"\end{table*}",
        ]
    )
    return "\n".join(lines) + "\n"


def render_markdown(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# AgentDojo Val-Reward Comparison",
        "",
        "`Joint accuracy = mean_i[(1 - attack_success_i) × "
        "(paired_clean_utility_i + attack_utility_i) / 2]` over attack samples.",
        "",
        "| Model | Clean utility | Utility under attack | ASR ↓ | Joint accuracy ↑ | Format valid | Cases (clean/attack) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    row["name"].replace("|", "\\|"),
                    percent(row["clean_utility"]),
                    percent(row["utility_under_attack"]),
                    percent(row["asr"]),
                    percent(row["joint_accuracy"]),
                    percent(row["format_valid_rate"]),
                    f"{row['num_clean_cases']}/{row['num_attack_cases']}",
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "Utility and attack-success components use VERL reward gating; all cases remain in the denominator.",
        ]
    )
    return "\n".join(lines) + "\n"


def render_csv(rows: list[dict[str, Any]]) -> str:
    fields = [
        "name",
        "model",
        "clean_utility",
        "utility_under_attack",
        "asr",
        "joint_accuracy",
        "format_valid_rate",
        "reward_valid_rate",
        "num_clean_cases",
        "num_attack_cases",
        "source",
    ]
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def write_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    output_md = args.output_md or args.input_dir / "summary.md"
    output_csv = args.output_csv or args.input_dir / "summary.csv"
    output_json = args.output_json or args.input_dir / "summary.json"
    output_tex = args.output_tex or args.input_dir / "summary_table.tex"
    output_subset_tex = args.output_subset_tex or args.input_dir / "subset_table.tex"
    if args.refresh_metrics:
        refresh_metrics(args.input_dir)
    rows = load_rows(args.input_dir)
    write_atomic(output_md, render_markdown(rows))
    write_atomic(output_csv, render_csv(rows))
    write_atomic(output_tex, render_overall_latex(rows))
    write_atomic(output_subset_tex, render_subset_latex(subset_rows(args.input_dir)))
    write_atomic(
        output_json,
        json.dumps(
            {
                "joint_accuracy_formula": (
                    "mean_i[(1 - attack_success_i) * "
                    "(paired_clean_utility_i + attack_utility_i) / 2]"
                ),
                "models": rows,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
    )
    print(render_markdown(rows), end="")
    print(f"Markdown: {output_md.resolve()}")
    print(f"CSV: {output_csv.resolve()}")
    print(f"JSON: {output_json.resolve()}")
    print(f"LaTeX overall: {output_tex.resolve()}")
    print(f"LaTeX subsets: {output_subset_tex.resolve()}")


if __name__ == "__main__":
    main()
