#!/usr/bin/env python3
"""Run AgentDojo with its official task checkers and report official metrics.

For a real local model, use ``run_official_metrics.sh --model local``. The
wrapper automatically reuses or starts an OpenAI-compatible vLLM server on
GPU1. ``--smoke`` uses a no-op pipeline to exercise the official environment,
attack, and checker plumbing without requiring a model server.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import re
import sys
import threading
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from statistics import mean
from typing import Any, Sequence

# Prefer the vendored official package snapshot when this script is run from
# the prepared GPU1 directory.  It falls back to the active environment when
# the snapshot has not been copied yet.
_ROOT = Path(__file__).resolve().parent
_VENDOR = _ROOT / "vendor"
if _VENDOR.is_dir():
    sys.path.insert(0, str(_VENDOR))

from agentdojo.agent_pipeline.agent_pipeline import AgentPipeline, PipelineConfig
from agentdojo.agent_pipeline.base_pipeline_element import BasePipelineElement
from agentdojo.attacks.attack_registry import load_attack
from agentdojo.benchmark import (
    benchmark_suite_with_injections,
    benchmark_suite_without_injections,
    load_task_results,
    run_task_with_injection_tasks,
    run_task_without_injection_tasks,
)
from agentdojo.functions_runtime import EmptyEnv, Env, FunctionsRuntime
from agentdojo.logging import Logger, OutputLogger
from agentdojo.task_suite.load_suites import get_suite
from agentdojo.types import ChatMessage, text_content_block_from_string


class NoOpPipeline(BasePipelineElement):
    """A deterministic pipeline used only for checker smoke tests."""

    # Include a model-family token so AgentDojo's official fixed attack
    # formatter can resolve the model name during smoke tests.
    name = "local noop-checker-smoke"

    def query(
        self,
        query: str,
        runtime: FunctionsRuntime,
        env: Env = EmptyEnv(),
        messages: Sequence[ChatMessage] = (),
        extra_args: dict[str, Any] | None = None,
    ) -> tuple[str, FunctionsRuntime, Env, Sequence[ChatMessage], dict[str, Any]]:
        message_history = list(messages)
        if not message_history:
            message_history.append(
                {
                    "role": "user",
                    "content": [text_content_block_from_string(query)],
                }
            )
        assistant_message = {
            "role": "assistant",
            "content": [text_content_block_from_string("I cannot complete this task.")],
            "tool_calls": [],
        }
        output_messages = [*message_history, assistant_message]
        Logger.get().log(output_messages)
        return query, runtime, env, output_messages, extra_args or {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-version", default="v1.2.2")
    parser.add_argument("--suites", default="workspace,banking,travel,slack")
    parser.add_argument("--mode", choices=("clean", "attack", "both"), default="both")
    parser.add_argument("--attack", default="important_instructions")
    parser.add_argument("--defense", default=None)
    parser.add_argument("--model", default="local", choices=("local", "vllm_parsed"))
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--local-port", type=int, default=8000)
    parser.add_argument("--tool-delimiter", default="tool")
    parser.add_argument("--system-message-name", default=None)
    parser.add_argument("--system-message", default=None)
    parser.add_argument("--tool-output-format", choices=("yaml", "json"), default=None)
    parser.add_argument("--logdir", type=Path, default=Path("runs"))
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument(
        "--output-md",
        type=Path,
        default=None,
        help="Markdown summary path (defaults to --output-json with a .md suffix).",
    )
    parser.add_argument("--max-user-tasks", type=int, default=None)
    parser.add_argument("--max-injection-tasks", type=int, default=None)
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help=(
            "Independent case workers; defaults to AGENTDOJO_EVAL_WORKERS (8) "
            "for model evaluation and 1 for smoke tests."
        ),
    )
    parser.add_argument("--force-rerun", action="store_true")
    parser.add_argument("--no-progress", action="store_false", dest="progress")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def average(values: Sequence[bool | float]) -> float:
    return float(mean([float(value) for value in values])) if values else 0.0


def average_or_none(values: Sequence[bool | float]) -> float | None:
    return float(mean([float(value) for value in values])) if values else None


def selected_ids(items: dict[str, Any], limit: int | None) -> list[str] | None:
    if limit is None:
        return None
    return list(items.keys())[:limit]


def model_label(model_id: str | None) -> str:
    """Return a short filesystem-safe checkpoint label for trace isolation."""
    if not model_id:
        return "auto"
    for part in Path(model_id).parts:
        if part.startswith("models--"):
            return re.sub(r"[^A-Za-z0-9_.-]+", "_", part.removeprefix("models--").replace("--", "_"))
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", Path(model_id).name)


def trace_path(
    logdir: Path,
    pipeline_name: str,
    suite_name: str,
    user_task_id: str,
    attack_name: str,
    injection_task_id: str,
) -> Path:
    safe_pipeline_name = pipeline_name.replace("/", "_")
    return logdir / safe_pipeline_name / suite_name / user_task_id / attack_name / f"{injection_task_id}.json"


class ProgressMonitor:
    """Print benchmark progress by observing completed official trace files."""

    def __init__(
        self,
        label: str,
        groups: dict[str, Sequence[Path]],
        *,
        force_rerun: bool,
        enabled: bool,
    ) -> None:
        self.label = label
        self.groups = {name: list(paths) for name, paths in groups.items() if paths}
        self.force_rerun = force_rerun
        self.enabled = enabled and bool(self.groups)
        self.started_at = 0.0
        self.baseline_mtimes: dict[Path, int] = {}
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None
        self.last_counts: tuple[int, ...] | None = None
        self.last_print_at = 0.0

    def _is_complete(self, path: Path) -> bool:
        try:
            stat = path.stat()
            if self.force_rerun and stat.st_mtime_ns <= self.baseline_mtimes.get(path, 0):
                return False
            payload = json.loads(path.read_text(encoding="utf-8"))
            return "utility" in payload and "security" in payload
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return False

    def _counts(self) -> tuple[int, ...]:
        return tuple(sum(self._is_complete(path) for path in paths) for paths in self.groups.values())

    def _print(self, counts: tuple[int, ...], *, final: bool = False) -> None:
        elapsed = int(time.monotonic() - self.started_at)
        parts = []
        for (name, paths), count in zip(self.groups.items(), counts):
            percent = 100.0 * count / len(paths)
            parts.append(f"{name} {count}/{len(paths)} ({percent:.1f}%)")
        suffix = "done" if final and all(count == len(paths) for count, paths in zip(counts, self.groups.values())) else "running"
        print(f"[progress] {self.label}: {', '.join(parts)} | {elapsed}s | {suffix}", flush=True)
        self.last_counts = counts
        self.last_print_at = time.monotonic()

    def _run(self) -> None:
        while not self.stop_event.wait(2.0):
            counts = self._counts()
            total = sum(len(paths) for paths in self.groups.values())
            completed = sum(counts)
            previous = sum(self.last_counts or ())
            step = max(1, total // 20)
            heartbeat_due = time.monotonic() - self.last_print_at >= 30
            if completed == total or completed - previous >= step or heartbeat_due:
                self._print(counts)

    def __enter__(self) -> "ProgressMonitor":
        if not self.enabled:
            return self
        self.started_at = time.monotonic()
        if self.force_rerun:
            for paths in self.groups.values():
                for path in paths:
                    try:
                        self.baseline_mtimes[path] = path.stat().st_mtime_ns
                    except FileNotFoundError:
                        self.baseline_mtimes[path] = 0
        self._print(self._counts())
        self.thread = threading.Thread(target=self._run, name="agentdojo-progress", daemon=True)
        self.thread.start()
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        if not self.enabled:
            return
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=3)
        self._print(self._counts(), final=exc_type is None)


def build_pipeline(args: argparse.Namespace) -> BasePipelineElement:
    if args.smoke:
        return NoOpPipeline()

    os.environ["LOCAL_LLM_PORT"] = str(args.local_port)
    config = PipelineConfig(
        llm=args.model,
        model_id=args.model_id,
        defense=args.defense,
        tool_delimiter=args.tool_delimiter,
        system_message_name=args.system_message_name,
        system_message=args.system_message,
        tool_output_format=args.tool_output_format,
    )
    pipeline = AgentPipeline.from_config(config)
    # Trace paths are keyed by pipeline.name in the official implementation.
    # Add the checkpoint label so a shared logdir cannot mix model caches.
    pipeline.name = f"{pipeline.name}-{model_label(args.model_id)}"
    return pipeline


def inspect_trace_validity(
    *,
    suite_name: str,
    pipeline_name: str,
    result_keys: Sequence[tuple[str, str]],
    attack_name: str,
    logdir: Path,
) -> tuple[dict[str, Any], set[tuple[str, str]]]:
    """Read official traces and separate evaluated cases from runtime errors."""
    valid_keys: set[tuple[str, str]] = set()
    invalid_cases: dict[str, str] = {}
    for user_task_id, injection_task_id in result_keys:
        case_name = f"{user_task_id}:{injection_task_id}"
        trace_injection_id = injection_task_id or "none"
        try:
            trace = load_task_results(
                pipeline_name,
                suite_name,
                user_task_id,
                attack_name,
                trace_injection_id,
                logdir,
            )
        except Exception as exc:
            invalid_cases[case_name] = f"Trace could not be loaded: {type(exc).__name__}: {exc}"[:2000]
            continue
        if trace.error is not None:
            invalid_cases[case_name] = trace.error[:2000]
            continue
        valid_keys.add((user_task_id, injection_task_id))

    num_cases = len(result_keys)
    validity = {
        "num_cases": num_cases,
        "num_valid_cases": len(valid_keys),
        "num_invalid_cases": len(invalid_cases),
        "valid_case_rate": (len(valid_keys) / num_cases) if num_cases else None,
        "invalid_cases": invalid_cases,
    }
    return validity, valid_keys


def split_work(items: Sequence[str], workers: int) -> list[list[str]]:
    """Split IDs across workers while preserving deterministic item order."""
    if workers <= 0:
        raise ValueError(f"--workers must be positive, got {workers}")
    num_chunks = min(workers, len(items))
    if num_chunks == 0:
        return []
    return [list(items[index::num_chunks]) for index in range(num_chunks)]


def worker_args(args: argparse.Namespace) -> dict[str, Any]:
    """Return a picklable evaluator configuration for spawned workers."""
    values = vars(args).copy()
    values["progress"] = False
    values["workers"] = 1
    return values


def clean_chunk_worker(
    args_values: dict[str, Any],
    suite_name: str,
    user_ids: Sequence[str],
) -> dict[str, bool]:
    """Run independent clean tasks inside one spawned process."""
    args = argparse.Namespace(**args_values)
    suite = get_suite(args.benchmark_version, suite_name)
    pipeline = build_pipeline(args)
    results: dict[str, bool] = {}
    with OutputLogger(str(args.logdir)):
        for user_id in user_ids:
            utility, _ = run_task_without_injection_tasks(
                suite,
                pipeline,
                suite.get_user_task_by_id(user_id),
                args.logdir,
                args.force_rerun,
                args.benchmark_version,
            )
            results[user_id] = bool(utility)
    return results


def capability_chunk_worker(
    args_values: dict[str, Any],
    suite_name: str,
    injection_ids: Sequence[str],
) -> dict[str, bool]:
    """Run injection goals directly, matching AgentDojo's capability check."""
    args = argparse.Namespace(**args_values)
    suite = get_suite(args.benchmark_version, suite_name)
    pipeline = build_pipeline(args)
    results: dict[str, bool] = {}
    with OutputLogger(str(args.logdir)):
        for injection_id in injection_ids:
            utility, _ = run_task_without_injection_tasks(
                suite,
                pipeline,
                suite.get_injection_task_by_id(injection_id),
                args.logdir,
                args.force_rerun,
                args.benchmark_version,
            )
            results[injection_id] = bool(utility)
    return results


def attack_chunk_worker(
    args_values: dict[str, Any],
    suite_name: str,
    user_ids: Sequence[str],
    injection_ids: Sequence[str],
) -> tuple[dict[tuple[str, str], bool], dict[tuple[str, str], bool]]:
    """Run all selected injections for a disjoint chunk of user tasks."""
    args = argparse.Namespace(**args_values)
    suite = get_suite(args.benchmark_version, suite_name)
    pipeline = build_pipeline(args)
    attack = load_attack(args.attack, suite, pipeline)
    utility_results: dict[tuple[str, str], bool] = {}
    security_results: dict[tuple[str, str], bool] = {}
    with OutputLogger(str(args.logdir)):
        for user_id in user_ids:
            utility, security = run_task_with_injection_tasks(
                suite,
                pipeline,
                suite.get_user_task_by_id(user_id),
                attack,
                args.logdir,
                args.force_rerun,
                injection_ids,
                args.benchmark_version,
            )
            utility_results.update(utility)
            security_results.update(security)
    return utility_results, security_results


def parallel_clean_results(
    suite: Any,
    args: argparse.Namespace,
    user_ids: Sequence[str],
) -> dict[str, Any]:
    """Run clean cases concurrently and return AgentDojo's SuiteResults shape."""
    chunks = split_work(user_ids, args.workers)
    results_by_id: dict[str, bool] = {}
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=len(chunks), mp_context=context) as executor:
        futures = [
            executor.submit(clean_chunk_worker, worker_args(args), suite.name, chunk)
            for chunk in chunks
        ]
        for future in as_completed(futures):
            results_by_id.update(future.result())
    return {
        "utility_results": {(user_id, ""): results_by_id[user_id] for user_id in user_ids},
        "security_results": {(user_id, ""): True for user_id in user_ids},
        "injection_tasks_utility_results": {},
    }


def parallel_attack_results(
    suite: Any,
    attack: Any,
    args: argparse.Namespace,
    user_ids: Sequence[str],
    injection_ids: Sequence[str],
) -> dict[str, Any]:
    """Run capability checks and attack trajectories across spawned workers."""
    user_chunks = split_work(user_ids, args.workers)
    capability_ids = [] if attack.is_dos_attack else list(injection_ids)
    capability_chunks = split_work(capability_ids, args.workers)
    max_workers = min(args.workers, len(user_chunks) + len(capability_chunks))
    context = multiprocessing.get_context("spawn")
    capability_results: dict[str, bool] = {}
    utility_results: dict[tuple[str, str], bool] = {}
    security_results: dict[tuple[str, str], bool] = {}

    with ProcessPoolExecutor(max_workers=max_workers, mp_context=context) as executor:
        capability_futures = [
            executor.submit(capability_chunk_worker, worker_args(args), suite.name, chunk)
            for chunk in capability_chunks
        ]
        attack_futures = [
            executor.submit(
                attack_chunk_worker,
                worker_args(args),
                suite.name,
                chunk,
                list(injection_ids),
            )
            for chunk in user_chunks
        ]
        for future in as_completed(capability_futures):
            capability_results.update(future.result())
        for future in as_completed(attack_futures):
            utility, security = future.result()
            utility_results.update(utility)
            security_results.update(security)

    if attack.is_dos_attack:
        attack_injection_ids = [next(iter(suite.injection_tasks))]
    else:
        attack_injection_ids = list(injection_ids)
    ordered_keys = [
        (user_id, injection_id)
        for user_id in user_ids
        for injection_id in attack_injection_ids
    ]
    return {
        "utility_results": {key: utility_results[key] for key in ordered_keys},
        "security_results": {key: security_results[key] for key in ordered_keys},
        "injection_tasks_utility_results": {
            injection_id: capability_results[injection_id] for injection_id in capability_ids
        },
    }


def run_clean(
    suite: Any,
    pipeline: BasePipelineElement,
    args: argparse.Namespace,
) -> dict[str, Any]:
    user_ids = selected_ids(suite.user_tasks, args.max_user_tasks)
    selected_user_ids = user_ids if user_ids is not None else list(suite.user_tasks)
    clean_paths = [
        trace_path(args.logdir, pipeline.name, suite.name, user_task_id, "none", "none")
        for user_task_id in selected_user_ids
    ]
    with ProgressMonitor(
        f"{suite.name}/clean",
        {"clean": clean_paths},
        force_rerun=args.force_rerun,
        enabled=args.progress,
    ):
        if args.workers > 1 and len(selected_user_ids) > 1:
            result = parallel_clean_results(suite, args, selected_user_ids)
        else:
            result = benchmark_suite_without_injections(
                pipeline,
                suite,
                logdir=args.logdir,
                force_rerun=args.force_rerun,
                user_tasks=user_ids,
                benchmark_version=args.benchmark_version,
            )
    utility_results = result["utility_results"]
    utilities = list(utility_results.values())
    validity, valid_keys = inspect_trace_validity(
        suite_name=suite.name,
        pipeline_name=pipeline.name,
        result_keys=list(utility_results),
        attack_name="none",
        logdir=args.logdir,
    )
    valid_utilities = [value for key, value in utility_results.items() if key in valid_keys]
    return {
        # This is the original official all-case aggregate. Runtime failures
        # remain False here for strict comparability with AgentDojo.
        "benign_utility": average(utilities),
        "benign_utility_valid_cases": average_or_none(valid_utilities),
        "num_clean_cases": len(utilities),
        "validity": validity,
        "official_utility_results": {
            f"{user_id}:{injection_id}": bool(value)
            for (user_id, injection_id), value in utility_results.items()
        },
    }


def run_attack(
    suite: Any,
    pipeline: BasePipelineElement,
    args: argparse.Namespace,
) -> dict[str, Any]:
    if args.attack == "none":
        raise ValueError("--mode attack/both requires --attack other than 'none'.")

    attack = load_attack(args.attack, suite, pipeline)
    user_ids = selected_ids(suite.user_tasks, args.max_user_tasks)
    injection_ids = selected_ids(suite.injection_tasks, args.max_injection_tasks)
    selected_user_ids = user_ids if user_ids is not None else list(suite.user_tasks)
    selected_injection_ids = injection_ids if injection_ids is not None else list(suite.injection_tasks)
    if attack.is_dos_attack:
        attack_injection_ids = [next(iter(suite.injection_tasks))]
        capability_paths: list[Path] = []
    else:
        attack_injection_ids = selected_injection_ids
        capability_paths = [
            trace_path(args.logdir, pipeline.name, suite.name, injection_task_id, "none", "none")
            for injection_task_id in selected_injection_ids
        ]
    attack_paths = [
        trace_path(args.logdir, pipeline.name, suite.name, user_task_id, args.attack, injection_task_id)
        for user_task_id in selected_user_ids
        for injection_task_id in attack_injection_ids
    ]
    with ProgressMonitor(
        f"{suite.name}/attack",
        {"capability": capability_paths, "attack": attack_paths},
        force_rerun=args.force_rerun,
        enabled=args.progress,
    ):
        if args.workers > 1 and len(selected_user_ids) + len(selected_injection_ids) > 1:
            result = parallel_attack_results(
                suite,
                attack,
                args,
                selected_user_ids,
                selected_injection_ids,
            )
        else:
            result = benchmark_suite_with_injections(
                pipeline,
                suite,
                attack,
                logdir=args.logdir,
                force_rerun=args.force_rerun,
                user_tasks=user_ids,
                injection_tasks=injection_ids,
                benchmark_version=args.benchmark_version,
            )

    utility_results = result["utility_results"]
    security_results = result["security_results"]
    user_utility = list(utility_results.values())
    attack_success = list(security_results.values())
    # AgentDojo's BaseInjectionTask.security() is the raw predicate "did the
    # injection goal execute?".  Therefore this is ASR, not safety-pass rate.
    joint_success = [utility_results[key] and not security_results[key] for key in utility_results]

    validity, valid_keys = inspect_trace_validity(
        suite_name=suite.name,
        pipeline_name=pipeline.name,
        result_keys=list(utility_results),
        attack_name=args.attack,
        logdir=args.logdir,
    )
    valid_utility = [value for key, value in utility_results.items() if key in valid_keys]
    valid_attack_success = [value for key, value in security_results.items() if key in valid_keys]
    valid_joint_success = [
        utility_results[key] and not security_results[key] for key in utility_results if key in valid_keys
    ]

    capability_results = result["injection_tasks_utility_results"]
    capability_keys = [(injection_task_id, "") for injection_task_id in capability_results]
    capability_validity, valid_capability_keys = inspect_trace_validity(
        suite_name=suite.name,
        pipeline_name=pipeline.name,
        result_keys=capability_keys,
        attack_name="none",
        logdir=args.logdir,
    )
    valid_capability = [
        value for injection_task_id, value in capability_results.items() if (injection_task_id, "") in valid_capability_keys
    ]

    return {
        "attack": args.attack,
        # The next four metrics preserve the official all-case semantics.
        "utility_under_attack": average(user_utility),
        "utility_and_no_attack_side_effect": average(joint_success),
        "targeted_asr": average(attack_success),
        "attack_block_rate": 1.0 - average(attack_success),
        # Valid-only metrics exclude traces with official runtime errors.
        "utility_under_attack_valid_cases": average_or_none(valid_utility),
        "utility_and_no_attack_side_effect_valid_cases": average_or_none(valid_joint_success),
        "targeted_asr_valid_cases": average_or_none(valid_attack_success),
        "attack_block_rate_valid_cases": (
            1.0 - average(valid_attack_success) if valid_attack_success else None
        ),
        "num_security_cases": len(user_utility),
        "validity": validity,
        "injection_task_capability": average(list(capability_results.values())),
        "injection_task_capability_valid_cases": average_or_none(valid_capability),
        "injection_task_capability_validity": capability_validity,
        "official_injection_task_utility_results": {
            injection_task_id: bool(value) for injection_task_id, value in capability_results.items()
        },
        "official_utility_results": {
            f"{user_id}:{injection_id}": bool(value)
            for (user_id, injection_id), value in utility_results.items()
        },
        "official_raw_security_results_attack_success": {
            f"{user_id}:{injection_id}": bool(value)
            for (user_id, injection_id), value in security_results.items()
        },
    }


def write_report(report: dict[str, Any], output_path: Path | None) -> None:
    if output_path is None:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f".{output_path.name}.tmp")
    temporary_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(output_path)


def metric_cell(value: float | None, num_cases: int) -> str:
    """Format a binary average as both a percentage and an exact count."""
    if value is None or num_cases <= 0:
        return "—"
    successes = round(float(value) * num_cases)
    return f"{100.0 * float(value):.2f}% ({successes}/{num_cases})"


def valid_cases_cell(clean: dict[str, Any] | None, attack: dict[str, Any] | None) -> str:
    parts: list[str] = []
    if clean is not None:
        validity = clean.get("validity", {})
        parts.append(f"clean {validity.get('num_valid_cases', 0)}/{validity.get('num_cases', 0)}")
    if attack is not None:
        validity = attack.get("validity", {})
        parts.append(f"attack {validity.get('num_valid_cases', 0)}/{validity.get('num_cases', 0)}")
    return "; ".join(parts) if parts else "—"


def render_markdown_report(report: dict[str, Any]) -> str:
    """Build one compact table from the official per-suite metrics."""
    rows: list[list[str]] = []
    totals = {
        "clean_cases": 0,
        "clean_utility": 0.0,
        "attack_cases": 0,
        "attack_utility": 0.0,
        "joint": 0.0,
        "asr": 0.0,
        "capability_cases": 0,
        "capability": 0.0,
        "clean_valid": 0,
        "attack_valid": 0,
    }

    for suite_name in report.get("suites", []):
        suite_result = report.get("results", {}).get(suite_name, {})
        clean = suite_result.get("clean")
        attack = suite_result.get("attack")

        clean_cases = int(clean.get("num_clean_cases", 0)) if clean else 0
        attack_cases = int(attack.get("num_security_cases", 0)) if attack else 0
        capability_cases = (
            int(attack.get("injection_task_capability_validity", {}).get("num_cases", 0))
            if attack
            else 0
        )

        clean_utility = clean.get("benign_utility") if clean else None
        attack_utility = attack.get("utility_under_attack") if attack else None
        joint = attack.get("utility_and_no_attack_side_effect") if attack else None
        asr = attack.get("targeted_asr") if attack else None
        block_rate = attack.get("attack_block_rate") if attack else None
        capability = attack.get("injection_task_capability") if attack else None

        rows.append(
            [
                suite_name,
                metric_cell(clean_utility, clean_cases),
                metric_cell(attack_utility, attack_cases),
                metric_cell(joint, attack_cases),
                metric_cell(asr, attack_cases),
                metric_cell(block_rate, attack_cases),
                metric_cell(capability, capability_cases),
                valid_cases_cell(clean, attack),
            ]
        )

        totals["clean_cases"] += clean_cases
        totals["attack_cases"] += attack_cases
        totals["capability_cases"] += capability_cases
        if clean_utility is not None:
            totals["clean_utility"] += float(clean_utility) * clean_cases
        if attack_utility is not None:
            totals["attack_utility"] += float(attack_utility) * attack_cases
        if joint is not None:
            totals["joint"] += float(joint) * attack_cases
        if asr is not None:
            totals["asr"] += float(asr) * attack_cases
        if capability is not None:
            totals["capability"] += float(capability) * capability_cases
        if clean:
            totals["clean_valid"] += int(clean.get("validity", {}).get("num_valid_cases", 0))
        if attack:
            totals["attack_valid"] += int(attack.get("validity", {}).get("num_valid_cases", 0))

    clean_cases = int(totals["clean_cases"])
    attack_cases = int(totals["attack_cases"])
    capability_cases = int(totals["capability_cases"])
    total_clean_utility = totals["clean_utility"] / clean_cases if clean_cases else None
    total_attack_utility = totals["attack_utility"] / attack_cases if attack_cases else None
    total_joint = totals["joint"] / attack_cases if attack_cases else None
    total_asr = totals["asr"] / attack_cases if attack_cases else None
    total_block_rate = 1.0 - total_asr if total_asr is not None else None
    total_capability = totals["capability"] / capability_cases if capability_cases else None
    valid_parts = []
    if clean_cases:
        valid_parts.append(f"clean {totals['clean_valid']}/{clean_cases}")
    if attack_cases:
        valid_parts.append(f"attack {totals['attack_valid']}/{attack_cases}")
    rows.append(
        [
            "**Weighted total**",
            metric_cell(total_clean_utility, clean_cases),
            metric_cell(total_attack_utility, attack_cases),
            metric_cell(total_joint, attack_cases),
            metric_cell(total_asr, attack_cases),
            metric_cell(total_block_rate, attack_cases),
            metric_cell(total_capability, capability_cases),
            "; ".join(valid_parts) if valid_parts else "—",
        ]
    )

    model = str(report.get("model", "unknown")).replace("|", "\\|")
    status = str(report.get("status", "unknown"))
    lines = [
        "# AgentDojo Official Metrics Summary",
        "",
        f"- Status: `{status}`",
        f"- Model: `{model}`",
        f"- Benchmark: `{report.get('benchmark_version', 'unknown')}`",
        f"- Mode: `{report.get('mode', 'unknown')}`",
        f"- Workers: `{report.get('workers', 1)}`",
        "",
        "| Suite | Clean utility | Utility under attack | Useful & safe | Targeted ASR ↓ | Attack block ↑ | Injection capability | Valid cases |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    lines.extend(
        [
            "",
            "`Useful & safe` means the user task succeeded and the injection goal did not execute. "
            "`Injection capability` tests whether the model can execute the injection goal when asked directly; "
            "a low value can make a low ASR look safer than it really is.",
        ]
    )
    if report.get("fatal_error"):
        error = str(report["fatal_error"]).replace("\n", " ")
        lines.extend(["", f"> Evaluation stopped early: `{error}`"])
    return "\n".join(lines) + "\n"


def write_markdown_report(report: dict[str, Any], output_path: Path | None) -> None:
    if output_path is None:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f".{output_path.name}.tmp")
    temporary_path.write_text(render_markdown_report(report), encoding="utf-8")
    temporary_path.replace(output_path)


def write_outputs(
    report: dict[str, Any],
    output_json: Path | None,
    output_md: Path | None,
) -> None:
    write_report(report, output_json)
    write_markdown_report(report, output_md)


def main() -> None:
    args = parse_args()
    if args.workers is None:
        args.workers = 1 if args.smoke else int(os.getenv("AGENTDOJO_EVAL_WORKERS", "8"))
    if args.workers < 1:
        raise ValueError(f"--workers must be positive, got {args.workers}")
    if args.mode in ("attack", "both") and args.attack == "none":
        raise ValueError("Use --mode clean when --attack none is selected.")
    args.logdir.mkdir(parents=True, exist_ok=True)
    output_md = args.output_md
    if output_md is None and args.output_json is not None:
        output_md = args.output_json.with_suffix(".md")

    pipeline = build_pipeline(args)
    report: dict[str, Any] = {
        "benchmark_version": args.benchmark_version,
        "suites": [name.strip() for name in args.suites.split(",") if name.strip()],
        "mode": args.mode,
        "workers": args.workers,
        "smoke": args.smoke,
        "model": getattr(pipeline, "name", None),
        "model_provider": args.model,
        "model_id": args.model_id,
        "logdir": str(args.logdir.resolve()),
        "generation": {
            "max_tokens": int(os.getenv("AGENTDOJO_MAX_TOKENS", "2048")),
            "enable_thinking": os.getenv("AGENTDOJO_ENABLE_THINKING", "0").lower()
            in {"1", "true", "yes"},
        },
        "status": "running",
        "completed_suites": [],
        "results": {},
    }
    write_outputs(report, args.output_json, output_md)

    # The official benchmark uses OutputLogger to provide the logger context
    # consumed by TraceLogger.  Keep the same context here, including for the
    # no-op smoke pipeline.
    try:
        with OutputLogger(str(args.logdir)):
            for suite_name in report["suites"]:
                suite = get_suite(args.benchmark_version, suite_name)
                suite_result: dict[str, Any] = {}
                report["results"][suite_name] = suite_result
                if args.mode in ("clean", "both"):
                    suite_result["clean"] = run_clean(suite, pipeline, args)
                    write_outputs(report, args.output_json, output_md)
                if args.mode in ("attack", "both"):
                    suite_result["attack"] = run_attack(suite, pipeline, args)
                    write_outputs(report, args.output_json, output_md)
                report["completed_suites"].append(suite_name)
                write_outputs(report, args.output_json, output_md)
    except Exception as exc:
        report["status"] = "failed"
        report["fatal_error"] = f"{type(exc).__name__}: {exc}"
        write_outputs(report, args.output_json, output_md)
        raise

    report["status"] = "complete"
    write_outputs(report, args.output_json, output_md)

    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
