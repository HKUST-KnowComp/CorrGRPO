#!/usr/bin/env python3
"""Score API-Bank predictions with the benchmark's official execution metric.

API calls are executed through the upstream ``ToolManager`` and judged by each
API class's ``check_api_call_correctness`` implementation.  The adapter only
translates RLLA JSON tool calls into the kwargs expected by API-Bank.  Response
tasks use the upstream ``rouge==1.0.1`` Rouge-L implementation.
"""

from __future__ import annotations

import argparse
import errno
import json
import os
import random
import re
import socket
import sys
import types
from collections import Counter
from pathlib import Path
from statistics import fmean
from typing import Any


ROOT = Path(__file__).resolve().parent
DEFAULT_RUNTIME = ROOT / "official_runtime"
DEFAULT_TEST_DATA = ROOT / "test-data"
DEFAULT_OUTPUT = ROOT / "results"
DEFAULT_SEARCH_MODEL = DEFAULT_RUNTIME / "models" / "paraphrase-MiniLM-L3-v2"
UPSTREAM_COMMIT = "483554eae102996f5ec1f4feab4e78ef29c2a394"
NETWORK_POLICY = "offline_external_network_blocked"


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def dump_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def dump_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def json_safe(value: Any) -> Any:
    try:
        json.dumps(value)
        return value
    except (TypeError, ValueError):
        return repr(value)


def split_by_uppercase(value: str) -> str:
    """Match the normalization in upstream ``lv3_evaluator.py``."""
    return "".join(" " + char if char.isupper() else char for char in value).strip()


def install_googletrans_import_stub() -> None:
    """Let upstream modules load when optional googletrans is unavailable.

    The original evaluator imports every API eagerly.  Translate itself is
    still marked as an execution dependency error if a prediction reaches it;
    the stub never fabricates a translation result.
    """
    try:
        __import__("googletrans")
        return
    except (ImportError, AttributeError):
        pass

    module = types.ModuleType("googletrans")
    module.LANGUAGES = {"en": "english", "zh-cn": "chinese (simplified)"}

    class MissingTranslator:
        def __init__(self, *_: Any, **__: Any) -> None:
            raise RuntimeError(
                "googletrans is unavailable in the isolated official scorer"
            )

    module.Translator = MissingTranslator
    sys.modules["googletrans"] = module


def disable_external_network() -> None:
    """Block outbound IP sockets so benchmark examples cannot leak arguments."""
    original_connect = socket.socket.connect
    original_connect_ex = socket.socket.connect_ex

    def connect(sock: socket.socket, address: Any) -> Any:
        if sock.family in {socket.AF_INET, socket.AF_INET6}:
            raise OSError(
                errno.ENETUNREACH,
                "external network access is disabled by the evaluation policy",
            )
        return original_connect(sock, address)

    def connect_ex(sock: socket.socket, address: Any) -> int:
        if sock.family in {socket.AF_INET, socket.AF_INET6}:
            return errno.ENETUNREACH
        return original_connect_ex(sock, address)

    def create_connection(*_: Any, **__: Any) -> Any:
        raise OSError(
            errno.ENETUNREACH,
            "external network access is disabled by the evaluation policy",
        )

    socket.socket.connect = connect
    socket.socket.connect_ex = connect_ex
    socket.create_connection = create_connection


class GroundTruthIndex:
    def __init__(self, runtime_dir: Path, test_data_dir: Path) -> None:
        self.runtime_dir = runtime_dir
        self.test_data_dir = test_data_dir
        self._v12_cache: dict[tuple[str, str], list[dict[str, Any]]] = {}
        self._v3 = load_json(test_data_dir / "level-3.json")

    def _v12_samples(self, version: str, filename: str) -> list[dict[str, Any]]:
        cache_key = (version, filename)
        if cache_key in self._v12_cache:
            return self._v12_cache[cache_key]
        directory = (
            "level-1-given-desc" if version == "v1" else "level-2-toolsearcher"
        )
        history = load_jsonl(
            self.runtime_dir / "lv1-lv2-samples" / directory / filename
        )
        samples: list[dict[str, Any]] = []
        for index, item in enumerate(history):
            if item.get("role") != "API":
                continue
            samples.append(item)
            if index + 1 >= len(history):
                raise ValueError(f"API item has no following history item: {filename}:{index}")
            # Match upstream Sample.from_chat_history exactly: it reserves the
            # item after every API as the odd sample id even when that item is
            # anomalously a User turn rather than an AI response.
            samples.append(history[index + 1])
        self._v12_cache[cache_key] = samples
        return samples

    def api_ground_truth(self, row: dict[str, Any]) -> dict[str, Any]:
        version = row["version"]
        if version in {"v1", "v2"}:
            samples = self._v12_samples(version, row["file"])
            sample = samples[int(row["source_id"])]
            if sample.get("role") != "API":
                raise ValueError(
                    f"prediction points to non-API sample: {row['file']}:{row['source_id']}"
                )
            return {
                "api_name": sample["api_name"],
                "parameters": sample["param_dict"],
                "result": sample["result"],
            }

        api = self._v3[int(row["sample_id"])]["apis"][int(row["api_id"])]
        return {
            "api_name": api["api_name"],
            "parameters": api["input"],
            "result": api["output"],
        }


class OfficialExecutor:
    def __init__(self, runtime_dir: Path, search_model: Path) -> None:
        if not runtime_dir.is_dir():
            raise FileNotFoundError(f"official runtime not found: {runtime_dir}")
        if not search_model.is_dir():
            raise FileNotFoundError(f"ToolSearcher model not found: {search_model}")

        self.runtime_dir = runtime_dir.resolve()
        self.search_model_path = search_model.resolve()
        dependency_dir = self.runtime_dir / "python_packages"
        for path in (dependency_dir, self.runtime_dir, ROOT):
            if path.is_dir() and str(path) not in sys.path:
                sys.path.insert(0, str(path))
        nltk_data_dir = self.runtime_dir / "nltk_data"
        if nltk_data_dir.is_dir():
            existing_nltk_data = os.environ.get("NLTK_DATA")
            os.environ["NLTK_DATA"] = os.pathsep.join(
                value
                for value in (str(nltk_data_dir), existing_nltk_data)
                if value
            )
        install_googletrans_import_stub()
        os.chdir(self.runtime_dir)

        from sentence_transformers import SentenceTransformer, util
        from tool_manager import ToolManager

        self.ToolManager = ToolManager
        self.util = util
        self.search_model = SentenceTransformer(
            str(self.search_model_path), device="cpu"
        )
        self._catalog_embeddings: dict[tuple[str, ...], Any] = {}
        self._patch_tool_searchers()
        # Upstream lv3_evaluator.py intentionally reuses one manager across the
        # ordered trace, unlike v1/v2's fresh manager per evaluation sample.
        self.v3_manager = self.ToolManager("./lv3_apis")
        disable_external_network()

    def _patch_tool_searchers(self) -> None:
        import apis.tool_search
        import lv3_apis.tool_search

        executor = self

        def cached_best_match(instance: Any, keywords: str) -> Any:
            descriptions = tuple(api["desc_for_search"] for api in instance.apis)
            embeddings = executor._catalog_embeddings.get(descriptions)
            if embeddings is None:
                embeddings = executor.search_model.encode(
                    list(descriptions), convert_to_tensor=True
                )
                executor._catalog_embeddings[descriptions] = embeddings
            keyword_embedding = executor.search_model.encode(
                keywords, convert_to_tensor=True
            )
            similarities = executor.util.cos_sim(
                keyword_embedding, embeddings
            )[0]
            best_match = None
            best_score = 0.0
            for api, similarity in zip(instance.apis, similarities, strict=True):
                score = float(similarity.item())
                if score > best_score:
                    best_match = api.copy()
                    best_score = score
            if best_match is None:
                raise RuntimeError("ToolSearcher found no positive cosine match")
            best_match.pop("desc_for_search")
            if "token" in best_match["input_parameters"]:
                return [instance.get_user_token_api, best_match]
            return best_match

        apis.tool_search.ToolSearcher.best_match_api = cached_best_match
        lv3_apis.tool_search.ToolSearcher.best_match_api = cached_best_match

    @staticmethod
    def _normalize_kwargs(manager: Any, name: str, parameters: dict[str, Any]) -> dict[str, Any]:
        """Bridge typed RLLA JSON to API-Bank's string-oriented parser boundary."""
        schema = manager.get_api_by_name(name)["input_parameters"]
        normalized = dict(parameters)
        for key, value in list(normalized.items()):
            expected_type = schema.get(key, {}).get("type")
            if expected_type == "bool" and isinstance(value, bool):
                normalized[key] = "True" if value else "False"
        return normalized

    @staticmethod
    def _classify_exception(exc: BaseException) -> str:
        message = str(exc)
        if "invalid tool name" in message:
            return "INVALID_TOOL_NAME"
        if "invalid parameter name" in message:
            return "INVALID_INPUT_PARAMETER"
        if "required positional argument" in message:
            return "MISSING_INPUT_ARGUMENT"
        if "googletrans is unavailable" in message:
            return "OPTIONAL_DEPENDENCY_ERROR"
        if "external network access is disabled" in message:
            return "NETWORK_DISABLED"
        if isinstance(exc, KeyError):
            return "KEY_ERROR"
        if isinstance(exc, TypeError):
            return "TYPE_ERROR"
        if isinstance(exc, AssertionError):
            return "ASSERTION_ERROR"
        return "EXECUTION_ERROR"

    def execute_and_check(
        self,
        version: str,
        predicted_call: dict[str, Any],
        ground_truth: dict[str, Any],
    ) -> dict[str, Any]:
        predicted_name = predicted_call["name"]
        expected_name = ground_truth["api_name"]
        if predicted_name != expected_name:
            return {
                "correct": False,
                "error_category": "API_NAME_MISMATCH",
                "execution_result": None,
            }

        manager = self.v3_manager if version == "v3" else self.ToolManager()
        parameters = self._normalize_kwargs(
            manager, predicted_name, predicted_call["parameters"]
        )
        if version == "v3" and predicted_name == "ToolSearcher":
            keywords = parameters.get("keywords")
            if isinstance(keywords, str):
                parameters["keywords"] = split_by_uppercase(keywords)
        try:
            execution_result = manager.api_call(predicted_name, **parameters)
            ground_truth_api = manager.init_tool(expected_name)
            correct = bool(
                ground_truth_api.check_api_call_correctness(
                    execution_result, ground_truth["result"]
                )
            )
        except Exception as exc:
            return {
                "correct": False,
                "error_category": self._classify_exception(exc),
                "execution_result": None,
                "execution_exception": f"{type(exc).__name__}: {exc}",
            }
        error_category = None if correct else "RESULT_MISMATCH"
        if not correct and isinstance(execution_result, dict):
            execution_exception = str(execution_result.get("exception") or "")
            if "external network access is disabled" in execution_exception:
                error_category = "NETWORK_DISABLED"
            elif "googletrans is unavailable" in execution_exception:
                error_category = "OPTIONAL_DEPENDENCY_ERROR"
        return {
            "correct": correct,
            "error_category": error_category,
            "execution_result": json_safe(execution_result),
        }


def score_api_rows(
    rows: list[dict[str, Any]],
    executor: OfficialExecutor,
    ground_truth_index: GroundTruthIndex,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    from evaluate_api_bank import parse_prediction_tool_calls

    error_counts: Counter[str] = Counter()
    correct_count = 0
    failed_dialogues: set[str] = set()
    all_dialogues: set[str] = set()

    for row in rows:
        ground_truth = ground_truth_index.api_ground_truth(row)
        prompt_template = str(row.get("prompt_template", "rlla"))
        calls, _ = parse_prediction_tool_calls(
            row["prediction"], prompt_template
        )
        if not calls:
            result = {
                "correct": False,
                "error_category": "NO_API_CALL",
                "execution_result": None,
            }
            predicted_call = None
        else:
            predicted_call = calls[0]
            result = executor.execute_and_check(
                row["version"], predicted_call, ground_truth
            )
        if result["correct"]:
            correct_count += 1
        else:
            error_counts[result["error_category"]] += 1

        if row["version"] == "v3":
            sample_id = str(row["sample_id"])
            all_dialogues.add(sample_id)
            if not result["correct"]:
                failed_dialogues.add(sample_id)

        row["official_scores"] = {
            **result,
            "predicted_call": predicted_call,
            "ground_truth_api_name": ground_truth["api_name"],
            "ground_truth_parameters": ground_truth["parameters"],
        }

    accuracy = correct_count / len(rows) if rows else 0.0
    summary: dict[str, Any] = {
        "version": rows[0]["version"] if rows else None,
        "task": "api",
        "metric": "official_execution_accuracy",
        "sample_count": len(rows),
        "correct_count": correct_count,
        "accuracy": accuracy,
        "official_execution_accuracy": accuracy,
        "error_counts": dict(sorted(error_counts.items())),
    }
    if all_dialogues:
        dialogue_correct = len(all_dialogues) - len(failed_dialogues)
        summary.update(
            {
                "dialogue_sample_count": len(all_dialogues),
                "dialogue_success_count": dialogue_correct,
                "dialogue_success_accuracy": dialogue_correct / len(all_dialogues),
            }
        )
    return rows, summary


def score_response_rows(
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    from evaluate_api_bank import extract_prediction_response
    from rouge import Rouge

    rouge = Rouge()
    scores: list[float] = []
    for row in rows:
        prompt_template = str(row.get("prompt_template", "rlla"))
        hypothesis, _ = extract_prediction_response(
            row["prediction"],
            prompt_template,
            bool(row.get("prompt_prefills_think", False)),
        )
        if row["version"] == "v3":
            hypothesis = hypothesis.replace("User:", "").replace("AI:", "").strip()
        if hypothesis:
            score = float(rouge.get_scores(hypothesis, row["reference"])[0]["rouge-l"]["f"])
        else:
            score = 0.0
        # evaluator_by_json.py rounds each v1/v2 sample before aggregation.
        if row["version"] in {"v1", "v2"}:
            score = round(score, 4)
        scores.append(score)
        row["official_scores"] = {
            "metric": "official_rouge_l_f1",
            "rouge_l_f1": score,
            "extracted_response": hypothesis,
        }
    official_rouge_l_f1 = fmean(scores) if scores else 0.0
    summary = {
        "version": rows[0]["version"] if rows else None,
        "task": "response",
        "metric": "official_rouge_l_f1",
        "sample_count": len(rows),
        "rouge_l_f1": official_rouge_l_f1,
        "official_rouge_l_f1": official_rouge_l_f1,
    }
    return rows, summary


def diagnostic_summary(previous: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "output_format_rate",
        "official_format_rate",
        "rlla_format_rate",
        "api_call_valid_rate",
        "tool_block_valid_rate",
        "function_name_accuracy",
        "parameter_name_accuracy",
        "parameter_value_accuracy",
        "call_exact_match_accuracy",
        "response_valid_rate",
        "response_block_valid_rate",
        "response_exact_match_accuracy",
    )
    diagnostics = {}
    for key in keys:
        if key in previous:
            output_key = (
                "static_label_call_exact_match_accuracy"
                if key == "call_exact_match_accuracy"
                else key
            )
            diagnostics[output_key] = previous[key]
    return diagnostics


def render_markdown(model_name: str, summaries: list[dict[str, Any]]) -> str:
    lines = [
        f"# API-Bank official evaluation: {model_name}",
        "",
        "API scores use upstream tool execution plus each API's "
        "`check_api_call_correctness`. Response scores use upstream Rouge-L.",
        "",
        "| Version | Task | Samples | Official metric | Score | Correct |",
        "| --- | --- | ---: | --- | ---: | ---: |",
    ]
    for summary in summaries:
        if summary["task"] == "api":
            score = summary["accuracy"]
            correct = f"{summary['correct_count']}/{summary['sample_count']}"
        else:
            score = summary["rouge_l_f1"]
            correct = "—"
        lines.append(
            f"| {summary['version']} | {summary['task']} | {summary['sample_count']} | "
            f"{summary['metric']} | {100 * score:.2f}% | {correct} |"
        )
    v3_api = next(
        (
            summary
            for summary in summaries
            if summary["version"] == "v3" and summary["task"] == "api"
        ),
        None,
    )
    if v3_api is not None:
        lines.extend(
            [
                "",
                f"v3 full-dialogue success: {v3_api['dialogue_success_count']}/"
                f"{v3_api['dialogue_sample_count']} "
                f"({100 * v3_api['dialogue_success_accuracy']:.2f}%).",
            ]
        )
    return "\n".join(lines) + "\n"


def score_model(args: argparse.Namespace) -> dict[str, Any]:
    random.seed(args.seed)
    runtime_dir = Path(args.runtime_dir).resolve()
    output_root = Path(args.output_dir).resolve()
    model_root = output_root / args.model_name
    gt_index = GroundTruthIndex(runtime_dir, Path(args.test_data_dir).resolve())
    executor = OfficialExecutor(runtime_dir, Path(args.tool_search_model).resolve())

    top_summary_path = model_root / "summary.json"
    existing_top = load_json(top_summary_path) if top_summary_path.is_file() else {}
    version_summaries: dict[str, Any] = {}
    ordered_summaries: list[dict[str, Any]] = []
    for version in args.versions:
        tasks: dict[str, Any] = {}
        for task in args.tasks:
            group_dir = model_root / version / task
            predictions_path = group_dir / "predictions.jsonl"
            rows = load_jsonl(predictions_path)
            previous_path = group_dir / "summary.json"
            previous = load_json(previous_path) if previous_path.is_file() else {}
            if task == "api":
                rows, summary = score_api_rows(rows, executor, gt_index)
            else:
                rows, summary = score_response_rows(rows)
            diagnostics = diagnostic_summary(previous)
            if diagnostics:
                summary["diagnostics"] = diagnostics
            dump_jsonl(predictions_path, rows)
            dump_json(previous_path, summary)
            tasks[task] = summary
            ordered_summaries.append(summary)
        version_summary = {"version": version, "tasks": tasks}
        version_summaries[version] = version_summary
        dump_json(model_root / version / "summary.json", version_summary)

    summary = {
        key: value
        for key, value in existing_top.items()
        if key not in {"versions", "evaluation_metric", "official_runtime"}
    }
    summary.update(
        {
            "model_name": args.model_name,
            "evaluation_metric": "official_api_bank_execution_accuracy",
            "official_runtime": {
                "upstream_commit": UPSTREAM_COMMIT,
                "runtime_dir": str(runtime_dir),
                "tool_search_model": str(Path(args.tool_search_model).resolve()),
                "network_policy": NETWORK_POLICY,
                "seed": args.seed,
            },
            "versions": version_summaries,
        }
    )
    dump_json(top_summary_path, summary)
    (model_root / "summary.md").write_text(
        render_markdown(args.model_name, ordered_summaries), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def normalized_for_mapping(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: normalized_for_mapping(item) for key, item in value.items()}
    if isinstance(value, list):
        return [normalized_for_mapping(item) for item in value]
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith(("[", "{")) and stripped.endswith(("]", "}")):
            for loader in (json.loads, __import__("ast").literal_eval):
                try:
                    return normalized_for_mapping(loader(stripped))
                except (ValueError, SyntaxError, TypeError, json.JSONDecodeError):
                    pass
    return value


def mapping_equivalent(left: Any, right: Any) -> bool:
    """Compare pre-ToolManager labels after its documented type coercions."""
    left = normalized_for_mapping(left)
    right = normalized_for_mapping(right)
    if isinstance(left, dict) and isinstance(right, dict):
        return set(left) == set(right) and all(
            mapping_equivalent(left[key], right[key]) for key in left
        )
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(
            mapping_equivalent(a, b) for a, b in zip(left, right, strict=True)
        )
    if isinstance(left, str) and isinstance(right, (int, float)) and not isinstance(right, bool):
        try:
            return float(left) == float(right)
        except ValueError:
            return False
    if isinstance(right, str) and isinstance(left, (int, float)) and not isinstance(left, bool):
        try:
            return float(right) == float(left)
        except ValueError:
            return False
    return left == right


def validate_mapping(args: argparse.Namespace) -> None:
    from evaluate_api_bank import parse_official_api_call

    runtime_dir = Path(args.runtime_dir).resolve()
    test_data_dir = Path(args.test_data_dir).resolve()
    gt_index = GroundTruthIndex(runtime_dir, test_data_dir)
    files = {
        "v1": ("level-1-api.json", "expected_output"),
        "v2": ("level-2-api.json", "expected_output"),
        "v3": ("level-3-batch-inf.json", "output"),
    }
    counts: Counter[str] = Counter()
    failures: list[dict[str, Any]] = []
    for version, (filename, reference_field) in files.items():
        for index, entry in enumerate(load_json(test_data_dir / filename)):
            row = {
                "version": version,
                "file": entry.get("file"),
                "source_id": entry.get("id"),
                "sample_id": entry.get("sample_id"),
                "api_id": entry.get("api_id"),
            }
            ground_truth = gt_index.api_ground_truth(row)
            reference = parse_official_api_call(entry[reference_field])
            matches = (
                reference["name"] == ground_truth["api_name"]
                and mapping_equivalent(
                    reference["parameters"], ground_truth["parameters"]
                )
            )
            counts[version] += 1
            if not matches:
                failures.append(
                    {
                        "version": version,
                        "index": index,
                        "reference": reference,
                        "ground_truth": ground_truth,
                    }
                )
    output = {
        "status": "ok" if not failures else "failed",
        "validated": dict(counts),
        "failure_count": len(failures),
        "failures": failures[:20],
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))
    if failures:
        raise AssertionError(f"ground-truth mapping failures: {len(failures)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    score_parser = subparsers.add_parser("score")
    score_parser.add_argument("--model-name", required=True)
    score_parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    score_parser.add_argument("--runtime-dir", type=Path, default=DEFAULT_RUNTIME)
    score_parser.add_argument("--test-data-dir", type=Path, default=DEFAULT_TEST_DATA)
    score_parser.add_argument("--tool-search-model", type=Path, default=DEFAULT_SEARCH_MODEL)
    score_parser.add_argument("--seed", type=int, default=42)
    score_parser.add_argument("--versions", nargs="+", choices=("v1", "v2", "v3"), default=["v1", "v2", "v3"])
    score_parser.add_argument("--tasks", nargs="+", choices=("api", "response"), default=["api", "response"])
    score_parser.set_defaults(func=score_model)

    validate_parser = subparsers.add_parser("validate-mapping")
    validate_parser.add_argument("--runtime-dir", type=Path, default=DEFAULT_RUNTIME)
    validate_parser.add_argument("--test-data-dir", type=Path, default=DEFAULT_TEST_DATA)
    validate_parser.set_defaults(func=validate_mapping)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
