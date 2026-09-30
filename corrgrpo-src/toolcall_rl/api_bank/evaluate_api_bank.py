#!/usr/bin/env python3
"""Evaluate local checkpoints on API-Bank with RLLA or official prompts.

The Hugging Face API-Bank release contains static API-call and response labels.
This evaluator can either adapt those examples to RLLA's training-time
system/history format or preserve API-Bank's official instruction/input prompt.
It generates with vLLM and writes physically separate v1/v2/v3 results.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import queue as queue_module
import re
import signal
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from collections import Counter, defaultdict
from pathlib import Path
from statistics import fmean
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = ROOT / "test-data"
DEFAULT_OUTPUT_DIR = ROOT / "results"
DEFAULT_OFFICIAL_RUNTIME_DIR = ROOT / "official_runtime"
DEFAULT_TOOL_SEARCH_MODEL = (
    DEFAULT_OFFICIAL_RUNTIME_DIR / "models" / "paraphrase-MiniLM-L3-v2"
)
DEFAULT_OFFICIAL_PYTHON = Path(
    os.environ.get(
        "API_BANK_OFFICIAL_PYTHON",
        sys.executable,
    )
)
PROMPT_TEMPLATE_CHOICES = ("rlla", "official")
PROMPT_TEMPLATE_METADATA = {
    "rlla": "exact_rlla_training_template",
    "official": "official_api_bank_instruction_input",
}

RLLA_TOOL_CATALOG_PLACEHOLDER = "__API_BANK_TOOL_CATALOG__"
RLLA_TRAINING_SYSTEM_TEMPLATE = """You are a helpful multi-turn dialogue assistant capable of leveraging tool calls to solve user tasks and provide structured chat responses.

**Available Tools**
In your response, you can use the following tools:
__API_BANK_TOOL_CATALOG__

**Steps for Each Turn**
1. **Think:** Recall relevant context and analyze the current user goal.
2. **Decide on Tool Usage:** If a tool is needed, specify the tool and its parameters.
3. **Respond Appropriately:** If a response is needed, generate one while maintaining consistency across user queries.

**Output Format**
```plaintext
<think> Your thoughts and reasoning </think>
<tool_call>
{"name": "Tool name", "parameters": {"Parameter name": "Parameter content", "... ...": "... ..."}}
{"name": "... ...", "parameters": {"... ...": "... ...", "... ...": "... ..."}}
...
</tool_call>
<response> AI's final response </response>
```

**Important Notes**
1. You must always include the `<think>` field to outline your reasoning. Provide at least one of `<tool_call>` or `<response>`. Decide whether to use `<tool_call>` (possibly multiple times), `<response>`, or both.
2. You can invoke multiple tool calls simultaneously in the `<tool_call>` fields. Each tool call should be a JSON object with a "name" field and an "parameters" field containing a dictionary of parameters. If no parameters are needed, leave the "parameters" field an empty dictionary.
3. Refer to the previous dialogue records in the history, including the user's queries, previous `<tool_call>`, `<response>`, and any tool feedback noted as `<obs>` (if exists)."""

DATASETS = {
    "v1": {
        "api": ("level-1-api.json", "expected_output"),
        "response": ("level-1-response.json", "expected_output"),
    },
    "v2": {
        "api": ("level-2-api.json", "expected_output"),
        "response": ("level-2-response.json", "expected_output"),
    },
    "v3": {
        "api": ("level-3-batch-inf.json", "output"),
        "response": ("level-3-batch-inf-response.json", "output"),
    },
}


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def dump_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def dump_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")


def validate_parallel_config(
    data_parallel_size: int,
    tensor_parallel_size: int,
    visible_devices: str | None = None,
) -> int:
    """Validate DP/TP sizes and return the number of required GPUs."""
    if data_parallel_size < 1:
        raise ValueError("data_parallel_size must be at least 1")
    if tensor_parallel_size < 1:
        raise ValueError("tensor_parallel_size must be at least 1")
    required_gpus = data_parallel_size * tensor_parallel_size
    if visible_devices is None:
        visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible_devices is not None:
        devices = [
            device.strip()
            for device in visible_devices.split(",")
            if device.strip() and device.strip() != "-1"
        ]
        if required_gpus > len(devices):
            raise ValueError(
                f"DP={data_parallel_size} x TP={tensor_parallel_size} requires "
                f"{required_gpus} GPUs, but CUDA_VISIBLE_DEVICES exposes "
                f"{len(devices)} ({visible_devices!r})"
            )
    return required_gpus


def partition_indexed_items(
    items: list[str], partition_count: int
) -> list[list[tuple[int, str]]]:
    """Round-robin items across workers while retaining global indices."""
    if partition_count < 1:
        raise ValueError("partition_count must be at least 1")
    partitions: list[list[tuple[int, str]]] = [
        [] for _ in range(partition_count)
    ]
    for index, item in enumerate(items):
        partitions[index % partition_count].append((index, item))
    return partitions


def _generation_result(output: Any) -> dict[str, Any]:
    generated = output.outputs[0]
    return {
        "text": generated.text,
        "token_count": len(generated.token_ids),
        "finish_reason": generated.finish_reason,
        "stop_reason": generated.stop_reason,
    }


def _data_parallel_worker(
    rank: int,
    indexed_prompts: list[tuple[int, str]],
    device_ids: list[str],
    engine_kwargs: dict[str, Any],
    sampling_kwargs: dict[str, Any],
    result_queue: Any,
) -> None:
    """Run one independent vLLM replica on its assigned CUDA devices."""
    try:
        # Put the worker and every vLLM subprocess it creates in an isolated
        # process group.  Killing only the multiprocessing worker is not
        # sufficient: EngineCore children otherwise survive as PPID 1 and
        # keep almost all GPU memory allocated after an initialization error.
        if os.name == "posix":
            os.setsid()
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(device_ids)
        # A parent environment configured for vLLM's native DP must not leak
        # ranks into these independent offline workers.
        for name in (
            "VLLM_DP_SIZE",
            "VLLM_DP_RANK",
            "VLLM_DP_RANK_LOCAL",
            "VLLM_DP_MASTER_IP",
            "VLLM_DP_MASTER_PORT",
        ):
            os.environ.pop(name, None)
        from vllm import LLM, SamplingParams

        llm = LLM(**engine_kwargs)
        outputs = llm.generate(
            [prompt for _, prompt in indexed_prompts],
            SamplingParams(**sampling_kwargs),
        )
        result_queue.put(
            {
                "status": "ok",
                "rank": rank,
                "outputs": [
                    (index, _generation_result(output))
                    for (index, _), output in zip(
                        indexed_prompts, outputs, strict=True
                    )
                ],
            }
        )
    except BaseException:
        result_queue.put(
            {
                "status": "error",
                "rank": rank,
                "traceback": traceback.format_exc(),
            }
        )


def terminate_data_parallel_processes(
    processes: list[Any], timeout: float = 10.0
) -> None:
    """Terminate DP workers together with all descendant vLLM processes."""
    process_group_ids = [process.pid for process in processes if process.pid is not None]
    for process in processes:
        process_group_id = process.pid
        if process_group_id is None:
            continue
        group_signalled = False
        if os.name == "posix":
            try:
                os.killpg(process_group_id, signal.SIGTERM)
                group_signalled = True
            except ProcessLookupError:
                pass
        if not group_signalled and process.is_alive():
            process.terminate()

    deadline = time.monotonic() + timeout
    for process in processes:
        process.join(timeout=max(0.0, deadline - time.monotonic()))

    # The worker may have exited after SIGTERM while an EngineCore grandchild
    # ignored it.  Address the process group by its original leader PID even
    # when that leader no longer exists, then reap the direct child.
    if os.name == "posix":
        for process_group_id in process_group_ids:
            try:
                os.killpg(process_group_id, signal.SIGKILL)
            except ProcessLookupError:
                pass
    for process in processes:
        if process.is_alive():
            process.kill()
        process.join()


def generate_data_parallel(
    prompts: list[str],
    data_parallel_size: int,
    tensor_parallel_size: int,
    engine_kwargs: dict[str, Any],
    sampling_kwargs: dict[str, Any],
) -> list[dict[str, Any]]:
    """Generate with independent vLLM replicas and restore input order."""
    if not prompts:
        return []
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible_devices is None:
        raise ValueError(
            "CUDA_VISIBLE_DEVICES must be set when data_parallel_size > 1"
        )
    devices = [device.strip() for device in visible_devices.split(",") if device.strip()]
    worker_count = min(data_parallel_size, len(prompts))
    shards = partition_indexed_items(prompts, worker_count)
    context = __import__("multiprocessing").get_context("spawn")
    result_queue = context.Queue()
    processes = []
    for rank, shard in enumerate(shards):
        start = rank * tensor_parallel_size
        worker_devices = devices[start : start + tensor_parallel_size]
        process = context.Process(
            target=_data_parallel_worker,
            args=(
                rank,
                shard,
                worker_devices,
                engine_kwargs,
                sampling_kwargs,
                result_queue,
            ),
            name=f"api-bank-dp-{rank}",
        )
        process.start()
        processes.append(process)

    indexed_outputs: dict[int, dict[str, Any]] = {}
    received_ranks: set[int] = set()
    error_message: str | None = None
    try:
        while len(received_ranks) < worker_count:
            try:
                message = result_queue.get(timeout=1.0)
            except queue_module.Empty:
                if all(not process.is_alive() for process in processes):
                    break
                continue
            rank = int(message["rank"])
            received_ranks.add(rank)
            if message["status"] == "error":
                error_message = (
                    f"data-parallel worker {rank} failed:\n"
                    f"{message['traceback']}"
                )
                break
            indexed_outputs.update(dict(message["outputs"]))
    finally:
        if error_message is not None or len(received_ranks) < worker_count:
            terminate_data_parallel_processes(processes)
        else:
            for process in processes:
                process.join()
        result_queue.close()
        result_queue.join_thread()

    if error_message is not None:
        raise RuntimeError(error_message)
    if len(received_ranks) < worker_count:
        exit_codes = {process.name: process.exitcode for process in processes}
        raise RuntimeError(
            "data-parallel workers exited without returning all results: "
            f"received={sorted(received_ranks)}, exit_codes={exit_codes}"
        )
    if len(indexed_outputs) != len(prompts):
        raise RuntimeError(
            f"expected {len(prompts)} outputs, received {len(indexed_outputs)}"
        )
    return [indexed_outputs[index] for index in range(len(prompts))]


def _as_tool_definition(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    name = value.get("name", value.get("apiCode"))
    if not isinstance(name, str) or not name:
        return None
    parameters = value.get("input_parameters", value.get("parameters", {}))
    if not isinstance(parameters, dict):
        parameters = {}
    return {
        "name": name,
        "description": str(value.get("description", "")),
        "parameters": parameters,
    }


def _json_lines(text: str) -> Iterable[dict[str, Any]]:
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped.startswith("{"):
            continue
        try:
            value = json.loads(stripped)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            yield value


def extract_tool_definitions(entry: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract the API catalogue exposed by one static API-Bank example."""
    tools: list[dict[str, Any]] = []
    seen: set[str] = set()
    for field in ("instruction", "input"):
        for value in _json_lines(str(entry.get(field, ""))):
            tool = _as_tool_definition(value)
            if tool is not None and tool["name"] not in seen:
                tools.append(tool)
                seen.add(tool["name"])
    if not tools:
        raise ValueError("example contains no parseable API descriptions")
    return tools


def render_tool_catalog(tools: list[dict[str, Any]]) -> str:
    return "\n".join(
        f"{index}. Name: {tool['name']}\n"
        f"Description: {tool['description']}\n"
        f"Parameters: {json.dumps(tool['parameters'], ensure_ascii=False)}"
        for index, tool in enumerate(tools, start=1)
    )


def build_system_prompt(entry: dict[str, Any]) -> str:
    return RLLA_TRAINING_SYSTEM_TEMPLATE.replace(
        RLLA_TOOL_CATALOG_PLACEHOLDER,
        render_tool_catalog(extract_tool_definitions(entry)),
    )


def _literal_or_text(value: str) -> Any:
    stripped = value.strip()
    for loader in (ast.literal_eval, json.loads):
        try:
            return loader(stripped)
        except (ValueError, SyntaxError, TypeError, json.JSONDecodeError):
            pass
    return stripped


def _normalize_parameter_value(value: Any) -> Any:
    """Recover list/dict values double-serialized by the API-Bank release."""
    if value is Ellipsis:
        # Models sometimes copy the prompt placeholder ``key=...``. Preserve
        # it as an invalid string value rather than leaking Python's Ellipsis
        # singleton into JSON result files.
        return "..."
    if isinstance(value, dict):
        return {
            str(key): _normalize_parameter_value(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_normalize_parameter_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_normalize_parameter_value(item) for item in value)
    if isinstance(value, set):
        return sorted(
            (_normalize_parameter_value(item) for item in value), key=repr
        )
    if not isinstance(value, str):
        return value
    stripped = value.strip()
    if not (
        (stripped.startswith("[") and stripped.endswith("]"))
        or (stripped.startswith("{") and stripped.endswith("}"))
    ):
        return value
    for loader in (ast.literal_eval, json.loads):
        try:
            return _normalize_parameter_value(loader(stripped))
        except (ValueError, SyntaxError, TypeError, json.JSONDecodeError):
            pass
    return value


def _parse_api_parameters_tolerantly(parameters_text: str) -> dict[str, Any]:
    """Parse upstream labels such as ``attendees='['A', 'B']'``.

    API-Bank contains several labels whose outer single-quoted value embeds an
    unescaped Python list/dict using the same quote character. Splitting only
    at commas followed by another ``name=`` assignment recovers those values
    without executing arbitrary text.
    """
    parameters_text = parameters_text.strip()
    if not parameters_text:
        return {}
    assignments = list(
        re.finditer(r"(?:^|,\s*)([A-Za-z_]\w*)\s*=\s*", parameters_text)
    )
    if not assignments or assignments[0].start() != 0:
        raise ValueError(f"cannot split API parameters: {parameters_text!r}")
    parameters: dict[str, Any] = {}
    for index, assignment in enumerate(assignments):
        end = assignments[index + 1].start() if index + 1 < len(assignments) else len(parameters_text)
        raw_value = parameters_text[assignment.end() : end].strip()
        if len(raw_value) >= 2 and raw_value[0] == raw_value[-1] and raw_value[0] in "'\"":
            value: Any = raw_value[1:-1]
            value = _normalize_parameter_value(value)
        else:
            for loader in (ast.literal_eval, json.loads):
                try:
                    value = loader(raw_value)
                    break
                except (ValueError, SyntaxError, TypeError, json.JSONDecodeError):
                    value = raw_value
        parameters[assignment.group(1)] = _normalize_parameter_value(value)
    return parameters


def parse_official_api_call(text: str) -> dict[str, Any]:
    """Parse API-Bank's ``[ApiName(key='value')]`` representation safely."""
    marker = text.find("API-Request:")
    candidate = text[marker + len("API-Request:") :] if marker >= 0 else text
    start = candidate.find("[")
    end = candidate.rfind("]")
    if start < 0 or end <= start:
        raise ValueError(f"no bracketed API call found: {text!r}")
    call_text = candidate[start + 1 : end].strip()
    try:
        expression = ast.parse(call_text, mode="eval").body
        if not isinstance(expression, ast.Call) or not isinstance(expression.func, ast.Name):
            raise ValueError(f"unsupported API call expression: {text!r}")
        if expression.args:
            raise ValueError("API-Bank calls with positional arguments are unsupported")
        parameters: dict[str, Any] = {}
        for keyword in expression.keywords:
            if keyword.arg is None:
                raise ValueError("API-Bank **kwargs calls are unsupported")
            parameters[keyword.arg] = _normalize_parameter_value(
                ast.literal_eval(keyword.value)
            )
        return {"name": expression.func.id, "parameters": parameters}
    except SyntaxError:
        match = re.fullmatch(
            r"([A-Za-z_]\w*)\s*\((.*)\)", call_text, flags=re.DOTALL
        )
        if match is None:
            raise ValueError(f"unsupported API call expression: {text!r}")
        return {
            "name": match.group(1),
            "parameters": _parse_api_parameters_tolerantly(match.group(2)),
        }


def _history_part(role: str, content: str) -> str:
    content = content.strip()
    if role == "user":
        return f"<user> {content} </user>"
    if role == "assistant":
        return f"<response> {content} </response>"
    raise ValueError(role)


def build_dialogue_history(entry: dict[str, Any]) -> str:
    """Convert API-Bank's transcript to RLLA's training-time history tags."""
    parts: list[str] = []
    current_role: str | None = None
    current_lines: list[str] = []

    def flush() -> None:
        nonlocal current_role, current_lines
        if current_role is not None and any(line.strip() for line in current_lines):
            parts.append(_history_part(current_role, "\n".join(current_lines)))
        current_role = None
        current_lines = []

    for raw_line in str(entry.get("input", "")).splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("{"):
            try:
                if _as_tool_definition(json.loads(line)) is not None:
                    continue
            except json.JSONDecodeError:
                pass
        if re.match(r"^Generate (API Request|AI Response):?", line, re.IGNORECASE):
            flush()
            continue
        role_match = re.match(r"^(User|AI):\s*(.*)$", line, re.DOTALL)
        if role_match:
            flush()
            current_role = "user" if role_match.group(1) == "User" else "assistant"
            current_lines = [role_match.group(2)]
            continue
        if line.startswith("API-Request:"):
            flush()
            call_text, separator, result_text = line.partition("->")
            call = parse_official_api_call(call_text)
            parts.append(
                "<tool_call>\n"
                + json.dumps(call, ensure_ascii=False)
                + "\n</tool_call>"
            )
            if separator:
                observation = [{"name": call["name"], "results": _literal_or_text(result_text)}]
                parts.append(
                    f"<obs> {json.dumps(observation, ensure_ascii=False)} </obs>"
                )
            continue
        if current_role is not None:
            current_lines.append(raw_line)

    flush()
    if not parts:
        raw_input = str(entry.get("input", "")).strip()
        if not raw_input:
            raise ValueError("example has empty dialogue input")
        parts.append(_history_part("user", raw_input))
    return "**Dialogue Records History**\n" + "\n\n".join(parts)


def build_rlla_messages(entry: dict[str, Any]) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": build_system_prompt(entry)},
        {"role": "user", "content": build_dialogue_history(entry)},
    ]


def build_official_prompt(entry: dict[str, Any]) -> str:
    """Reproduce the prompt fields distributed by API-Bank."""
    instruction = str(entry.get("instruction", ""))
    model_input = str(entry.get("input", ""))
    if not instruction.strip() or not model_input.strip():
        raise ValueError("official API-Bank prompt requires instruction and input")
    return instruction.rstrip() + "\n" + model_input.lstrip("\n")


def build_prompt_messages(
    entry: dict[str, Any], prompt_template: str
) -> list[dict[str, str]]:
    if prompt_template == "rlla":
        return build_rlla_messages(entry)
    if prompt_template == "official":
        return [{"role": "user", "content": build_official_prompt(entry)}]
    raise ValueError(f"unsupported prompt template: {prompt_template}")


def render_model_prompt(
    messages: list[dict[str, str]], prompt_template: str, tokenizer: Any
) -> str:
    """Render the model input without altering API-Bank's official prompt."""
    if prompt_template == "official":
        if len(messages) != 1 or messages[0].get("role") != "user":
            raise ValueError("official prompt must contain exactly one user message")
        return messages[0]["content"]
    if prompt_template == "rlla":
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    raise ValueError(f"unsupported prompt template: {prompt_template}")


def _decode_json_objects(block: str) -> tuple[list[dict[str, Any]], bool]:
    decoder = json.JSONDecoder()
    calls: list[dict[str, Any]] = []
    position = 0
    valid = True
    while position < len(block):
        while position < len(block) and block[position].isspace():
            position += 1
        if position >= len(block):
            break
        try:
            value, position = decoder.raw_decode(block, position)
        except json.JSONDecodeError:
            valid = False
            break
        if not isinstance(value, dict):
            valid = False
            continue
        name = value.get("name")
        parameters = value.get("parameters", value.get("arguments"))
        if not isinstance(name, str) or not isinstance(parameters, dict):
            valid = False
            continue
        calls.append({"name": name, "parameters": parameters})
    return calls, valid


def parse_rlla_tool_calls(text: str) -> tuple[list[dict[str, Any]], bool]:
    blocks = re.findall(r"<tool_call>\s*(.*?)\s*</tool_call>", text, re.DOTALL)
    calls: list[dict[str, Any]] = []
    valid = bool(blocks)
    for block in blocks:
        decoded, block_valid = _decode_json_objects(block)
        calls.extend(decoded)
        valid = valid and block_valid
    if calls:
        return calls, valid
    try:
        return [parse_official_api_call(text)], False
    except (ValueError, SyntaxError):
        return [], False


def parse_prediction_tool_calls(
    text: str, prompt_template: str
) -> tuple[list[dict[str, Any]], bool]:
    """Parse calls in the output syntax requested by the selected prompt."""
    if prompt_template == "rlla":
        return parse_rlla_tool_calls(text)
    if prompt_template == "official":
        try:
            return [parse_official_api_call(text)], True
        except (ValueError, SyntaxError):
            return [], False
    raise ValueError(f"unsupported prompt template: {prompt_template}")


def extract_rlla_response(
    text: str, prompt_prefills_think: bool = False
) -> tuple[str, bool]:
    matches = re.findall(r"<response>\s*(.*?)\s*</response>", text, re.DOTALL)
    if matches:
        return "\n".join(match.strip() for match in matches), True
    fallback = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    if (
        prompt_prefills_think
        and "<think>" not in text
        and fallback.count("</think>") == 1
    ):
        # Some Qwen thinking chat templates put ``<think>`` in the rendered
        # assistant prefix.  The generated continuation therefore contains
        # only the matching closing tag; remove the prefixed reasoning before
        # applying the response fallback.
        fallback = fallback.split("</think>", 1)[1]
    fallback = re.sub(r"<tool_call>.*?</tool_call>", "", fallback, flags=re.DOTALL)
    return fallback.strip(), False


def extract_official_response(text: str) -> tuple[str, bool]:
    """Extract API-Bank's plain ``AI: ...`` response continuation."""
    response = text.strip()
    if "<response>" in response:
        response, _ = extract_rlla_response(response)
    else:
        response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL)
        # Qwen thinking templates may put the opening token in the rendered
        # prompt, leaving only the closing token in the generated continuation.
        if "</think>" in response:
            response = response.split("</think>", 1)[1]
    response = re.sub(r"^\s*AI:\s*", "", response, flags=re.IGNORECASE)
    return response.strip(), bool(response.strip())


def extract_prediction_response(
    text: str,
    prompt_template: str,
    prompt_prefills_think: bool = False,
) -> tuple[str, bool]:
    if prompt_template == "rlla":
        return extract_rlla_response(text, prompt_prefills_think)
    if prompt_template == "official":
        return extract_official_response(text)
    raise ValueError(f"unsupported prompt template: {prompt_template}")


# Backward-compatible import used by older callers.
extract_response = extract_rlla_response


def prompt_prefills_rlla_think(prompt: str) -> bool:
    """Return whether the rendered assistant prefix already opens ``<think>``."""
    return re.search(r"<think>\s*$", prompt) is not None


def rlla_format_valid(text: str, prompt_prefills_think: bool = False) -> bool:
    has_complete_think = (
        re.search(r"<think>.*?</think>", text, re.DOTALL) is not None
    )
    has_action = (
        re.search(r"<tool_call>.*?</tool_call>", text, re.DOTALL) is not None
        or re.search(r"<response>.*?</response>", text, re.DOTALL) is not None
    )
    action_starts = [
        match.start()
        for match in re.finditer(r"<(?:tool_call|response)>", text)
    ]
    has_prefilled_think = (
        prompt_prefills_think
        and "<think>" not in text
        and text.count("</think>") == 1
        and bool(action_starts)
        and text.index("</think>") < min(action_starts)
    )
    return (has_complete_think or has_prefilled_think) and has_action


def _rouge_tokens(text: str) -> list[str]:
    return re.findall(r"\w+|[^\w\s]", text.lower(), flags=re.UNICODE)


def rouge_l_f1(reference: str, prediction: str) -> float:
    """Dependency-free Rouge-L F1 with the standard LCS definition."""
    reference_tokens = _rouge_tokens(reference)
    prediction_tokens = _rouge_tokens(prediction)
    if not reference_tokens or not prediction_tokens:
        return 0.0
    previous = [0] * (len(prediction_tokens) + 1)
    for reference_token in reference_tokens:
        current = [0]
        for index, prediction_token in enumerate(prediction_tokens, start=1):
            if reference_token == prediction_token:
                current.append(previous[index - 1] + 1)
            else:
                current.append(max(previous[index], current[-1]))
        previous = current
    lcs = previous[-1]
    precision = lcs / len(prediction_tokens)
    recall = lcs / len(reference_tokens)
    return 2 * precision * recall / (precision + recall) if lcs else 0.0


def score_api_prediction(
    prediction: str,
    reference: str,
    prompt_template: str = "rlla",
    prompt_prefills_think: bool = False,
) -> dict[str, Any]:
    expected = parse_official_api_call(reference)
    calls, api_call_valid = parse_prediction_tool_calls(prediction, prompt_template)
    tool_block_valid = api_call_valid if prompt_template == "rlla" else False
    rlla_valid = rlla_format_valid(prediction, prompt_prefills_think)
    format_valid = (
        rlla_valid
        if prompt_template == "rlla"
        else api_call_valid
    )
    first = calls[0] if calls else {"name": None, "parameters": {}}
    expected_parameters = expected["parameters"]
    predicted_parameters = first["parameters"]
    function_correct = first["name"] == expected["name"]
    parameter_names_correct = set(predicted_parameters) == set(expected_parameters)
    parameter_values_correct = parameter_names_correct and all(
        predicted_parameters[key] == value for key, value in expected_parameters.items()
    )
    exact = (
        len(calls) == 1
        and function_correct
        and parameter_names_correct
        and parameter_values_correct
    )
    return {
        "reference_call": expected,
        "predicted_calls": calls,
        "format_valid": format_valid,
        "rlla_format_valid": rlla_valid,
        "official_format_valid": api_call_valid if prompt_template == "official" else False,
        "api_call_valid": api_call_valid,
        "tool_block_valid": tool_block_valid,
        "function_name_correct": function_correct,
        "parameter_names_correct": parameter_names_correct,
        "parameter_values_correct": parameter_values_correct,
        "call_exact_match": exact,
    }


def score_response_prediction(
    prediction: str,
    reference: str,
    prompt_template: str = "rlla",
    prompt_prefills_think: bool = False,
) -> dict[str, Any]:
    response, response_valid = extract_prediction_response(
        prediction, prompt_template, prompt_prefills_think
    )
    response_block_valid = response_valid if prompt_template == "rlla" else False
    rlla_valid = rlla_format_valid(prediction, prompt_prefills_think)
    format_valid = (
        rlla_valid
        if prompt_template == "rlla"
        else response_valid
    )
    normalized_prediction = " ".join(response.split())
    normalized_reference = " ".join(reference.split())
    return {
        "extracted_response": response,
        "format_valid": format_valid,
        "rlla_format_valid": rlla_valid,
        "official_format_valid": response_valid if prompt_template == "official" else False,
        "response_valid": response_valid,
        "response_block_valid": response_block_valid,
        "exact_match": normalized_prediction == normalized_reference,
        "rouge_l_f1": rouge_l_f1(reference, response),
    }


def score_record(record: dict[str, Any]) -> dict[str, Any]:
    prompt_template = str(record.get("prompt_template", "rlla"))
    prompt_prefills_think = bool(record.get("prompt_prefills_think", False))
    scoring = (
        score_api_prediction(
            record["prediction"],
            record["reference"],
            prompt_template,
            prompt_prefills_think,
        )
        if record["task"] == "api"
        else score_response_prediction(
            record["prediction"],
            record["reference"],
            prompt_template,
            prompt_prefills_think,
        )
    )
    return {**record, "scores": scoring}


def summarize_group(
    version: str, task: str, records: list[dict[str, Any]]
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "version": version,
        "task": task,
        "sample_count": len(records),
    }
    if not records:
        return summary
    scores = [record["scores"] for record in records]
    prompt_template = str(records[0].get("prompt_template", "rlla"))
    summary["prompt_template"] = prompt_template
    summary["output_format_rate"] = fmean(score["format_valid"] for score in scores)
    if prompt_template == "rlla":
        summary["rlla_format_rate"] = summary["output_format_rate"]
    else:
        summary["official_format_rate"] = summary["output_format_rate"]
    if task == "api":
        summary.update(
            {
                "api_call_valid_rate": fmean(
                    score["api_call_valid"] for score in scores
                ),
                "function_name_accuracy": fmean(
                    score["function_name_correct"] for score in scores
                ),
                "parameter_name_accuracy": fmean(
                    score["parameter_names_correct"] for score in scores
                ),
                "parameter_value_accuracy": fmean(
                    score["parameter_values_correct"] for score in scores
                ),
                "call_exact_match_accuracy": fmean(
                    score["call_exact_match"] for score in scores
                ),
            }
        )
        if prompt_template == "rlla":
            summary["tool_block_valid_rate"] = summary["api_call_valid_rate"]
        if version == "v3":
            sample_calls: dict[str, list[bool]] = defaultdict(list)
            for record in records:
                sample_id = str(record.get("sample_id", record["index"]))
                sample_calls[sample_id].append(record["scores"]["call_exact_match"])
            successful = sum(all(call_results) for call_results in sample_calls.values())
            summary.update(
                {
                    "dialogue_sample_count": len(sample_calls),
                    "dialogue_success_count": successful,
                    "dialogue_success_accuracy": successful / len(sample_calls),
                }
            )
    else:
        summary.update(
            {
                "response_valid_rate": fmean(
                    score["response_valid"] for score in scores
                ),
                "response_exact_match_accuracy": fmean(
                    score["exact_match"] for score in scores
                ),
                "rouge_l_f1": fmean(score["rouge_l_f1"] for score in scores),
            }
        )
        if prompt_template == "rlla":
            summary["response_block_valid_rate"] = summary["response_valid_rate"]
    return summary


def selected_groups(args: argparse.Namespace) -> list[tuple[str, str, Path, str]]:
    groups: list[tuple[str, str, Path, str]] = []
    for version in args.versions:
        for task in args.tasks:
            filename, reference_field = DATASETS[version][task]
            if version == "v3" and task == "api" and args.v3_icl:
                filename = "level-3-batch-inf-icl.json"
            groups.append((version, task, Path(args.data_dir) / filename, reference_field))
    return groups


def load_group_records(
    version: str,
    task: str,
    path: Path,
    reference_field: str,
    max_samples: int,
    prompt_template: str,
) -> list[dict[str, Any]]:
    entries = load_json(path)
    if not isinstance(entries, list):
        raise TypeError(f"expected a JSON list: {path}")
    if max_samples > 0:
        entries = entries[:max_samples]
    records = []
    for index, entry in enumerate(entries):
        records.append(
            {
                "version": version,
                "task": task,
                "index": index,
                "source_file": path.name,
                "file": entry.get("file"),
                "source_id": entry.get("id"),
                "sample_id": entry.get("sample_id"),
                "api_id": entry.get("api_id"),
                "reference": entry[reference_field],
                "prompt_template": prompt_template,
                "messages": build_prompt_messages(entry, prompt_template),
            }
        )
    return records


def model_name_from_path(model_path: str) -> str:
    path = Path(model_path.rstrip("/"))
    if path.name == "hf":
        return path.parent.name
    cache_match = re.search(r"/models--[^/]+--([^/]+)/snapshots/[^/]+$", model_path)
    return cache_match.group(1) if cache_match else path.name


def prepare_tokenizer_source(
    model_path: str,
) -> tuple[str, tempfile.TemporaryDirectory[str] | None]:
    """Return a tokenizer path compatible with Transformers 4.51.

    Some VERL exports put a list in ``extra_special_tokens``. Transformers
    4.51 interprets that field as a mapping and crashes. Build a small temporary
    tokenizer-only copy with the list moved to ``additional_special_tokens``;
    never mutate the checkpoint itself.
    """
    source_root = Path(model_path)
    config_path = source_root / "tokenizer_config.json"
    if not config_path.is_file():
        return model_path, None
    config = load_json(config_path)
    extra_tokens = config.get("extra_special_tokens")
    if not isinstance(extra_tokens, list):
        return model_path, None

    temporary = tempfile.TemporaryDirectory(prefix="api-bank-tokenizer-")
    temporary_root = Path(temporary.name)
    exact_names = {
        "config.json",
        "generation_config.json",
        "special_tokens_map.json",
        "added_tokens.json",
        "vocab.json",
        "merges.txt",
        "chat_template.jinja",
    }
    for source in source_root.iterdir():
        if not source.is_file():
            continue
        if (
            source.name in exact_names
            or "tokenizer" in source.name
            or source.suffix in {".model", ".tiktoken", ".py"}
        ):
            shutil.copy2(source, temporary_root / source.name)

    config.pop("extra_special_tokens")
    existing = config.get("additional_special_tokens", [])
    if not isinstance(existing, list):
        existing = []
    config["additional_special_tokens"] = list(dict.fromkeys([*existing, *extra_tokens]))
    dump_json(temporary_root / "tokenizer_config.json", config)
    return str(temporary_root), temporary


def write_scored_results(
    output_root: Path,
    model_name: str,
    grouped_records: dict[tuple[str, str], list[dict[str, Any]]],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    model_root = output_root / model_name
    version_summaries: dict[str, dict[str, Any]] = {}
    all_group_summaries = []
    for (version, task), raw_records in grouped_records.items():
        scored = [score_record(record) for record in raw_records]
        group_dir = model_root / version / task
        dump_jsonl(group_dir / "predictions.jsonl", scored)
        group_summary = summarize_group(version, task, scored)
        dump_json(group_dir / "summary.json", group_summary)
        all_group_summaries.append(group_summary)
        version_summaries.setdefault(version, {"version": version, "tasks": {}})[
            "tasks"
        ][task] = group_summary

    for version, summary in version_summaries.items():
        dump_json(model_root / version / "summary.json", summary)

    summary = {
        "model_name": model_name,
        **metadata,
        "versions": version_summaries,
    }
    dump_json(model_root / "summary.json", summary)
    (model_root / "summary.md").write_text(
        render_summary_markdown(summary, all_group_summaries), encoding="utf-8"
    )
    return summary


def render_summary_markdown(
    summary: dict[str, Any], group_summaries: list[dict[str, Any]]
) -> str:
    lines = [
        f"# API-Bank evaluation: {summary['model_name']}",
        "",
        "API-call scores use structured label exact match. Response scores use Rouge-L F1.",
        "All values below are percentages.",
        "",
        "| Version | Task | Samples | Output format | Primary score | Extra |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for group in group_summaries:
        if group["task"] == "api":
            primary = group.get("call_exact_match_accuracy", 0.0)
            extra_label = (
                group.get("dialogue_success_accuracy", group.get("function_name_accuracy", 0.0))
            )
        else:
            primary = group.get("rouge_l_f1", 0.0)
            extra_label = group.get("response_exact_match_accuracy", 0.0)
        lines.append(
            f"| {group['version']} | {group['task']} | {group['sample_count']} | "
            f"{100 * group.get('output_format_rate', group.get('rlla_format_rate', 0.0)):.2f} | "
            f"{100 * primary:.2f} | {100 * extra_label:.2f} |"
        )
    lines.extend(
        [
            "",
            "For v3 API calls, `Extra` is full-dialogue success (every call in a sample must match). ",
            "For v1/v2 API calls it is function-name accuracy; for responses it is exact match.",
            "",
        ]
    )
    return "\n".join(lines)


def run_official_scorer(
    args: argparse.Namespace, model_name: str
) -> dict[str, Any]:
    if args.skip_official_score:
        return load_json(Path(args.output_dir) / model_name / "summary.json")
    official_python = Path(args.official_python)
    if not official_python.is_file():
        raise FileNotFoundError(f"official scoring Python not found: {official_python}")
    command = [
        str(official_python),
        str(ROOT / "official_accuracy.py"),
        "score",
        "--model-name",
        model_name,
        "--output-dir",
        str(args.output_dir),
        "--runtime-dir",
        str(args.official_runtime_dir),
        "--test-data-dir",
        str(args.data_dir),
        "--tool-search-model",
        str(args.tool_search_model),
        "--seed",
        str(args.official_score_seed),
        "--versions",
        *args.versions,
        "--tasks",
        *args.tasks,
    ]
    environment = os.environ.copy()
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": "",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
    completed = None
    for attempt in range(2):
        completed = subprocess.run(
            command,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=environment,
        )
        if completed.returncode == 0:
            break
        combined_output = completed.stdout + completed.stderr
        transient_import_error = "cannot import name 'is_offline_mode'" in combined_output
        if attempt == 0 and transient_import_error:
            print(
                "Official scorer hit a transient transformers/huggingface_hub "
                "import error; retrying once.",
                file=sys.stderr,
            )
            continue
        break
    assert completed is not None
    if completed.returncode != 0:
        if completed.stdout:
            print(completed.stdout, file=sys.stderr)
        if completed.stderr:
            print(completed.stderr, file=sys.stderr)
        raise RuntimeError(
            f"official API-Bank scorer failed with exit code {completed.returncode}"
        )
    return load_json(Path(args.output_dir) / model_name / "summary.json")


def run_generation(args: argparse.Namespace) -> None:
    required_gpus = validate_parallel_config(
        args.data_parallel_size, args.tensor_parallel_size
    )
    import multiprocessing

    multiprocessing.set_start_method("spawn", force=True)
    from transformers import AutoTokenizer

    if args.prompt_template == "official" and args.force_think_prefix:
        raise ValueError(
            "--force-think-prefix is part of the RLLA format and cannot be "
            "used with --prompt-template official"
        )

    model_name = args.model_name or model_name_from_path(args.model_path)
    if args.prompt_template == "official" and args.model_name is None:
        model_name = f"{model_name}_official_prompt"
    tokenizer_source, tokenizer_temporary = prepare_tokenizer_source(args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_source, local_files_only=True, trust_remote_code=True
    )
    grouped_records: dict[tuple[str, str], list[dict[str, Any]]] = {}
    flat_records: list[dict[str, Any]] = []
    prompts: list[str] = []
    for version, task, path, reference_field in selected_groups(args):
        records = load_group_records(
            version,
            task,
            path,
            reference_field,
            args.max_samples,
            args.prompt_template,
        )
        grouped_records[(version, task)] = records
        for record in records:
            prompt = render_model_prompt(
                record.pop("messages"), args.prompt_template, tokenizer
            )
            if args.force_think_prefix:
                prompt += "<think> "
            record["prompt_prefills_think"] = (
                args.prompt_template == "rlla"
                and prompt_prefills_rlla_think(prompt)
            )
            record["prompt_token_count"] = len(tokenizer.encode(prompt))
            if args.save_prompts:
                record["prompt"] = prompt
            flat_records.append(record)
            prompts.append(prompt)

    engine_kwargs = {
        "model": args.model_path,
        "tokenizer": tokenizer_source,
        "tensor_parallel_size": args.tensor_parallel_size,
        "dtype": args.dtype,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "max_model_len": args.max_model_len,
        "enforce_eager": args.enforce_eager,
        "trust_remote_code": True,
        "disable_log_stats": True,
    }
    sampling_kwargs = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
        "seed": args.seed,
    }
    if args.data_parallel_size == 1:
        from vllm import LLM, SamplingParams

        llm = LLM(**engine_kwargs)
        raw_outputs = llm.generate(prompts, SamplingParams(**sampling_kwargs))
        generation_results = [_generation_result(output) for output in raw_outputs]
    else:
        generation_results = generate_data_parallel(
            prompts,
            args.data_parallel_size,
            args.tensor_parallel_size,
            engine_kwargs,
            sampling_kwargs,
        )
    for record, generated in zip(flat_records, generation_results, strict=True):
        continuation = generated["text"]
        record["prediction"] = (
            "<think> " + continuation if args.force_think_prefix else continuation
        )
        record["response_token_count"] = generated["token_count"]
        record["finish_reason"] = generated["finish_reason"]
        record["stop_reason"] = generated["stop_reason"]
    if tokenizer_temporary is not None:
        tokenizer_temporary.cleanup()

    write_scored_results(
        Path(args.output_dir),
        model_name,
        grouped_records,
        {
            "model_path": str(Path(args.model_path).resolve()),
            "prompt_template": PROMPT_TEMPLATE_METADATA[args.prompt_template],
            "prompt_template_mode": args.prompt_template,
            "v3_icl": args.v3_icl,
            "generation": {
                "temperature": args.temperature,
                "top_p": args.top_p,
                "max_tokens": args.max_tokens,
                "max_model_len": args.max_model_len,
                "data_parallel_size": args.data_parallel_size,
                "active_data_parallel_size": min(
                    args.data_parallel_size, len(prompts)
                ),
                "tensor_parallel_size": args.tensor_parallel_size,
                "required_gpu_count": required_gpus,
                "force_think_prefix": args.force_think_prefix,
                "seed": args.seed,
            },
        },
    )
    summary = run_official_scorer(args, model_name)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def run_score_only(args: argparse.Namespace) -> None:
    model_root = Path(args.output_dir) / args.model_name
    grouped_records: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for version in args.versions:
        for task in args.tasks:
            path = model_root / version / task / "predictions.jsonl"
            if not path.is_file():
                raise FileNotFoundError(path)
            records = []
            with path.open(encoding="utf-8") as stream:
                for line in stream:
                    record = json.loads(line)
                    record.pop("scores", None)
                    record.setdefault("prompt_template", args.prompt_template)
                    record.setdefault(
                        "prompt_prefills_think", args.prompt_prefills_think
                    )
                    records.append(record)
            grouped_records[(version, task)] = records
    previous_summary_path = model_root / "summary.json"
    metadata: dict[str, Any] = {"rescored": True}
    if previous_summary_path.is_file():
        previous = load_json(previous_summary_path)
        metadata.update(
            {
                key: value
                for key, value in previous.items()
                if key not in {"model_name", "versions"}
            }
        )
        metadata["rescored"] = True
    write_scored_results(
        Path(args.output_dir), args.model_name, grouped_records, metadata
    )
    summary = run_official_scorer(args, args.model_name)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def gold_prediction(
    task: str, reference: str, prompt_template: str = "rlla"
) -> str:
    if prompt_template == "official":
        return reference if task == "api" else f"AI: {reference}"
    if task == "api":
        call = parse_official_api_call(reference)
        return (
            "<think> Use the required API. </think>\n<tool_call>\n"
            + json.dumps(call, ensure_ascii=False)
            + "\n</tool_call>"
        )
    return f"<think> Return the requested result. </think>\n<response> {reference} </response>"


def run_validate(args: argparse.Namespace) -> None:
    counts = Counter()
    failures: list[str] = []
    for version, task, path, reference_field in selected_groups(args):
        records = load_group_records(
            version, task, path, reference_field, 0, args.prompt_template
        )
        for record in records:
            record["prediction"] = gold_prediction(
                task, record["reference"], args.prompt_template
            )
            scored = score_record(record)
            if task == "api" and not scored["scores"]["call_exact_match"]:
                failures.append(f"{version}/{task}/{record['index']}")
            if task == "response" and scored["scores"]["rouge_l_f1"] != 1.0:
                failures.append(f"{version}/{task}/{record['index']}")
            counts[f"{version}/{task}"] += 1
    if failures:
        raise AssertionError(f"gold validation failures: {failures[:20]}")
    print(json.dumps({"status": "ok", "validated": counts}, indent=2))


def add_selection_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--versions", nargs="+", choices=tuple(DATASETS), default=list(DATASETS)
    )
    parser.add_argument(
        "--tasks", nargs="+", choices=("api", "response"), default=["api", "response"]
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument(
        "--prompt-template",
        choices=PROMPT_TEMPLATE_CHOICES,
        default="rlla",
        help=(
            "Use the exact RLLA training template or API-Bank's distributed "
            "instruction+input prompt."
        ),
    )
    parser.add_argument(
        "--v3-icl",
        action="store_true",
        help="Use level-3-batch-inf-icl.json instead of the zero-shot v3 call file.",
    )


def add_official_score_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--official-python", type=Path, default=DEFAULT_OFFICIAL_PYTHON)
    parser.add_argument(
        "--official-runtime-dir", type=Path, default=DEFAULT_OFFICIAL_RUNTIME_DIR
    )
    parser.add_argument(
        "--tool-search-model", type=Path, default=DEFAULT_TOOL_SEARCH_MODEL
    )
    parser.add_argument(
        "--official-score-seed",
        type=int,
        default=42,
        help="Random seed for executable official APIs.",
    )
    parser.add_argument(
        "--skip-official-score",
        action="store_true",
        help="Write generation/static diagnostics without executing official APIs.",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Generate predictions and score them.")
    add_selection_arguments(run_parser)
    add_official_score_arguments(run_parser)
    run_parser.add_argument("--model-path", required=True)
    run_parser.add_argument("--model-name")
    run_parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    run_parser.add_argument("--max-samples", type=int, default=0)
    run_parser.add_argument("--max-tokens", type=int, default=1024)
    run_parser.add_argument("--max-model-len", type=int, default=8192)
    run_parser.add_argument(
        "--data-parallel-size",
        type=int,
        default=1,
        help="Number of replicated vLLM engines used to distribute requests.",
    )
    run_parser.add_argument("--tensor-parallel-size", type=int, default=1)
    run_parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    run_parser.add_argument("--dtype", default="bfloat16")
    run_parser.add_argument("--temperature", type=float, default=0.0)
    run_parser.add_argument("--top-p", type=float, default=1.0)
    run_parser.add_argument("--seed", type=int, default=42)
    run_parser.add_argument("--force-think-prefix", action="store_true")
    run_parser.add_argument("--enforce-eager", action="store_true")
    run_parser.add_argument("--save-prompts", action="store_true")
    run_parser.set_defaults(func=run_generation)

    score_parser = subparsers.add_parser("score", help="Rescore saved predictions.")
    add_selection_arguments(score_parser)
    add_official_score_arguments(score_parser)
    score_parser.add_argument("--model-name", required=True)
    score_parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    score_parser.add_argument(
        "--prompt-prefills-think",
        action="store_true",
        help=(
            "For legacy saved predictions, mark that the rendered chat prompt "
            "already supplied the opening <think> tag."
        ),
    )
    score_parser.set_defaults(func=run_score_only)

    validate_parser = subparsers.add_parser(
        "validate", help="Validate all data adapters and scorers against gold labels."
    )
    add_selection_arguments(validate_parser)
    validate_parser.set_defaults(func=run_validate)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
