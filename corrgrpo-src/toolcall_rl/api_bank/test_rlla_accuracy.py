#!/usr/bin/env python3
"""Diverse regression tests for accuracy on RLLA-formatted model replies.

These tests deliberately separate the evaluator's static diagnostics from
API-Bank's official execution metric.  A reply can have an execution-correct
first call while failing strict formatting or static call exact match.
"""

from __future__ import annotations

import json
import unittest
from copy import deepcopy
from statistics import fmean
from typing import Any

import evaluate_api_bank as evaluator
import official_accuracy


SEARCH_REFERENCE = (
    "API-Request: [Search(query='北京 天气', limit=3, "
    "filters={'lang': 'zh', 'safe': True})]"
)
SEARCH_PARAMETERS = {
    "query": "北京 天气",
    "limit": 3,
    "filters": {"lang": "zh", "safe": True},
}


def tool_reply(
    name: str = "Search",
    parameters: dict[str, Any] | None = None,
    *,
    include_think: bool = True,
    use_arguments_alias: bool = False,
    response: str | None = None,
) -> str:
    """Build one representative RLLA reply without hiding raw edge cases."""
    parameter_key = "arguments" if use_arguments_alias else "parameters"
    call = {
        "name": name,
        parameter_key: SEARCH_PARAMETERS if parameters is None else parameters,
    }
    parts = []
    if include_think:
        parts.append("<think>需要调用搜索工具。</think>")
    parts.append(
        "<tool_call>\n"
        + json.dumps(call, ensure_ascii=False)
        + "\n</tool_call>"
    )
    if response is not None:
        parts.append(f"<response>{response}</response>")
    return "\n".join(parts)


def api_record(
    prediction: str,
    *,
    version: str = "v1",
    index: int = 0,
    sample_id: str | None = None,
) -> dict[str, Any]:
    return {
        "version": version,
        "task": "api",
        "index": index,
        "sample_id": sample_id,
        "prompt_template": "rlla",
        "prediction": prediction,
        "reference": SEARCH_REFERENCE,
    }


class StaticRllaApiAccuracyTest(unittest.TestCase):
    def test_diverse_api_replies_score_each_accuracy_dimension(self):
        cases = [
            {
                "name": "canonical_exact_unicode_and_nested_parameters",
                "prediction": tool_reply(),
                "expected": (True, True, True, True, True, True),
            },
            {
                "name": "arguments_alias_is_accepted",
                "prediction": tool_reply(use_arguments_alias=True),
                "expected": (True, True, True, True, True, True),
            },
            {
                "name": "tool_call_plus_response_is_still_one_exact_call",
                "prediction": tool_reply(response="正在查询。"),
                "expected": (True, True, True, True, True, True),
            },
            {
                "name": "missing_think_keeps_semantic_accuracy_but_fails_format",
                "prediction": tool_reply(include_think=False),
                "expected": (False, True, True, True, True, True),
            },
            {
                "name": "two_calls_make_static_exact_match_false",
                "prediction": (
                    "<think>需要两个调用。</think>\n<tool_call>\n"
                    + json.dumps(
                        {"name": "Search", "parameters": SEARCH_PARAMETERS},
                        ensure_ascii=False,
                    )
                    + "\n"
                    + json.dumps(
                        {"name": "Audit", "parameters": {}},
                        ensure_ascii=False,
                    )
                    + "\n</tool_call>"
                ),
                "expected": (True, True, True, True, True, False),
            },
            {
                "name": "wrong_function_with_same_parameters",
                "prediction": tool_reply(name="WeatherSearch"),
                "expected": (True, True, False, True, True, False),
            },
            {
                "name": "missing_parameter_name",
                "prediction": tool_reply(
                    parameters={"query": "北京 天气", "limit": 3}
                ),
                "expected": (True, True, True, False, False, False),
            },
            {
                "name": "extra_parameter_name",
                "prediction": tool_reply(
                    parameters={**SEARCH_PARAMETERS, "timezone": "Asia/Shanghai"}
                ),
                "expected": (True, True, True, False, False, False),
            },
            {
                "name": "wrong_nested_boolean_value",
                "prediction": tool_reply(
                    parameters={
                        **SEARCH_PARAMETERS,
                        "filters": {"lang": "zh", "safe": False},
                    }
                ),
                "expected": (True, True, True, True, False, False),
            },
            {
                "name": "numeric_string_is_not_static_exact",
                "prediction": tool_reply(
                    parameters={**SEARCH_PARAMETERS, "limit": "3"}
                ),
                "expected": (True, True, True, True, False, False),
            },
            {
                "name": "malformed_json_has_no_valid_call",
                "prediction": (
                    '<think>调用。</think><tool_call>{"name":"Search",'
                    '"parameters":{"query":"北京 天气"}</tool_call>'
                ),
                "expected": (True, False, False, False, False, False),
            },
            {
                "name": "response_only_has_no_api_call",
                "prediction": "<think>无需工具。</think><response>直接回答。</response>",
                "expected": (True, False, False, False, False, False),
            },
        ]

        keys = (
            "rlla_format_valid",
            "api_call_valid",
            "function_name_correct",
            "parameter_names_correct",
            "parameter_values_correct",
            "call_exact_match",
        )
        for case in cases:
            with self.subTest(case=case["name"]):
                scores = evaluator.score_api_prediction(
                    case["prediction"], SEARCH_REFERENCE, prompt_template="rlla"
                )
                self.assertEqual(
                    tuple(scores[key] for key in keys), case["expected"]
                )

    def test_api_summary_averages_independent_dimensions(self):
        predictions = [
            tool_reply(),
            tool_reply(include_think=False),
            tool_reply(parameters={**SEARCH_PARAMETERS, "limit": 99}),
            "<think>无需工具。</think><response>直接回答。</response>",
        ]
        records = [
            evaluator.score_record(api_record(prediction, index=index))
            for index, prediction in enumerate(predictions)
        ]
        summary = evaluator.summarize_group("v1", "api", records)

        self.assertEqual(summary["sample_count"], 4)
        self.assertEqual(summary["rlla_format_rate"], 0.75)
        self.assertEqual(summary["tool_block_valid_rate"], 0.75)
        self.assertEqual(summary["function_name_accuracy"], 0.75)
        self.assertEqual(summary["parameter_name_accuracy"], 0.75)
        self.assertEqual(summary["parameter_value_accuracy"], 0.5)
        self.assertEqual(summary["call_exact_match_accuracy"], 0.5)

    def test_v3_dialogue_success_requires_every_call_in_dialogue(self):
        records = []
        predictions = [
            ("dialogue-a", tool_reply()),
            ("dialogue-a", tool_reply()),
            ("dialogue-b", tool_reply()),
            (
                "dialogue-b",
                tool_reply(parameters={**SEARCH_PARAMETERS, "limit": 99}),
            ),
            ("dialogue-c", tool_reply()),
        ]
        for index, (sample_id, prediction) in enumerate(predictions):
            records.append(
                evaluator.score_record(
                    api_record(
                        prediction,
                        version="v3",
                        index=index,
                        sample_id=sample_id,
                    )
                )
            )

        summary = evaluator.summarize_group("v3", "api", records)
        self.assertEqual(summary["call_exact_match_accuracy"], 0.8)
        self.assertEqual(summary["dialogue_sample_count"], 3)
        self.assertEqual(summary["dialogue_success_count"], 2)
        self.assertAlmostEqual(summary["dialogue_success_accuracy"], 2 / 3)


class RllaResponseAccuracyTest(unittest.TestCase):
    def test_diverse_response_replies_and_summary(self):
        reference = "会议已安排在明天下午三点。"
        predictions = [
            "<think>已经完成。</think><response>会议已安排在明天下午三点。</response>",
            "<think>已经完成。</think>  会议已安排在明天下午三点。  ",
            "<think>已经完成。</think><response>会议安排在明天下午。</response>",
            "<think>没有内容。</think><response></response>",
        ]
        records = []
        for index, prediction in enumerate(predictions):
            records.append(
                evaluator.score_record(
                    {
                        "version": "v1",
                        "task": "response",
                        "index": index,
                        "prompt_template": "rlla",
                        "prediction": prediction,
                        "reference": reference,
                    }
                )
            )

        first, fallback, partial, empty = [record["scores"] for record in records]
        self.assertTrue(first["response_block_valid"])
        self.assertTrue(first["exact_match"])
        self.assertFalse(fallback["response_block_valid"])
        self.assertTrue(fallback["exact_match"])
        self.assertGreater(partial["rouge_l_f1"], 0.0)
        self.assertLess(partial["rouge_l_f1"], 1.0)
        self.assertEqual(empty["rouge_l_f1"], 0.0)

        summary = evaluator.summarize_group("v1", "response", records)
        self.assertEqual(summary["rlla_format_rate"], 0.75)
        self.assertEqual(summary["response_block_valid_rate"], 0.75)
        self.assertEqual(summary["response_exact_match_accuracy"], 0.5)
        self.assertAlmostEqual(
            summary["rouge_l_f1"],
            fmean(record["scores"]["rouge_l_f1"] for record in records),
        )

    def test_multiple_response_blocks_are_joined_in_order(self):
        prediction = (
            "<think>分两段回答。</think>"
            "<response>第一段</response><response>第二段</response>"
        )
        scores = evaluator.score_response_prediction(
            prediction, "第一段 第二段", prompt_template="rlla"
        )
        self.assertEqual(scores["extracted_response"], "第一段\n第二段")
        self.assertTrue(scores["exact_match"])


class FakeGroundTruthIndex:
    @staticmethod
    def api_ground_truth(row: dict[str, Any]) -> dict[str, Any]:
        return deepcopy(row["ground_truth"])


class FakeExecutionExecutor:
    """Small deterministic executor used to test official aggregation logic."""

    @staticmethod
    def execute_and_check(
        version: str,
        predicted_call: dict[str, Any],
        ground_truth: dict[str, Any],
    ) -> dict[str, Any]:
        del version
        name = predicted_call["name"]
        if name != ground_truth["api_name"]:
            return {
                "correct": False,
                "error_category": "API_NAME_MISMATCH",
                "execution_result": None,
            }
        try:
            parameters = predicted_call["parameters"]
            if name == "Add":
                result = parameters["a"] + parameters["b"]
            elif name == "Echo":
                result = parameters["value"]
            else:
                raise KeyError(name)
        except Exception as exc:  # exercise official error accounting
            return {
                "correct": False,
                "error_category": "EXECUTION_ERROR",
                "execution_result": None,
                "execution_exception": f"{type(exc).__name__}: {exc}",
            }
        correct = result == ground_truth["result"]
        return {
            "correct": correct,
            "error_category": None if correct else "RESULT_MISMATCH",
            "execution_result": result,
        }


def official_row(
    prediction: str,
    api_name: str,
    ground_truth_parameters: dict[str, Any],
    result: Any,
) -> dict[str, Any]:
    return {
        "version": "v1",
        "task": "api",
        "prompt_template": "rlla",
        "prediction": prediction,
        "ground_truth": {
            "api_name": api_name,
            "parameters": ground_truth_parameters,
            "result": result,
        },
    }


class OfficialExecutionAccuracyWithRllaRepliesTest(unittest.TestCase):
    def test_official_execution_accuracy_uses_parsed_first_call_and_result(self):
        rows = [
            official_row(
                '<think>算术。</think><tool_call>{"name":"Add",'
                '"parameters":{"a":2,"b":3}}</tool_call>',
                "Add",
                {"a": 2, "b": 3},
                5,
            ),
            # Different parameters can still be execution-correct because the
            # official metric checks the returned result, not static equality.
            official_row(
                '<think>等价计算。</think><tool_call>{"name":"Add",'
                '"arguments":{"a":1,"b":4}}</tool_call>',
                "Add",
                {"a": 2, "b": 3},
                5,
            ),
            # Official scoring deliberately executes only the first call.
            official_row(
                '<think>两个调用。</think><tool_call>{"name":"Add",'
                '"parameters":{"a":2,"b":3}}'
                '{"name":"Echo","parameters":{"value":"ignored"}}</tool_call>',
                "Add",
                {"a": 2, "b": 3},
                5,
            ),
            official_row(
                "<think>格式坏了。</think><tool_call>{not json}</tool_call>",
                "Add",
                {"a": 2, "b": 3},
                5,
            ),
            official_row(
                '<think>工具错了。</think><tool_call>{"name":"Echo",'
                '"parameters":{"value":5}}</tool_call>',
                "Add",
                {"a": 2, "b": 3},
                5,
            ),
            # RLLA format is invalid without <think>, but the parsed call can
            # still receive official execution credit.
            official_row(
                '<tool_call>{"name":"Add","parameters":{"a":2,"b":3}}'
                "</tool_call>",
                "Add",
                {"a": 2, "b": 3},
                5,
            ),
            official_row(
                '<think>回显。</think><tool_call>{"name":"Echo",'
                '"parameters":{"value":{"city":"北京",'
                '"tags":["晴",true]}}}</tool_call>',
                "Echo",
                {"value": {"city": "北京", "tags": ["晴", True]}},
                {"city": "北京", "tags": ["晴", True]},
            ),
            official_row(
                '<think>结果错误。</think><tool_call>{"name":"Add",'
                '"parameters":{"a":10,"b":20}}</tool_call>',
                "Add",
                {"a": 2, "b": 3},
                5,
            ),
        ]

        scored_rows, summary = official_accuracy.score_api_rows(
            deepcopy(rows), FakeExecutionExecutor(), FakeGroundTruthIndex()
        )

        self.assertEqual(summary["sample_count"], 8)
        self.assertEqual(summary["correct_count"], 5)
        self.assertEqual(summary["accuracy"], 5 / 8)
        self.assertEqual(summary["official_execution_accuracy"], 5 / 8)
        self.assertEqual(
            summary["error_counts"],
            {
                "API_NAME_MISMATCH": 1,
                "NO_API_CALL": 1,
                "RESULT_MISMATCH": 1,
            },
        )
        self.assertTrue(scored_rows[1]["official_scores"]["correct"])
        self.assertEqual(
            scored_rows[1]["official_scores"]["predicted_call"]["parameters"],
            {"a": 1, "b": 4},
        )
        self.assertIsNone(scored_rows[3]["official_scores"]["predicted_call"])


if __name__ == "__main__":
    unittest.main()
