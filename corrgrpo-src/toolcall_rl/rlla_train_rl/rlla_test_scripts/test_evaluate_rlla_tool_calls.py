"""CPU-only tests for RLLA percentage and exact-match evaluation logic."""

import csv
import unittest
from collections import Counter
from pathlib import Path

from rlla_test_scripts.evaluate_rlla_tool_calls import (
    add_qwen3_think_prefill,
    all_tool_fields_correct,
    complete_experiment_groups,
    normalize_rewards,
    ordered_summaries,
    repair_missing_tool_call_closing_tag,
    repair_missing_tool_call_opening_tag,
    render_complete_groups_table,
    reward_to_percent,
    score_response,
)


GROUND_TRUTH = """<think>use the news tool</think>
<tool_call>
{"name": "GetNews", "parameters": {"page": "1"}}
</tool_call>"""


class RLLAToolCallEvaluationTest(unittest.TestCase):
    def test_only_qwen3_thinking_rl_models_get_exact_think_prefill(self):
        for model_name in (
            "qwen3_grpo_4b_think",
            "qwen3_grpo_cov_coeff_train_4b_think",
        ):
            with self.subTest(model_name=model_name):
                prompt, prefill = add_qwen3_think_prefill("assistant\n", model_name)
                self.assertEqual(prompt, "assistant\n<think> ")
                self.assertEqual(prefill, "<think> ")

    def test_existing_qwen3_thinking_prefill_is_not_duplicated(self):
        prompt, prefill = add_qwen3_think_prefill(
            "assistant\n<think>\n", "qwen3_grpo_4b_think"
        )
        self.assertEqual(prompt, "assistant\n<think>\n")
        self.assertEqual(prefill, "<think>\n")

    def test_non_qwen3_prompt_is_unchanged(self):
        prompt, prefill = add_qwen3_think_prefill("assistant\n", "qwen25_7b")
        self.assertEqual(prompt, "assistant\n")
        self.assertEqual(prefill, "")

    def test_other_qwen3_prompts_are_unchanged(self):
        for model_name in (
            "qwen3_8b",
            "qwen3_4b",
            "qwen3_4b_thinking",
            "qwen3_grpo_8b",
            "qwen3_grpo_cov_coeff_train_8b",
        ):
            with self.subTest(model_name=model_name):
                prompt, prefill = add_qwen3_think_prefill("assistant\n", model_name)
                self.assertEqual(prompt, "assistant\n")
                self.assertEqual(prefill, "")

    def test_evaluation_list_contains_base_sft_and_rl_models(self):
        path = Path(__file__).with_name("evaluation_models.tsv")
        with path.open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream, delimiter="|"))

        self.assertEqual(
            Counter(row["model_type"] for row in rows),
            {"base": 9, "sft": 8, "grpo": 6, "grpo_cov_coeff": 6},
        )
        self.assertEqual(len({row["model_name"] for row in rows}), len(rows))
        self.assertFalse(any("smoke" in row["model_name"] for row in rows))

    def test_model_families_use_requested_stage_order(self):
        path = Path(__file__).with_name("evaluation_models.tsv")
        with path.open(encoding="utf-8", newline="") as stream:
            stages = [row["model_type"] for row in csv.DictReader(stream, delimiter="|")]

        family_sizes = (2, 4, 4, 4, 2, 2, 4, 4, 3)
        expected = (
            ["base", "sft"],
            ["base", "sft", "grpo", "grpo_cov_coeff"],
            ["base", "sft", "grpo", "grpo_cov_coeff"],
            ["base", "sft", "grpo", "grpo_cov_coeff"],
            ["base", "sft"],
            ["base", "sft"],
            ["base", "sft", "grpo", "grpo_cov_coeff"],
            ["base", "sft", "grpo", "grpo_cov_coeff"],
            ["base", "grpo", "grpo_cov_coeff"],
        )
        offset = 0
        for size, expected_stages in zip(family_sizes, expected, strict=True):
            self.assertEqual(stages[offset : offset + size], expected_stages)
            offset += size
        self.assertEqual(offset, len(stages))

    def test_base_names_do_not_have_artificial_base_suffix(self):
        path = Path(__file__).with_name("evaluation_models.tsv")
        with path.open(encoding="utf-8", newline="") as stream:
            base_names = {
                row["model_name"]
                for row in csv.DictReader(stream, delimiter="|")
                if row["model_type"] == "base"
            }

        self.assertEqual(
            base_names,
            {
                "qwen25_0_5b_instruct",
                "qwen25_1_5b_instruct",
                "qwen25_3b_instruct",
                "qwen25_7b_instruct",
                "qwen3_8b_base",  # Official Qwen3-8B-Base model name.
                "qwen3_4b",
                "qwen3_4b_thinking",
                "qwen3_8b",
                "llama32_3b_instruct",
            },
        )

    def test_exactly_five_model_families_have_all_four_stages(self):
        directory = Path(__file__).with_name("rlla_tool_call_results")
        if not directory.is_dir():
            self.skipTest("model result directory is only available after evaluation")
        summaries = ordered_summaries(
            directory, Path(__file__).with_name("evaluation_models.tsv")
        )
        groups = complete_experiment_groups(summaries)
        self.assertEqual(len(groups), 5)
        self.assertTrue(
            all(
                [row["model_type"] for row in group]
                == ["base", "sft", "grpo", "grpo_cov_coeff"]
                for group in groups
            )
        )

    def test_complete_group_table_puts_every_delta_beside_its_value(self):
        metric_names = (
            "combined_reward",
            "tool_call_reward",
            "function_name",
            "parameter_name",
            "parameter_value",
            "format",
            "all_tool_fields_correct",
        )
        group = []
        for stage, value in (
            ("base", 5.0),
            ("sft", 10.0),
            ("grpo", 15.0),
            ("grpo_cov_coeff", 20.0),
        ):
            group.append(
                {
                    "model_type": stage,
                    "model_name": f"model_{stage}",
                    "metrics_percent": {name: value for name in metric_names},
                    "all_tool_fields_correct_count": int(value / 10),
                    "tool_call_sample_count": 10,
                }
            )

        table = "\n".join(render_complete_groups_table([group]))
        for header in (
            "Combined",
            "Total reward",
            "Function",
            "Param name",
            "Param value",
            "Format",
            "All correct",
        ):
            self.assertIn(f"<th>{header}</th>", table)
        self.assertNotIn("<th>Δ", table)
        self.assertEqual(table.count("20.00% (Δ +5.00 pp)"), 6)
        self.assertIn("20.00% (2/10) (Δ +5.00 pp)", table)

    def test_qwen25_7b_normal_eos_gets_missing_tool_call_close_repaired(self):
        response, repaired = repair_missing_tool_call_closing_tag(
            '<think>x</think>\n<tool_call>\n{"name":"f","parameters":{}}',
            "qwen25_7b_instruct",
            "stop",
        )
        self.assertTrue(repaired)
        self.assertTrue(response.endswith("\n</tool_call>"))

    def test_length_truncation_does_not_get_tool_call_close_repaired(self):
        original = '<think>x</think>\n<tool_call>\n{"name":"f"'
        response, repaired = repair_missing_tool_call_closing_tag(
            original, "qwen25_7b_instruct", "length"
        )
        self.assertFalse(repaired)
        self.assertEqual(response, original)

    def test_other_models_do_not_get_tool_call_close_repaired(self):
        original = '<think>x</think>\n<tool_call>\n{"name":"f","parameters":{}}'
        response, repaired = repair_missing_tool_call_closing_tag(
            original, "qwen25_3b_instruct", "stop"
        )
        self.assertFalse(repaired)
        self.assertEqual(response, original)

    def test_qwen3_thinking_sft_gets_missing_tool_call_open_repaired(self):
        original = '{"name":"f","parameters":{}}\n</tool_call>'
        response, repaired = repair_missing_tool_call_opening_tag(
            original, "qwen3_4b_think_rlla_sft_400", "stop"
        )
        self.assertTrue(repaired)
        self.assertEqual(
            response,
            '<tool_call>\n{"name":"f","parameters":{}}\n</tool_call>',
        )

    def test_invalid_json_does_not_get_tool_call_open_repaired(self):
        original = 'not-json\n</tool_call>'
        response, repaired = repair_missing_tool_call_opening_tag(
            original, "qwen3_4b_think_rlla_sft_400", "stop"
        )
        self.assertFalse(repaired)
        self.assertEqual(response, original)

    def test_length_truncation_does_not_get_tool_call_open_repaired(self):
        original = '{"name":"f","parameters":{}}\n</tool_call>'
        response, repaired = repair_missing_tool_call_opening_tag(
            original, "qwen3_4b_think_rlla_sft_400", "length"
        )
        self.assertFalse(repaired)
        self.assertEqual(response, original)

    def test_other_models_do_not_get_tool_call_open_repaired(self):
        original = '{"name":"f","parameters":{}}\n</tool_call>'
        response, repaired = repair_missing_tool_call_opening_tag(
            original, "qwen3_8b_rlla_sft_400", "stop"
        )
        self.assertFalse(repaired)
        self.assertEqual(response, original)

    def test_reward_ranges_map_to_zero_and_one_hundred_percent(self):
        self.assertEqual(reward_to_percent(-3.0, -3.0, 3.0), 0.0)
        self.assertEqual(reward_to_percent(0.0, -3.0, 3.0), 50.0)
        self.assertEqual(reward_to_percent(3.0, -3.0, 3.0), 100.0)

    def test_exact_tool_call_gets_full_component_scores(self):
        result = score_response(GROUND_TRUTH, GROUND_TRUTH, "qwen_test", 20)

        self.assertTrue(result["all_tool_fields_correct"])
        self.assertEqual(result["percent_rewards"]["tool_call_reward_percent"], 100.0)
        self.assertEqual(result["percent_rewards"]["function_name_percent"], 100.0)
        self.assertEqual(result["percent_rewards"]["parameter_name_percent"], 100.0)
        self.assertEqual(result["percent_rewards"]["parameter_value_percent"], 100.0)

    def test_wrong_parameter_value_is_not_all_correct(self):
        prediction = GROUND_TRUTH.replace('"1"', '"2"')
        result = score_response(prediction, GROUND_TRUTH, "qwen_test", 20)

        self.assertFalse(result["all_tool_fields_correct"])
        self.assertEqual(result["percent_rewards"]["function_name_percent"], 100.0)
        self.assertEqual(result["percent_rewards"]["parameter_name_percent"], 100.0)
        self.assertEqual(result["percent_rewards"]["parameter_value_percent"], 0.0)

    def test_extra_parameter_name_is_not_all_correct(self):
        prediction = GROUND_TRUTH.replace(
            '"page": "1"', '"page": "1", "unexpected": true'
        )
        result = score_response(prediction, GROUND_TRUTH, "qwen_test", 20)

        self.assertFalse(result["all_tool_fields_correct"])
        self.assertLess(result["percent_rewards"]["parameter_name_percent"], 100.0)

    def test_non_tool_sample_cannot_count_as_all_correct(self):
        full_rewards = {
            "function_name_reward": 0.5,
            "parameter_reward": 1.0,
            "values_reward": 1.5,
        }
        self.assertFalse(all_tool_fields_correct(full_rewards, has_tool_call=False))

    def test_normalization_uses_each_rlla_component_range(self):
        normalized = normalize_rewards(
            {
                "score": 0.5,
                "accuracy_reward": 0.0,
                "format_reward": 0.5,
                "function_name_reward": 0.0,
                "parameter_reward": 0.0,
                "values_reward": 0.0,
            }
        )
        self.assertTrue(all(value == 50.0 for value in normalized.values()))


if __name__ == "__main__":
    unittest.main(verbosity=2)
