#!/usr/bin/env python3
"""Tests for AgentDojo VERL validation metric aggregation."""

from __future__ import annotations

import json
import unittest

from verl_training.summarize_val_reward_eval import build_report, final_session_records


def row(
    uid: str,
    case_type: str,
    *,
    utility: float,
    attack_success: float = 0.0,
    format_valid: float = 1.0,
    valid: float = 1.0,
    suite: str = "workspace",
    attack: str | None = None,
    user_task_id: str = "user_task_0",
) -> dict[str, object]:
    return {
        "uid": uid,
        "gts": json.dumps(
            {
                "case_type": case_type,
                "suite": suite,
                "attack": attack,
                "user_task_id": user_task_id,
            }
        ),
        "utility_reward": utility,
        "attack_success": attack_success,
        "format_reward": format_valid,
        "valid_reward": valid,
    }


class TestValidationAggregation(unittest.TestCase):
    def test_three_requested_metrics_use_correct_denominators(self) -> None:
        records = [
            row("clean-a_0_0", "clean", utility=1, user_task_id="task-a"),
            row("clean-b_0_0", "clean", utility=0, user_task_id="task-b"),
            row(
                "attack-a_0_0",
                "security",
                utility=1,
                attack_success=0,
                attack="important_instructions",
                user_task_id="task-a",
            ),
            row(
                "attack-b_0_0",
                "security",
                utility=0,
                attack_success=1,
                attack="important_instructions",
                user_task_id="task-b",
            ),
            row(
                "attack-c_0_0",
                "security",
                utility=1,
                attack_success=0,
                attack="tool_knowledge",
                user_task_id="task-b",
            ),
        ]
        metrics = build_report(records)["overall"]

        self.assertEqual(metrics["clean_utility"]["value"], 0.5)
        self.assertAlmostEqual(metrics["utility_under_attack"]["value"], 2 / 3)
        self.assertAlmostEqual(metrics["asr"]["value"], 1 / 3)
        # Per attack sample: [1.0, 0.0, 0.5], whose mean is 0.5.
        self.assertAlmostEqual(metrics["joint_accuracy"]["value"], 0.5)
        self.assertAlmostEqual(metrics["joint_accuracy"]["score_sum"], 1.5)
        self.assertEqual(metrics["joint_accuracy"]["num_cases"], 3)
        self.assertEqual(metrics["clean_utility"]["num_cases"], 2)
        self.assertEqual(metrics["utility_under_attack"]["num_cases"], 3)
        self.assertEqual(metrics["asr"]["num_cases"], 3)

    def test_format_invalid_attack_uses_gated_components(self) -> None:
        records = [
            row("clean-a_0_0", "clean", utility=1, user_task_id="task-a"),
            row("clean-b_0_0", "clean", utility=1, user_task_id="task-b"),
            row(
                "safe_0_0",
                "security",
                utility=1,
                attack_success=0,
                attack="important_instructions",
                user_task_id="task-a",
            ),
            # These are the fields emitted by reward.py even if the official
            # checker result before gating was useful or compromised.
            row(
                "malformed_0_0",
                "security",
                utility=0,
                attack_success=0,
                format_valid=0,
                valid=0,
                attack="important_instructions",
                user_task_id="task-b",
            ),
        ]
        metrics = build_report(records)["overall"]

        self.assertEqual(metrics["utility_under_attack"]["value"], 0.5)
        self.assertEqual(metrics["asr"]["value"], 0.0)
        # The malformed attack contributes attack utility 0, while its paired
        # clean utility is still 1, so its requested joint score is 0.5.
        self.assertEqual(metrics["joint_accuracy"]["value"], 0.75)
        self.assertEqual(metrics["format_valid_rate"]["value"], 0.75)
        self.assertEqual(metrics["utility_under_attack"]["num_cases"], 2)

    def test_only_final_agent_loop_output_is_counted(self) -> None:
        records = [
            row("prompt-a_7_0", "clean", utility=0, user_task_id="task-a"),
            row("prompt-a_7_2", "clean", utility=1, user_task_id="task-a"),
            row("prompt-a_7_1", "clean", utility=0, user_task_id="task-a"),
            row("prompt-b_9_0", "clean", utility=0, user_task_id="task-b"),
        ]
        selected = final_session_records(records)
        self.assertEqual([item["uid"] for item in selected], ["prompt-a_7_2", "prompt-b_9_0"])
        self.assertEqual(build_report(records)["overall"]["clean_utility"]["value"], 0.5)

    def test_reports_per_suite_and_attack(self) -> None:
        records = [
            row("a_0_0", "clean", utility=1, suite="workspace", user_task_id="workspace-task"),
            row("b-clean_0_0", "clean", utility=0, suite="slack", user_task_id="slack-task"),
            row(
                "b_0_0",
                "security",
                utility=0,
                attack_success=1,
                suite="slack",
                attack="tool_knowledge",
                user_task_id="slack-task",
            ),
        ]
        report = build_report(records)

        self.assertEqual(set(report["by_suite"]), {"slack", "workspace"})
        self.assertEqual(set(report["by_attack"]), {"tool_knowledge"})
        self.assertEqual(report["by_attack"]["tool_knowledge"]["asr"]["value"], 1.0)

    def test_security_case_requires_matching_clean_task(self) -> None:
        records = [
            row(
                "attack_0_0",
                "security",
                utility=1,
                attack="tool_knowledge",
                user_task_id="missing-task",
            )
        ]
        with self.assertRaisesRegex(ValueError, "No matching clean sample"):
            build_report(records)

    def test_missing_reward_field_fails_loudly(self) -> None:
        record = row("broken_0_0", "clean", utility=1)
        del record["format_reward"]
        with self.assertRaisesRegex(ValueError, "Missing reward fields"):
            build_report([record])


if __name__ == "__main__":
    unittest.main(verbosity=2)
