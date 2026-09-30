#!/usr/bin/env python3
"""Tests for LaTeX best/runner-up highlighting."""

from __future__ import annotations

import unittest

from verl_training.aggregate_val_reward_results import rank_styles


def row(name: str, score: float) -> dict[str, object]:
    return {"name": name, "score": score}


class TestRankStyles(unittest.TestCase):
    def test_unique_best_and_runner_up(self) -> None:
        rows = [
            row("Qwen2.5-3B-Instruct", 0.1),
            row("qwen25_3b_instruct_agentdojo_grpo", 0.2),
            row("qwen25_3b_instruct_agentdojo_grpo_cov_coeff", 0.3),
        ]
        styles = rank_styles(rows, "score", lower_is_better=False, tie_seed="test")
        self.assertEqual(styles[rows[2]["name"]], "bold")
        self.assertEqual(styles[rows[1]["name"]], "underline")

    def test_lower_is_better(self) -> None:
        rows = [row("base_3b", 0.3), row("model_3b_grpo", 0.2), row("model_3b_cov_coeff", 0.1)]
        styles = rank_styles(rows, "score", lower_is_better=True, tie_seed="test")
        self.assertEqual(styles["model_3b_cov_coeff"], "bold")
        self.assertEqual(styles["model_3b_grpo"], "underline")

    def test_tied_best_has_one_bold_and_one_underline(self) -> None:
        rows = [row("base_3b", 0.5), row("model_3b_grpo", 0.5), row("model_3b_cov_coeff", 0.5)]
        first = rank_styles(rows, "score", lower_is_better=False, tie_seed="test")
        second = rank_styles(rows, "score", lower_is_better=False, tie_seed="test")
        self.assertEqual(list(first.values()).count("bold"), 1)
        self.assertEqual(list(first.values()).count("underline"), 1)
        self.assertEqual(first, second)

    def test_tied_runner_up_prefers_cov_coeff(self) -> None:
        rows = [row("base_3b", 0.5), row("model_3b_grpo", 0.9), row("model_3b_cov_coeff", 0.5)]
        styles = rank_styles(rows, "score", lower_is_better=False, tie_seed="test")
        self.assertEqual(styles["model_3b_grpo"], "bold")
        self.assertEqual(styles["model_3b_cov_coeff"], "underline")
        self.assertNotIn("base_3b", styles)


if __name__ == "__main__":
    unittest.main()
