#!/usr/bin/env python3
"""Comprehensive, model-free tests for every reward.py component.

The suite intentionally mixes normal solutions, malformed responses, wrong
answers, runtime failures, an infinite loop, blocked file access, AST variants,
custom weight configurations, and efficiency edge cases.
"""

from __future__ import annotations

import json
import math
import unittest

from reward import (
    compute_ast_similarity,
    compute_efficiency_score,
    compute_score,
    extract_code,
)


REFERENCE_CODE = """class Solution:
    def solve(self, value):
        return value + 1
"""

BASE_VERIFIER = {
    "task_id": "reward-comprehensive-test",
    "prompt": "# no helper imports required",
    "test": "def check(candidate):\n    assert candidate(1) == 2\n",
    "entry_point": "Solution().solve",
    "num_tests": 1,
    "reference_solution": REFERENCE_CODE,
}


def fenced(code: str, language: str = "python") -> str:
    return f"```{language}\n{code.rstrip()}\n```"


def score(solution: str, verifier=None, **kwargs):
    return compute_score(
        "leetcodedataset",
        solution,
        BASE_VERIFIER if verifier is None else verifier,
        **kwargs,
    )


class CodeExtractionTests(unittest.TestCase):
    def test_python_py_and_unlabelled_fences(self):
        for language in ("python", "py", ""):
            with self.subTest(language=language or "unlabelled"):
                code, has_fence = extract_code(fenced(REFERENCE_CODE, language))
                self.assertTrue(has_fence)
                self.assertIn("class Solution", code)

    def test_multiple_fences_prefers_solution_over_starter(self):
        response = """Starter:
```python
def helper():
    pass
```
Answer:
```python
class Solution:
    def solve(self, value):
        return value + 1
```
"""
        code, has_fence = extract_code(response)
        self.assertTrue(has_fence)
        self.assertTrue(code.startswith("class Solution"))
        self.assertNotIn("def helper", code)

    def test_think_and_plain_prose_fallbacks(self):
        responses = (
            "analysis that must be removed</think>\n" + REFERENCE_CODE,
            "Here is the final implementation:\n" + REFERENCE_CODE,
        )
        for response in responses:
            with self.subTest(response=response[:20]):
                code, has_fence = extract_code(response)
                self.assertFalse(has_fence)
                self.assertTrue(code.startswith("class Solution"))

    def test_empty_response(self):
        self.assertEqual(extract_code("   "), ("", False))


class StagedRewardTests(unittest.TestCase):
    def assert_stages(self, solution: str, expected):
        result = score(solution)
        observed = tuple(
            result[name]
            for name in (
                "format_score",
                "syntax_score",
                "compile_score",
                "runtime_success_score",
                "pass_score",
            )
        )
        self.assertEqual(observed, expected)
        return result

    def test_diverse_stage_outcomes(self):
        cases = {
            "empty": ("", (0.0, 0.0, 0.0, 0.0, 0.0)),
            "syntax_error": (
                "```python\nclass Solution\n    pass\n```",
                (1.0, 0.0, 0.0, 0.0, 0.0),
            ),
            "full_source_compile_error": (
                fenced(
                    "class Solution:\n"
                    "    def solve(self, value):\n"
                    "        return value + 1\n"
                    "return"
                ),
                (1.0, 1.0, 0.0, 0.0, 0.0),
            ),
            "runtime_exception": (
                fenced(
                    "class Solution:\n"
                    "    def solve(self, value):\n"
                    "        return 1 / 0"
                ),
                (1.0, 1.0, 1.0, 0.0, 0.0),
            ),
            "wrong_answer": (
                fenced(
                    "class Solution:\n"
                    "    def solve(self, value):\n"
                    "        return value"
                ),
                (1.0, 1.0, 1.0, 1.0, 0.0),
            ),
            "success": (
                fenced(REFERENCE_CODE),
                (1.0, 1.0, 1.0, 1.0, 1.0),
            ),
        }
        for name, (solution, expected) in cases.items():
            with self.subTest(name=name):
                self.assert_stages(solution, expected)

    def test_valid_unfenced_solution_passes_but_has_no_format_reward(self):
        result = self.assert_stages(REFERENCE_CODE, (0.0, 1.0, 1.0, 1.0, 1.0))
        self.assertEqual(result["format_reward"], 0.0)

    def test_fence_without_solution_class_has_no_format_reward(self):
        result = score(fenced("def solve(value):\n    return value + 1"))
        self.assertEqual(result["format_score"], 0.0)
        self.assertEqual(result["syntax_score"], 1.0)
        self.assertEqual(result["compile_score"], 1.0)
        self.assertEqual(result["runtime_error"], 1.0)

    def test_stdout_is_captured_and_does_not_break_worker_protocol(self):
        solution = fenced(
            "class Solution:\n"
            "    def solve(self, value):\n"
            "        print('candidate output')\n"
            "        return value + 1"
        )
        self.assertEqual(score(solution)["pass_score"], 1.0)

    def test_file_access_is_blocked_as_runtime_error(self):
        solution = fenced(
            "class Solution:\n"
            "    def solve(self, value):\n"
            "        open('/tmp/reward-test-must-not-exist', 'w')\n"
            "        return value + 1"
        )
        result = score(solution)
        self.assertEqual(result["compile_score"], 1.0)
        self.assertEqual(result["runtime_success_score"], 0.0)
        self.assertEqual(result["runtime_error"], 1.0)
        self.assertEqual(result["pass_score"], 0.0)

    def test_infinite_loop_times_out(self):
        solution = fenced(
            "class Solution:\n"
            "    def solve(self, value):\n"
            "        while True:\n"
            "            pass"
        )
        result = score(solution, timeout_seconds=0.15)
        self.assertEqual(result["timeout"], 1.0)
        self.assertEqual(result["runtime_error"], 0.0)
        self.assertEqual(result["execution_error"], 1.0)
        self.assertEqual(result["pass_score"], 0.0)

    def test_aliases_and_metadata_are_consistent(self):
        result = score(fenced(REFERENCE_CODE))
        aliases = {
            "format_score": "format_reward",
            "syntax_score": "syntax_reward",
            "compile_score": "compile_reward",
            "runtime_success_score": "runtime_reward",
            "ast_similarity_score": "ast_similarity_reward",
            "efficiency_score": "efficiency_reward",
            "pass_score": "accuracy_reward",
        }
        for score_name, reward_name in aliases.items():
            self.assertEqual(result[score_name], result[reward_name])
        self.assertEqual(result["num_tests"], 1.0)
        self.assertGreater(result["generated_runtime_seconds"], 0.0)


class AstRewardTests(unittest.TestCase):
    def test_identical_renamed_and_literal_changed_are_equivalent(self):
        variants = (
            REFERENCE_CODE,
            "class Solution:\n    def solve(self, x):\n        return x + 1\n",
            "class Solution:\n    def solve(self, value):\n        return value + 999\n",
        )
        for candidate in variants:
            with self.subTest(candidate=candidate.splitlines()[-1]):
                similarity, available = compute_ast_similarity(candidate, REFERENCE_CODE)
                self.assertEqual(similarity, 1.0)
                self.assertEqual(available, 1.0)

    def test_structurally_different_candidate_gets_partial_similarity(self):
        candidate = """class Solution:
    def solve(self, value):
        total = 0
        for item in range(value):
            if item % 2:
                total += item
        return total
"""
        similarity, available = compute_ast_similarity(candidate, REFERENCE_CODE)
        self.assertEqual(available, 1.0)
        self.assertGreater(similarity, 0.0)
        self.assertLess(similarity, 1.0)

    def test_invalid_candidate_keeps_reference_available(self):
        similarity, available = compute_ast_similarity("class Solution", REFERENCE_CODE)
        self.assertEqual((similarity, available), (0.0, 1.0))

    def test_empty_or_invalid_reference_disables_ast_reward(self):
        for reference in ("", "class Solution"):
            with self.subTest(reference=reference):
                self.assertEqual(compute_ast_similarity(REFERENCE_CODE, reference), (0.0, 0.0))

    def test_fenced_reference_and_clean_reference_with_import(self):
        fenced_similarity = compute_ast_similarity(REFERENCE_CODE, fenced(REFERENCE_CODE))
        self.assertEqual(fenced_similarity, (1.0, 1.0))

        with_import = "from typing import Any\n\n" + REFERENCE_CODE
        similarity, available = compute_ast_similarity(with_import, with_import)
        self.assertEqual((similarity, available), (1.0, 1.0))


class EfficiencyRewardTests(unittest.TestCase):
    def test_formula_and_boundary_cases(self):
        cases = {
            "faster": ((0.25, 1.0, 1.0), (0.75, 1.0, 1.0)),
            "equal": ((1.0, 1.0, 1.0), (0.0, 1.0, 1.0)),
            "slower": ((2.0, 1.0, 1.0), (0.0, 1.0, 1.0)),
            "failed_sample": ((0.25, 1.0, 0.0), (0.0, 0.0, 1.0)),
            "non_exact_pass": ((0.25, 1.0, 0.999), (0.0, 0.0, 1.0)),
            "missing_generated": ((0.0, 1.0, 1.0), (0.0, 0.0, 1.0)),
            "missing_reference": ((0.25, 0.0, 1.0), (0.0, 0.0, 0.0)),
            "negative_reference": ((0.25, -1.0, 1.0), (0.0, 0.0, 0.0)),
            "nan_generated": ((math.nan, 1.0, 1.0), (0.0, 0.0, 1.0)),
            "infinite_reference": ((0.25, math.inf, 1.0), (0.0, 0.0, 0.0)),
        }
        for name, (inputs, expected) in cases.items():
            with self.subTest(name=name):
                self.assertEqual(compute_efficiency_score(*inputs), expected)

    def test_integration_uses_measured_runtime_and_exact_formula(self):
        verifier = {**BASE_VERIFIER, "reference_runtime_seconds": 100.0}
        result = score(fenced(REFERENCE_CODE), verifier)
        expected = max(
            0.0,
            min(
                1.0,
                1.0
                - result["generated_runtime_seconds"]
                / result["reference_runtime_seconds"],
            ),
        )
        self.assertEqual(result["pass_score"], 1.0)
        self.assertEqual(result["efficiency_applied"], 1.0)
        self.assertEqual(result["efficiency_reference_available"], 1.0)
        self.assertAlmostEqual(result["efficiency_score"], expected, places=12)

    def test_wrong_answer_never_uses_efficiency(self):
        verifier = {**BASE_VERIFIER, "reference_runtime_seconds": 100.0}
        wrong = fenced(
            "class Solution:\n"
            "    def solve(self, value):\n"
            "        return value"
        )
        result = score(wrong, verifier)
        self.assertEqual(result["pass_score"], 0.0)
        self.assertEqual(result["efficiency_reference_available"], 1.0)
        self.assertEqual(result["efficiency_applied"], 0.0)
        self.assertEqual(result["efficiency_score"], 0.0)

    def test_missing_or_invalid_reference_runtime_disables_efficiency(self):
        for runtime in (None, "not-a-number", 0.0, -1.0):
            verifier = {**BASE_VERIFIER, "reference_runtime_seconds": runtime}
            with self.subTest(runtime=runtime):
                result = score(fenced(REFERENCE_CODE), verifier)
                self.assertEqual(result["efficiency_applied"], 0.0)
                self.assertEqual(result["efficiency_score"], 0.0)


class ScoreCompositionTests(unittest.TestCase):
    def test_manual_weighted_score_for_wrong_answer(self):
        wrong = fenced(
            "class Solution:\n"
            "    def solve(self, value):\n"
            "        return value"
        )
        result = score(
            wrong,
            format_weight=1.0,
            syntax_weight=2.0,
            compile_weight=3.0,
            runtime_weight=4.0,
            ast_similarity_weight=0.0,
            efficiency_weight=0.0,
            correctness_weight=5.0,
        )
        self.assertAlmostEqual(result["score"], 10.0 / 15.0)

    def test_correctness_only_weights(self):
        weights = {
            "format_weight": 0.0,
            "syntax_weight": 0.0,
            "compile_weight": 0.0,
            "runtime_weight": 0.0,
            "ast_similarity_weight": 0.0,
            "efficiency_weight": 0.0,
            "correctness_weight": 1.0,
        }
        self.assertEqual(score(fenced(REFERENCE_CODE), **weights)["score"], 1.0)
        self.assertEqual(score("", **weights)["score"], 0.0)

    def test_inactive_only_weight_returns_zero_without_division_error(self):
        result = score(
            "",
            format_weight=0.0,
            syntax_weight=0.0,
            compile_weight=0.0,
            runtime_weight=0.0,
            ast_similarity_weight=0.0,
            efficiency_weight=1.0,
            correctness_weight=0.0,
        )
        self.assertEqual(result["score"], 0.0)

    def test_invalid_weight_configurations_raise(self):
        with self.assertRaises(ValueError):
            score("", format_weight=-1.0)
        with self.assertRaises(ValueError):
            score(
                "",
                format_weight=0.0,
                syntax_weight=0.0,
                compile_weight=0.0,
                runtime_weight=0.0,
                ast_similarity_weight=0.0,
                efficiency_weight=0.0,
                correctness_weight=0.0,
            )


class VerifierInputTests(unittest.TestCase):
    def test_dict_json_string_and_bytes_ground_truth(self):
        encodings = (
            BASE_VERIFIER,
            json.dumps(BASE_VERIFIER),
            json.dumps(BASE_VERIFIER).encode("utf-8"),
        )
        for verifier in encodings:
            with self.subTest(type=type(verifier).__name__):
                self.assertEqual(score(fenced(REFERENCE_CODE), verifier)["pass_score"], 1.0)

    def test_missing_fields_or_unsupported_type_fail_at_compile_stage(self):
        invalid_verifiers = (
            {"prompt": "# missing test and entry point"},
            ["not", "a", "mapping"],
            b"not-json",
        )
        for verifier in invalid_verifiers:
            with self.subTest(type=type(verifier).__name__):
                result = score(fenced(REFERENCE_CODE), verifier)
                self.assertEqual(result["syntax_score"], 1.0)
                self.assertEqual(result["compile_score"], 0.0)
                self.assertEqual(result["pass_score"], 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
