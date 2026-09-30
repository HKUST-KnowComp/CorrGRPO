#!/usr/bin/env python3
"""Smoke test every staged reward outcome without loading a model."""

from __future__ import annotations

from reward import compute_ast_similarity, compute_efficiency_score, compute_score


VERIFIER = {
    "task_id": "reward-stage-smoke-test",
    "prompt": "# no helper imports required",
    "test": "def check(candidate):\n    assert candidate(1) == 2\n",
    "entry_point": "Solution().solve",
    "num_tests": 1,
}

CASES = {
    "syntax_error": (
        "```python\nclass Solution\n    pass\n```",
        (1.0, 0.0, 0.0, 0.0, 0.0),
    ),
    "compile_error": (
        "```python\nclass Solution:\n    pass\nreturn\n```",
        (1.0, 1.0, 0.0, 0.0, 0.0),
    ),
    "runtime_error": (
        "```python\nclass Solution:\n    def solve(self, value):\n        raise ValueError\n```",
        (1.0, 1.0, 1.0, 0.0, 0.0),
    ),
    "assertion_failure": (
        "```python\nclass Solution:\n    def solve(self, value):\n        return value\n```",
        (1.0, 1.0, 1.0, 1.0, 0.0),
    ),
    "success": (
        "```python\nclass Solution:\n    def solve(self, value):\n        return value + 1\n```",
        (1.0, 1.0, 1.0, 1.0, 1.0),
    ),
}

FIELDS = (
    "format_score",
    "syntax_score",
    "compile_score",
    "runtime_success_score",
    "pass_score",
)


def main() -> int:
    failures = 0
    for name, (solution, expected) in CASES.items():
        result = compute_score("leetcodedataset", solution, VERIFIER)
        observed = tuple(result[field] for field in FIELDS)
        passed = observed == expected
        failures += int(not passed)
        print(
            f"{name:18s} expected={expected} observed={observed} "
            f"weighted_score={result['score']:.2f} {'OK' if passed else 'FAIL'}"
        )

    reference = "class Solution:\n    def solve(self, value):\n        return value + 1\n"
    renamed = "class Solution:\n    def solve(self, x):\n        return x + 1\n"
    ast_cases = {
        "identical_ast": (reference, reference, 1.0, 1.0),
        "renamed_variables": (renamed, reference, 1.0, 1.0),
        "invalid_candidate": ("class Solution", reference, 0.0, 1.0),
        "missing_reference": (reference, "", 0.0, 0.0),
    }
    for name, (candidate, expected_reference, expected_similarity, expected_available) in ast_cases.items():
        similarity, available = compute_ast_similarity(candidate, expected_reference)
        passed = similarity == expected_similarity and available == expected_available
        failures += int(not passed)
        print(
            f"{name:18s} similarity={similarity:.4f} available={available:.0f} "
            f"{'OK' if passed else 'FAIL'}"
        )

    efficiency_cases = {
        "faster_pass": ((0.25, 1.0, 1.0), (0.75, 1.0, 1.0)),
        "same_speed_pass": ((1.0, 1.0, 1.0), (0.0, 1.0, 1.0)),
        "slower_pass": ((2.0, 1.0, 1.0), (0.0, 1.0, 1.0)),
        "faster_but_fail": ((0.25, 1.0, 0.0), (0.0, 0.0, 1.0)),
        "missing_runtime": ((0.25, 0.0, 1.0), (0.0, 0.0, 0.0)),
    }
    for name, (inputs, expected) in efficiency_cases.items():
        observed = compute_efficiency_score(*inputs)
        passed = observed == expected
        failures += int(not passed)
        print(
            f"{name:18s} expected={expected} observed={observed} "
            f"{'OK' if passed else 'FAIL'}"
        )

    timed_verifier = {**VERIFIER, "reference_runtime_seconds": 100.0}
    timed_result = compute_score("leetcodedataset", CASES["success"][0], timed_verifier)
    timed_passed = (
        timed_result["pass_score"] == 1.0
        and timed_result["generated_runtime_seconds"] > 0.0
        and timed_result["efficiency_applied"] == 1.0
        and 0.0 <= timed_result["efficiency_score"] <= 1.0
    )
    failures += int(not timed_passed)
    print(
        "timed_execution    "
        f"runtime={timed_result['generated_runtime_seconds']:.6f}s "
        f"efficiency={timed_result['efficiency_score']:.6f} "
        f"{'OK' if timed_passed else 'FAIL'}"
    )
    return int(failures > 0)


if __name__ == "__main__":
    raise SystemExit(main())
