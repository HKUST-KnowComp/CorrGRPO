#!/usr/bin/env python3

import csv
import tempfile
import unittest
from pathlib import Path

from compute_asb_official_pna_asr_nrp import sample_level_nrp


FIELDS = [
    "Agent Name",
    "Attack Tool",
    "Attack Successful",
    "Original Task Successful",
]


def write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


class SampleLevelNrpTest(unittest.TestCase):
    def test_pairs_by_sample_key_before_averaging(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_rows(root / "asb_clean_combined_attack.csv", [
                {"Agent Name": "a", "Attack Tool": "t1", "Attack Successful": "0", "Original Task Successful": "1"},
                {"Agent Name": "a", "Attack Tool": "t2", "Attack Successful": "0", "Original Task Successful": "0"},
                {"Agent Name": "b", "Attack Tool": "t3", "Attack Successful": "0", "Original Task Successful": "1"},
                {"Agent Name": "b", "Attack Tool": "t4", "Attack Successful": "0", "Original Task Successful": "0"},
            ])
            # Deliberately reorder attack rows. Aggregate PNA*(1-ASR) would be
            # 37.5%, while the correctly paired sample-level mean is 25%.
            write_rows(root / "asb_indirect_prompt_injection_context_ignoring.csv", [
                {"Agent Name": "b", "Attack Tool": "t4", "Attack Successful": "0", "Original Task Successful": "0"},
                {"Agent Name": "a", "Attack Tool": "t1", "Attack Successful": "1", "Original Task Successful": "0"},
                {"Agent Name": "b", "Attack Tool": "t3", "Attack Successful": "0", "Original Task Successful": "0"},
                {"Agent Name": "a", "Attack Tool": "t2", "Attack Successful": "0", "Original Task Successful": "0"},
            ])

            model = {"result_dir": str(root)}
            value, pairing = sample_level_nrp(
                model,
                [("clean_combined_attack", {})],
                model,
                [("indirect_prompt_injection_context_ignoring", {})],
            )

            self.assertEqual(value, 25.0)
            self.assertEqual(pairing["status"], "complete")
            self.assertEqual(pairing["matched_samples"], 4)
            self.assertEqual(pairing["retained_samples"], 1)
            self.assertEqual(pairing["missing_clean_samples"], 0)


if __name__ == "__main__":
    unittest.main()
