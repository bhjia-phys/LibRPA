#!/usr/bin/env python3
"""Tests for the Gate 1 stru_out symmetry compatibility overlay."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import build_stru_symmetry_overlay_v1 as overlay


SOURCE = """\
1 0 0
0 1 0
0 0 1
1 0 0
0 1 0
0 0 1
2
0 0 0 1
0.25 0.25 0.25 1
1 1 1
0 0 0
"""

VALID_TAIL = """\
2 row
1 0 0 0 1 0 0 0 1 0 0 0
-1 0 0 0 -1 0 0 0 -1 0.25 0.25 0.25
"""


class OverlayTests(unittest.TestCase):
    def run_overlay(self, source: str, tail: str) -> tuple[str, dict[str, object]]:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_path = root / "source-stru_out"
            tail_path = root / "symmetry-tail.txt"
            output_path = root / "stru_out"
            report_path = root / "report.json"
            source_path.write_text(source, encoding="utf-8")
            tail_path.write_text(tail, encoding="utf-8")
            report = overlay.build_overlay(
                source_path=source_path,
                tail_path=tail_path,
                output_path=output_path,
                report_path=report_path,
                expected_grid=(1, 1, 1),
                n_scf_kpoints=1,
                metric_tolerance=1.0e-12,
                atom_tolerance=1.0e-12,
            )
            return output_path.read_text(encoding="utf-8"), report

    def test_appends_validated_symmetry_tail(self) -> None:
        rendered, report = self.run_overlay(SOURCE, VALID_TAIL)

        self.assertEqual(rendered, SOURCE + VALID_TAIL)
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(report["grid"], [1, 1, 1])
        self.assertEqual(report["n_symops"], 2)
        self.assertEqual(report["convention"], "row")
        self.assertEqual(report["identity_count"], 1)
        self.assertIs(report["source_prefix_byte_identical"], True)
        self.assertLessEqual(report["metric_max_abs"], 1.0e-12)
        self.assertLessEqual(report["atom_fractional_max_abs"], 1.0e-12)

    def test_rejects_operation_that_breaks_lattice_metric(self) -> None:
        bad_tail = VALID_TAIL.replace(
            "-1 0 0 0 -1 0 0 0 -1", "1 1 0 0 1 0 0 0 1"
        )

        with self.assertRaisesRegex(ValueError, "lattice metric"):
            self.run_overlay(SOURCE, bad_tail)

    def test_rejects_operation_that_breaks_atom_mapping(self) -> None:
        bad_tail = VALID_TAIL.replace("0.25 0.25 0.25", "0.125 0.125 0.125")

        with self.assertRaisesRegex(ValueError, "atom mapping"):
            self.run_overlay(SOURCE, bad_tail)

    def test_rejects_source_with_existing_symmetry_block(self) -> None:
        with self.assertRaisesRegex(ValueError, "already contains"):
            self.run_overlay(SOURCE + VALID_TAIL, VALID_TAIL)


if __name__ == "__main__":
    unittest.main()
