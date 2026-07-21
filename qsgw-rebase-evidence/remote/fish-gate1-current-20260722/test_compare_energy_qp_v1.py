#!/usr/bin/env python3
"""Tests for the structured energy_qp comparator."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import compare_energy_qp_v1 as comparator


def energy_qp(last_value: str = "4.0000000000E-01") -> str:
    return f"""\
  state     occ_num        e_gs(Ha)        e_qp(Ha)
---------------------------------------------------------------------------------------------------
  K_point    1 :           0.0000          0.0000          0.0000
---------------------------------------------------------------------------------------------------
       1    2.0000   -2.0000000000E-01   -2.5000000000E-01
       2    0.0000    3.0000000000E-01    {last_value}
---------------------------------------------------------------------------------------------------
"""


class EnergyQpComparatorTests(unittest.TestCase):
    def compare(self, candidate_text: str, tolerance: float) -> dict[str, object]:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference = root / "reference"
            candidate = root / "candidate"
            reference.write_text(energy_qp(), encoding="utf-8")
            candidate.write_text(candidate_text, encoding="utf-8")
            return comparator.compare_files(reference, candidate, tolerance)

    def test_accepts_qp_roundoff_with_exact_input_columns(self) -> None:
        report = self.compare(energy_qp("4.0000000010E-01"), 2.0e-10)

        self.assertIs(report["passed"], True)
        self.assertEqual(report["kpoint_count"], 1)
        self.assertEqual(report["state_count"], 2)
        self.assertEqual(report["occupation_max_abs_difference"], 0.0)
        self.assertEqual(report["ks_energy_max_abs_difference_ha"], 0.0)
        self.assertAlmostEqual(report["qp_energy_max_abs_difference_ha"], 1.0e-10)

    def test_rejects_qp_difference_above_tolerance(self) -> None:
        report = self.compare(energy_qp("4.0000000100E-01"), 2.0e-10)

        self.assertIs(report["passed"], False)
        self.assertGreater(
            report["qp_energy_max_abs_difference_ha"],
            report["thresholds"]["qp_energy_max_abs_tolerance_ha"],
        )

    def test_rejects_changed_ks_energy(self) -> None:
        changed = energy_qp().replace("3.0000000000E-01", "3.0000000001E-01")
        report = self.compare(changed, 1.0e-9)

        self.assertIs(report["passed"], False)
        self.assertGreater(report["ks_energy_max_abs_difference_ha"], 0.0)

    def test_rejects_changed_state_key(self) -> None:
        changed = energy_qp().replace("       2    0.0000", "       3    0.0000")

        with self.assertRaisesRegex(ValueError, "state keys differ"):
            self.compare(changed, 1.0e-9)


if __name__ == "__main__":
    unittest.main()
