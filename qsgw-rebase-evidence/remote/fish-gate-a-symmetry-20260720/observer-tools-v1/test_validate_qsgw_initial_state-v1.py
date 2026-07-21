#!/usr/bin/env python3
"""Focused tests for the QSGW initial-state observer."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from validate_qsgw_initial_state import HA2EV, validate


def _band_out(efermi: float = 0.25) -> str:
    return (
        "1\n1\n2\n2\n"
        f"{efermi}\n"
        "1 1\n"
        "1 2.0 -1.0 -27.211386245988\n"
        "2 0.0 1.0 27.211386245988\n"
    )


def _matrix_trace() -> str:
    return (
        "# header\n"
        "0 0 occupation 0 0 -1 0.0 0 0 2.0 0.0\n"
        "0 0 occupation 0 0 -1 0.0 0 1 0.0 0.0\n"
    )


def _summary(efermi: float = 0.25) -> str:
    fields = [
        "0", "0", "0", "0", f"{efermi * HA2EV:.17e}", "2", "2",
        "-1", "-1", "0.2", "0", "1", "0", "0", "0", "none", "none",
    ]
    return "# header\n" + " ".join(fields) + "\n"


class InitialStateTests(unittest.TestCase):
    def _validate(self, summary_efermi: float) -> dict[str, object]:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            matrix = root / "qsgw_matrices.dat"
            summary = root / "qsgw_iterations.dat"
            band = root / "band_out"
            matrix.write_text(_matrix_trace(), encoding="utf-8")
            summary.write_text(_summary(summary_efermi), encoding="utf-8")
            band.write_text(_band_out(), encoding="utf-8")
            return validate(matrix, summary, band, 1.0e-12, 1.0e-12)

    def test_preserved_initial_state_passes(self) -> None:
        report = self._validate(0.25)
        self.assertTrue(report["passed"])
        self.assertLess(report["efermi_max_abs_difference_ha"], 1.0e-15)
        self.assertEqual(report["occupation_max_abs_difference"], 0.0)

    def test_shifted_initial_fermi_is_rejected(self) -> None:
        report = self._validate(0.26)
        self.assertFalse(report["passed"])
        self.assertAlmostEqual(
            report["efermi_max_abs_difference_ha"], 0.01, places=14
        )


if __name__ == "__main__":
    unittest.main()
