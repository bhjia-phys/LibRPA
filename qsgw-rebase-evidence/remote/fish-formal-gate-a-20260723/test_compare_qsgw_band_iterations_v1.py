#!/usr/bin/env python3

from __future__ import annotations

import argparse
import tempfile
import unittest
from pathlib import Path

from compare_qsgw_band_iterations_v1 import (
    ComparisonError,
    compare_tables,
    indirect_gap_ev,
    parse_band_table,
    parse_iteration_list,
    run,
)


def write_table(path: Path, shift_ev: float = 0.0) -> None:
    path.write_text(
        "1 0.0 0.0 0.0 2.0 -1.0 0.0 0.5 0.0 2.0\n"
        f"2 0.5 0.0 0.0 2.0 {-0.8 + shift_ev:.8f} "
        f"0.0 {0.4 + shift_ev:.8f} 0.0 {2.2 + shift_ev:.8f}\n",
        encoding="ascii",
    )


class BandIterationComparatorTest(unittest.TestCase):
    def test_parse_and_gap(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "band.dat"
            write_table(path)
            rows = parse_band_table(path)
            self.assertEqual(len(rows), 2)
            self.assertEqual(len(rows[0].energies_ev), 3)
            self.assertAlmostEqual(indirect_gap_ev(rows, 1), 1.2)

    def test_table_comparison_reports_energy_and_gap(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            legacy = root / "legacy.dat"
            candidate = root / "candidate.dat"
            write_table(legacy)
            write_table(candidate, shift_ev=5.0e-5)
            report = compare_tables(
                parse_band_table(legacy),
                parse_band_table(candidate),
                occupied_bands=1,
                coordinate_tolerance=1.0e-7,
                energy_tolerance_ev=1.0e-4,
                gap_tolerance_ev=2.0e-4,
            )
            self.assertTrue(report["passed"])
            self.assertAlmostEqual(report["energy_max_abs_ev"], 5.0e-5)
            self.assertAlmostEqual(report["gap_abs_difference_ev"], 0.0)

    def test_multiround_run_requires_every_iteration(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            legacy = root / "legacy"
            candidate = root / "candidate"
            legacy.mkdir()
            candidate.mkdir()
            for iteration in (1, 2):
                write_table(
                    legacy / f"QSGW_band_spin_1_{iteration}.dat"
                )
                write_table(
                    candidate / f"QSGW_band_spin_1_{iteration}.dat"
                )
            args = argparse.Namespace(
                legacy_dir=legacy,
                candidate_dir=candidate,
                iterations=(1, 2),
                spins=1,
                occupied_bands=1,
                coordinate_tolerance=1.0e-7,
                energy_tolerance_ev=1.0e-4,
                gap_tolerance_ev=2.0e-4,
            )
            report = run(args)
            self.assertTrue(report["passed"])
            self.assertEqual(report["iterations"], [1, 2])

            (candidate / "QSGW_band_spin_1_2.dat").unlink()
            with self.assertRaises(ComparisonError):
                run(args)

    def test_bad_iteration_list_is_rejected(self) -> None:
        self.assertEqual(parse_iteration_list("1,2,5"), (1, 2, 5))
        with self.assertRaises(argparse.ArgumentTypeError):
            parse_iteration_list("2,1")
        with self.assertRaises(argparse.ArgumentTypeError):
            parse_iteration_list("1,1")


if __name__ == "__main__":
    unittest.main()
