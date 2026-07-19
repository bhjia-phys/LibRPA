#!/usr/bin/env python3
"""Tests for the qsgw_band iteration-zero initial-state observer.

Synthetic 2 k-point x 3 band x 1 spin dataset: the channel-1 iteration-0
eigenvalue trace must reproduce the ``band_KS_eigenvalue_k_NNNNN.txt``
inputs key-for-key, and occupations (trace-provided or the per-k-point
weight totals) must be consistent within tolerance.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent
MODULE_PATH = HERE / "validate_qsgw_band_initial_state_v1.py"

HA_TO_EV = 27.211386245988

K1_HA = (0.1, 0.25, 0.4)
K2_HA = (0.15, 0.3, 0.45)
K1_WEIGHTS = (0.9, 0.09, 0.01)
K2_WEIGHTS = (0.95, 0.04, 0.01)


def trace_text(*, perturb_ha=None, drop=(), duplicate=None):
    lines = ["# iter channel spin kpoint kx ky kz band energy_eV"]
    for kpoint, bands in enumerate((K1_HA, K2_HA)):
        for band, eigenvalue_ha in enumerate(bands):
            if (kpoint, band) in drop:
                continue
            shift = (perturb_ha or {}).get((kpoint, band), 0.0)
            row = (
                f"0 1 0 {kpoint} 0.0 0.0 0.0 {band} "
                f"{(eigenvalue_ha + shift) * HA_TO_EV:.17e}"
            )
            lines.append(row)
            if duplicate == (kpoint, band):
                lines.append(row)
    lines.append("0 0 0 0 0.0 0.0 0.0 0 1.00000000000000000e+00")
    lines.append("1 1 0 0 0.0 0.0 0.0 0 2.00000000000000000e+00")
    return "\n".join(lines) + "\n"


def band_file_text(eigenvalues_ha, weights):
    rows = []
    for band, (eigenvalue_ha, weight) in enumerate(
        zip(eigenvalues_ha, weights), 1
    ):
        rows.append(
            f"1 {band} {weight:.17e} {eigenvalue_ha:.17e} "
            f"{eigenvalue_ha * HA_TO_EV:.17e}"
        )
    return "\n".join(rows) + "\n"


def occupation_trace_text(*, perturb=None):
    lines = ["# iter channel spin kpoint kx ky kz band occupation"]
    for kpoint, weights in enumerate((K1_WEIGHTS, K2_WEIGHTS)):
        for band, weight in enumerate(weights):
            value = weight + (perturb or {}).get((kpoint, band), 0.0)
            lines.append(f"0 1 0 {kpoint} 0.0 0.0 0.0 {band} {value:.17e}")
    return "\n".join(lines) + "\n"


def write_case(root, *, trace=None, k2_weights=K2_WEIGHTS, k2_present=True):
    trace_path = root / "qsgw_eigenvalues.dat"
    trace_path.write_text(
        trace if trace is not None else trace_text(), encoding="utf-8"
    )
    band_dir = root / "band"
    band_dir.mkdir()
    (band_dir / "band_KS_eigenvalue_k_00001.txt").write_text(
        band_file_text(K1_HA, K1_WEIGHTS), encoding="utf-8"
    )
    if k2_present:
        (band_dir / "band_KS_eigenvalue_k_00002.txt").write_text(
            band_file_text(K2_HA, k2_weights), encoding="utf-8"
        )
    return trace_path, band_dir


class BandInitialStateValidationTest(unittest.TestCase):
    def run_validator(self, root, trace_path, band_dir, *extra_args):
        output = root / "report.json"
        completed = subprocess.run(
            [
                sys.executable,
                "-B",
                str(MODULE_PATH),
                str(trace_path),
                str(band_dir),
                "2",
                str(output),
                *extra_args,
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        report = (
            json.loads(output.read_text(encoding="utf-8"))
            if output.exists()
            else None
        )
        return completed, report

    def test_identical_initial_state_passes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            trace_path, band_dir = write_case(root)
            completed, report = self.run_validator(root, trace_path, band_dir)
            self.assertIsNotNone(report, completed.stderr)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertTrue(report["passed"])
            self.assertEqual(report["key_count"], 6)
            self.assertLess(report["eigenvalue_max_abs_diff_ha"], 1.0e-12)
            self.assertLess(report["occupation_max_abs_diff"], 1.0e-12)

    def test_eigenvalue_shift_below_tolerance_passes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            trace_path, band_dir = write_case(
                root, trace=trace_text(perturb_ha={(1, 2): 1.0e-13})
            )
            completed, report = self.run_validator(root, trace_path, band_dir)
            self.assertIsNotNone(report, completed.stderr)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertTrue(report["passed"])
            self.assertGreater(
                report["eigenvalue_max_abs_diff_ha"], 5.0e-14
            )
            self.assertLessEqual(
                report["eigenvalue_max_abs_diff_ha"], 1.0e-12
            )

    def test_eigenvalue_shift_above_tolerance_fails(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            trace_path, band_dir = write_case(
                root, trace=trace_text(perturb_ha={(1, 2): 1.0e-11})
            )
            completed, report = self.run_validator(root, trace_path, band_dir)
            self.assertIsNotNone(report, completed.stderr)
            self.assertEqual(completed.returncode, 2, completed.stderr)
            self.assertFalse(report["passed"])
            self.assertGreater(
                report["eigenvalue_max_abs_diff_ha"], 1.0e-12
            )
            self.assertEqual(report["worst"][0]["check"], "eigenvalue")
            self.assertEqual(report["worst"][0]["kpoint"], 1)
            self.assertEqual(report["worst"][0]["band"], 2)

    def test_missing_trace_key_fails_key_set(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            trace_path, band_dir = write_case(
                root, trace=trace_text(drop={(0, 1)})
            )
            completed, report = self.run_validator(root, trace_path, band_dir)
            self.assertIsNotNone(report, completed.stderr)
            self.assertEqual(completed.returncode, 2, completed.stderr)
            self.assertFalse(report["passed"])
            self.assertEqual(report["key_count"], 5)
            self.assertEqual(
                report["key_mismatch"]["missing_in_trace"], [[0, 0, 1]]
            )

    def test_duplicate_trace_key_is_a_format_error(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            trace_path, band_dir = write_case(
                root, trace=trace_text(duplicate=(0, 0))
            )
            completed, report = self.run_validator(root, trace_path, band_dir)
            self.assertIsNotNone(report, completed.stderr)
            self.assertEqual(completed.returncode, 1, completed.stderr)
            self.assertFalse(report["passed"])
            self.assertIn("duplicate", report["error"])

    def test_inconsistent_occupation_totals_fail(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            trace_path, band_dir = write_case(
                root, k2_weights=(0.90, 0.04, 0.01)
            )
            completed, report = self.run_validator(root, trace_path, band_dir)
            self.assertIsNotNone(report, completed.stderr)
            self.assertEqual(completed.returncode, 2, completed.stderr)
            self.assertFalse(report["passed"])
            self.assertGreater(report["occupation_max_abs_diff"], 1.0e-12)

    def test_per_state_convention_passes_with_flag(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            per_state_k1 = tuple(2.0 * w for w in K1_WEIGHTS)
            per_state_k2 = tuple(2.0 * w for w in K2_WEIGHTS)
            trace_path, band_dir = write_case(root)
            (band_dir / "band_KS_eigenvalue_k_00001.txt").write_text(
                band_file_text(K1_HA, per_state_k1), encoding="utf-8"
            )
            (band_dir / "band_KS_eigenvalue_k_00002.txt").write_text(
                band_file_text(K2_HA, per_state_k2), encoding="utf-8"
            )
            occupation_path = root / "qsgw_occupations.dat"
            occupation_path.write_text(
                occupation_trace_text(), encoding="utf-8"
            )
            completed, report = self.run_validator(
                root,
                trace_path,
                band_dir,
                "--occupation-trace",
                str(occupation_path),
                "--input-occupation-convention",
                "per-state",
            )
            self.assertIsNotNone(report, completed.stderr)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertTrue(report["passed"])
            self.assertEqual(
                report["input_occupation_convention"], "per-state"
            )
            self.assertLessEqual(
                report["occupation_max_abs_diff"], 1.0e-12
            )

    def test_per_state_convention_fails_without_flag(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            per_state_k1 = tuple(2.0 * w for w in K1_WEIGHTS)
            per_state_k2 = tuple(2.0 * w for w in K2_WEIGHTS)
            trace_path, band_dir = write_case(root)
            (band_dir / "band_KS_eigenvalue_k_00001.txt").write_text(
                band_file_text(K1_HA, per_state_k1), encoding="utf-8"
            )
            (band_dir / "band_KS_eigenvalue_k_00002.txt").write_text(
                band_file_text(K2_HA, per_state_k2), encoding="utf-8"
            )
            occupation_path = root / "qsgw_occupations.dat"
            occupation_path.write_text(
                occupation_trace_text(), encoding="utf-8"
            )
            completed, report = self.run_validator(
                root,
                trace_path,
                band_dir,
                "--occupation-trace",
                str(occupation_path),
            )
            self.assertIsNotNone(report, completed.stderr)
            self.assertEqual(completed.returncode, 2, completed.stderr)
            self.assertFalse(report["passed"])

    def test_trace_occupation_within_tolerance_passes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            trace_path, band_dir = write_case(root)
            occupation_path = root / "qsgw_occupations.dat"
            occupation_path.write_text(
                occupation_trace_text(), encoding="utf-8"
            )
            completed, report = self.run_validator(
                root,
                trace_path,
                band_dir,
                "--occupation-trace",
                str(occupation_path),
            )
            self.assertIsNotNone(report, completed.stderr)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertTrue(report["passed"])
            self.assertEqual(report["occupation_check"], "trace_occupation")
            self.assertLessEqual(
                report["occupation_max_abs_diff"], 1.0e-12
            )

    def test_trace_occupation_mismatch_fails(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            trace_path, band_dir = write_case(root)
            occupation_path = root / "qsgw_occupations.dat"
            occupation_path.write_text(
                occupation_trace_text(perturb={(1, 0): 1.0e-6}),
                encoding="utf-8",
            )
            completed, report = self.run_validator(
                root,
                trace_path,
                band_dir,
                "--occupation-trace",
                str(occupation_path),
            )
            self.assertIsNotNone(report, completed.stderr)
            self.assertEqual(completed.returncode, 2, completed.stderr)
            self.assertFalse(report["passed"])
            self.assertGreater(report["occupation_max_abs_diff"], 1.0e-12)

    def test_missing_band_file_is_a_format_error(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            trace_path, band_dir = write_case(root, k2_present=False)
            completed, report = self.run_validator(root, trace_path, band_dir)
            self.assertIsNotNone(report, completed.stderr)
            self.assertEqual(completed.returncode, 1, completed.stderr)
            self.assertFalse(report["passed"])
            self.assertIn("missing", report["error"])


if __name__ == "__main__":
    unittest.main()
