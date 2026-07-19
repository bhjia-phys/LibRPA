#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent
MODULE_PATH = HERE / "validate_qsgw_hartree_trace_v1.py"
SPEC = importlib.util.spec_from_file_location("hartree_trace", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def row(
    iteration: int,
    component: str,
    kpoint: int,
    column: int,
    value: float,
) -> str:
    return (
        f"{iteration} 0 {component} 0 {kpoint} -1 0.0 0 {column} "
        f"{value:.17e} 0.0"
    )


def matrix_trace(
    *, initial_delta: float = 0.0, charge_drift: float = 0.0,
    final_delta_shift: float = 0.0,
) -> str:
    rows = []
    occupations = {
        0: ((2.0, 0.0), (2.0, 0.0)),
        1: ((2.0, 0.0), (2.0, 0.0)),
        2: ((2.1 + charge_drift, 0.0), (1.9, 0.0)),
    }
    for iteration, values_by_kpoint in occupations.items():
        for kpoint, values in enumerate(values_by_kpoint):
            for column, value in enumerate(values):
                rows.append(row(iteration, "occupation", kpoint, column, value))
    for kpoint in range(2):
        rows.append(row(1, "delta_vh", kpoint, 0, initial_delta))
    rows.append(row(2, "delta_vh", 0, 0, 0.05 + final_delta_shift))
    rows.append(row(2, "delta_vh", 1, 0, -0.02))
    return "\n".join(rows) + "\n"


def iteration_trace(*, final_count: float = 2.0) -> str:
    return "\n".join(
        (
            "0 0 0 0 0 0 2.0",
            "1 0 0 0 0 0 2.0",
            f"2 0 0 0 0 0 {final_count:.17e}",
        )
    ) + "\n"


def bz_sampling() -> str:
    return "\n".join(
        (
            "1 1 2",
            "2 2",
            "1 0.5 0 0 0",
            "2 0.5 0 0 0",
        )
    ) + "\n"


class HartreeTraceValidationTest(unittest.TestCase):
    def compare(self, **overrides):
        arguments = {
            "old_matrix_text": matrix_trace(),
            "current_matrix_text": matrix_trace(),
            "current_iteration_text": iteration_trace(),
            "bz_sampling_text": bz_sampling(),
            "iterations": [0, 1, 2],
        }
        arguments.update(overrides)
        return MODULE.compare_hartree_traces(**arguments)

    def test_equivalent_charge_conserving_hartree_traces_pass(self):
        report = self.compare()
        self.assertTrue(report["passed"])
        self.assertEqual(report["max_charge_drift"], 0.0)
        self.assertEqual(report["current_initial_delta_vh_max_abs_ha"], 0.0)

    def test_nonzero_initial_delta_is_rejected(self):
        report = self.compare(current_matrix_text=matrix_trace(initial_delta=1.0e-8))
        self.assertFalse(report["passed"])
        self.assertGreater(
            report["current_initial_delta_vh_max_abs_ha"], 1.0e-10
        )

    def test_density_delta_charge_drift_is_rejected(self):
        report = self.compare(current_matrix_text=matrix_trace(charge_drift=1.0e-6))
        self.assertFalse(report["passed"])
        self.assertGreater(report["max_charge_drift"], 1.0e-10)

    def test_old_current_delta_vh_mismatch_is_rejected(self):
        report = self.compare(
            current_matrix_text=matrix_trace(final_delta_shift=1.0e-6)
        )
        self.assertFalse(report["passed"])
        self.assertGreater(
            report["delta_vh_max_abs_difference_ha"], 1.0e-8
        )

    def test_cli_writes_passing_report(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            old = root / "old.dat"
            current = root / "current.dat"
            iterations = root / "iterations.dat"
            sampling = root / "bz_sampling_out"
            output = root / "report.json"
            old.write_text(matrix_trace(), encoding="utf-8")
            current.write_text(matrix_trace(), encoding="utf-8")
            iterations.write_text(iteration_trace(), encoding="utf-8")
            sampling.write_text(bz_sampling(), encoding="utf-8")
            completed = subprocess.run(
                [
                    sys.executable,
                    "-B",
                    str(MODULE_PATH),
                    str(old),
                    str(current),
                    str(iterations),
                    str(sampling),
                    str(output),
                    "--iterations",
                    "0:2",
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertTrue(
                json.loads(output.read_text(encoding="utf-8"))["passed"]
            )


if __name__ == "__main__":
    unittest.main()
