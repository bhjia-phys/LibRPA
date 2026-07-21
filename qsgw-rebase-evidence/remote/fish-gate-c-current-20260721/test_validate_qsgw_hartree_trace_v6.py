#!/usr/bin/env python3

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from test_validate_qsgw_hartree_dump_v2 import CONTRACT, _FixedDirectory, _write_fixture
from validate_qsgw_hartree_trace_v6 import (
    HartreeTraceValidationError,
    validate_hartree_trace,
)


def _summary(electron_counts=(1.0, 1.0, 1.0)) -> str:
    rows = []
    for iteration, electron_count in enumerate(electron_counts):
        rows.append(
            f"{iteration} 0 0 0 0 0 {electron_count:.17e} "
            "0 0 0.2 0 1 0 0 0 none none"
        )
    return (
        CONTRACT
        + "# iter max_delta_eV residual_l2_Ha residual_max_Ha efermi_eV "
        "gap_eV electron_count requested_mode applied_mode beta fallback rcond "
        "coefficient_l1 coefficient_count converged coefficients fallback_reason\n"
        + "\n".join(rows)
        + "\n"
    )


def _replace_component_value(
    text: str, iteration: int, component: str, value: float
) -> str:
    lines = []
    matched = False
    for line in text.splitlines():
        fields = line.split()
        if (
            len(fields) == 11
            and fields[0] == str(iteration)
            and fields[2] == component
        ):
            fields[9] = f"{value:.17e}"
            fields[10] = "0.00000000000000000e+00"
            line = " ".join(fields)
            matched = True
        lines.append(line)
    if not matched:
        raise AssertionError(f"component {component} was not found")
    return "\n".join(lines) + "\n"


class HartreeTraceValidatorTests(unittest.TestCase):
    def fixture_root(self):
        fixed_root = os.environ.get("LIBRPA_QSGW_HARTREE_TRACE_TEST_TMP")
        if fixed_root:
            root = Path(fixed_root)
            if not root.is_dir():
                raise RuntimeError(
                    "LIBRPA_QSGW_HARTREE_TRACE_TEST_TMP must already exist"
                )
            return _FixedDirectory(root)
        return tempfile.TemporaryDirectory(ignore_cleanup_errors=True)

    def matrix_text(self) -> str:
        with self.fixture_root() as temporary:
            paths = _write_fixture(Path(temporary))
            return paths["matrix"].read_text(encoding="utf-8")

    def test_two_round_grid_hartree_trace_passes(self) -> None:
        report = validate_hartree_trace(self.matrix_text(), _summary())
        self.assertTrue(report["passed"])
        self.assertEqual(report["delta_vh_max_abs_ha_by_iteration"][1], 0.0)
        self.assertEqual(report["delta_vh_max_abs_ha_by_iteration"][2], 2.0)

    def test_raw_h_must_include_delta_vh(self) -> None:
        matrix = _replace_component_value(self.matrix_text(), 2, "raw_h", 2.1)
        report = validate_hartree_trace(matrix, _summary())
        self.assertFalse(report["passed"])
        self.assertFalse(report["closure_passed"])

    def test_summary_and_occupation_electron_counts_must_agree(self) -> None:
        report = validate_hartree_trace(
            self.matrix_text(), _summary((1.0, 1.0, 1.1))
        )
        self.assertFalse(report["passed"])
        self.assertFalse(report["charge_passed"])

    def test_first_hartree_update_must_be_zero(self) -> None:
        matrix = _replace_component_value(self.matrix_text(), 1, "delta_vh", 0.1)
        matrix = _replace_component_value(matrix, 1, "raw_h", 0.1)
        report = validate_hartree_trace(matrix, _summary())
        self.assertFalse(report["passed"])
        self.assertTrue(report["closure_passed"])
        self.assertFalse(report["response_passed"])

    def test_band_contract_is_rejected_by_grid_only_validator(self) -> None:
        matrix = self.matrix_text().replace(
            "# band disabled_stage1\n# h_qsgw_cut disabled_non_band\n",
            "# band fixed_reference_operator_fourier_live\n"
            "# h_qsgw_cut band_postprocess\n"
            "# qsgw_band0_unoccupied_keep 10\n"
            "# qsgw_band0_cut_mode 0\n"
            "# qsgw_band0_cut_shift_ha 20\n",
        )
        summary = _summary().replace(
            "# band disabled_stage1\n# h_qsgw_cut disabled_non_band\n",
            "# band fixed_reference_operator_fourier_live\n"
            "# h_qsgw_cut band_postprocess\n"
            "# qsgw_band0_unoccupied_keep 10\n"
            "# qsgw_band0_cut_mode 0\n"
            "# qsgw_band0_cut_shift_ha 20\n",
        )
        with self.assertRaisesRegex(
            HartreeTraceValidationError, "grid-only Hartree closure"
        ):
            validate_hartree_trace(matrix, summary)


if __name__ == "__main__":
    unittest.main()
