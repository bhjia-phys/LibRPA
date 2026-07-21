#!/usr/bin/env python3

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from compare_qsgw_grid_channels_v1 import (
    GridChannelComparisonError,
    compare_grid_channels,
)
from test_validate_qsgw_hartree_dump_v2 import _FixedDirectory, _write_fixture
from test_validate_qsgw_hartree_trace_v6 import _summary


def _band_contract(text: str) -> str:
    return text.replace(
        "# band disabled_stage1\n# h_qsgw_cut disabled_non_band\n",
        "# band fixed_reference_operator_fourier_live\n"
        "# h_qsgw_cut band_postprocess\n"
        "# qsgw_band0_unoccupied_keep 10\n"
        "# qsgw_band0_cut_mode 0\n"
        "# qsgw_band0_cut_shift_ha 20\n",
    )


def _row(iteration: int, component: str, value: float) -> str:
    return (
        f"{iteration} 1 {component} 0 0 -1 0.00000000000000000e+00 "
        f"0 0 {value:.17e} 0.00000000000000000e+00"
    )


def _band_matrix(grid_text: str) -> str:
    rows = [
        _row(0, "h0", 0.0),
        _row(0, "vxc_dft", 0.0),
        _row(0, "occupation", 1.0),
        _row(0, "wfc_spinor0", 1.0),
    ]
    diagnostics = (
        "basis_inverse_residual",
        "basis_condition_estimate",
        "fourier_orthogonality_residual",
        "source_roundtrip_relative_error",
        "target_hermiticity_error",
        "target_relative_hermiticity_error",
        "repaired_target_hermiticity_error",
    )
    for iteration, delta in ((1, 0.0), (2, 2.0)):
        rows.extend(
            [
                _row(iteration, "exx", 0.0),
                _row(iteration, "vc", 0.0),
                _row(iteration, "delta_vh", delta),
                _row(iteration, "raw_h", delta),
                _row(iteration, "mixed_h", delta),
                _row(iteration, "rotation_u", 1.0),
                _row(iteration, "occupation", 1.0),
                _row(iteration, "wfc_spinor0", 1.0),
            ]
        )
        rows.extend(
            _row(iteration, component, 1.0 if component == "basis_condition_estimate" else 0.0)
            for component in diagnostics
        )
    return _band_contract(grid_text) + "\n".join(rows) + "\n"


def _band_eigenvalues(grid_text: str) -> str:
    rows = "\n".join(
        f"{iteration} 1 0 0 0.0 0.0 0.0 0 {float(iteration):.17e}"
        for iteration in range(3)
    )
    return _band_contract(grid_text) + rows + "\n"


def _replace_matrix_value(
    text: str, channel: int, iteration: int, component: str, value: float
) -> str:
    output = []
    found = False
    for line in text.splitlines():
        fields = line.split()
        if (
            len(fields) == 11
            and fields[0] == str(iteration)
            and fields[1] == str(channel)
            and fields[2] == component
        ):
            fields[9] = f"{value:.17e}"
            line = " ".join(fields)
            found = True
        output.append(line)
    if not found:
        raise AssertionError("matrix row was not found")
    return "\n".join(output) + "\n"


class GridChannelComparatorTests(unittest.TestCase):
    def fixture_root(self):
        fixed_root = os.environ.get("LIBRPA_QSGW_GRID_COMPARE_TEST_TMP")
        if fixed_root:
            root = Path(fixed_root)
            if not root.is_dir():
                raise RuntimeError(
                    "LIBRPA_QSGW_GRID_COMPARE_TEST_TMP must already exist"
                )
            return _FixedDirectory(root)
        return tempfile.TemporaryDirectory(ignore_cleanup_errors=True)

    def fixture(self):
        with self.fixture_root() as temporary:
            paths = _write_fixture(Path(temporary))
            grid_matrix = paths["matrix"].read_text(encoding="utf-8")
            grid_eigen = paths["eigenvalue"].read_text(encoding="utf-8")
        grid_summary = _summary()
        return (
            grid_matrix,
            grid_eigen,
            grid_summary,
            _band_matrix(grid_matrix),
            _band_eigenvalues(grid_eigen),
            _band_contract(grid_summary),
        )

    def test_identical_grid_channels_pass(self) -> None:
        report = compare_grid_channels(*self.fixture())
        self.assertTrue(report["passed"])
        self.assertEqual(report["matrix"]["max_abs_ha"], 0.0)

    def test_band_only_channel_does_not_change_grid_equivalence(self) -> None:
        fixture = list(self.fixture())
        fixture[3] = _replace_matrix_value(fixture[3], 1, 2, "raw_h", 9.0)
        report = compare_grid_channels(*fixture)
        self.assertTrue(report["passed"])

    def test_grid_channel_matrix_difference_fails(self) -> None:
        fixture = list(self.fixture())
        fixture[3] = _replace_matrix_value(fixture[3], 0, 2, "vc", 0.1)
        report = compare_grid_channels(*fixture)
        self.assertFalse(report["passed"])
        self.assertFalse(report["matrix"]["passed"])

    def test_grid_relevant_contract_difference_is_rejected(self) -> None:
        fixture = list(self.fixture())
        for index in (3, 4, 5):
            fixture[index] = fixture[index].replace(
                "# hartree_normalization weighted_occupations\n",
                "# hartree_normalization legacy_extra_inverse_nk\n",
            )
        with self.assertRaisesRegex(
            GridChannelComparisonError, "grid-relevant QSGW contracts differ"
        ):
            compare_grid_channels(*fixture)


if __name__ == "__main__":
    unittest.main()
