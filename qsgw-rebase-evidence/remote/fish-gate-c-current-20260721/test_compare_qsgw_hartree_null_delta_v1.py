#!/usr/bin/env python3

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from compare_qsgw_hartree_null_delta_v1 import (
    HartreeNullDeltaComparisonError,
    compare_hartree_null_delta,
)
from test_validate_qsgw_hartree_dump_v2 import _FixedDirectory, _write_fixture
from test_validate_qsgw_hartree_trace_v6 import _summary


def _disabled_contract(text: str) -> str:
    return (
        text.replace("# hartree delta_density\n", "# hartree disabled_stage1\n")
        .replace("# hartree_coulomb full\n", "")
        .replace("# hartree_normalization weighted_occupations\n", "")
        .replace(
            "# qsgw_input_contract qsgw_input.hartree-full.contract\n",
            "# qsgw_input_contract qsgw_input.contract\n",
        )
        .replace(f"# qsgw_input_contract_sha256 {'0' * 64}\n", f"# qsgw_input_contract_sha256 {'1' * 64}\n")
    )


def _through_iteration_one(text: str, *, omit_delta: bool = False) -> str:
    output: list[str] = []
    for line in text.splitlines():
        fields = line.split()
        if fields and fields[0].isdigit():
            if int(fields[0]) > 1:
                continue
            if omit_delta and len(fields) >= 3 and fields[2] == "delta_vh":
                continue
        output.append(line)
    return "\n".join(output) + "\n"


def _replace_matrix_value(
    text: str, iteration: int, component: str, value: float
) -> str:
    output: list[str] = []
    found = False
    for line in text.splitlines():
        fields = line.split()
        if (
            len(fields) == 11
            and fields[0] == str(iteration)
            and fields[2] == component
        ):
            fields[9] = f"{value:.17e}"
            line = " ".join(fields)
            found = True
        output.append(line)
    if not found:
        raise AssertionError("matrix row was not found")
    return "\n".join(output) + "\n"


class HartreeNullDeltaComparatorTests(unittest.TestCase):
    def fixture_root(self):
        fixed_root = os.environ.get("LIBRPA_QSGW_HARTREE_NULL_TEST_TMP")
        if fixed_root:
            root = Path(fixed_root)
            if not root.is_dir():
                raise RuntimeError(
                    "LIBRPA_QSGW_HARTREE_NULL_TEST_TMP must already exist"
                )
            return _FixedDirectory(root)
        return tempfile.TemporaryDirectory(ignore_cleanup_errors=True)

    def fixture(self) -> tuple[str, str, str, str, str, str]:
        with self.fixture_root() as temporary:
            paths = _write_fixture(Path(temporary))
            enabled_matrix = paths["matrix"].read_text(encoding="utf-8")
            enabled_eigenvalue = paths["eigenvalue"].read_text(encoding="utf-8")
        enabled_summary = _summary()
        disabled_matrix = _disabled_contract(
            _through_iteration_one(enabled_matrix, omit_delta=True)
        )
        disabled_eigenvalue = _disabled_contract(
            _through_iteration_one(enabled_eigenvalue)
        )
        disabled_summary = _disabled_contract(
            _through_iteration_one(enabled_summary)
        )
        return (
            enabled_matrix,
            enabled_eigenvalue,
            enabled_summary,
            disabled_matrix,
            disabled_eigenvalue,
            disabled_summary,
        )

    def test_zero_delta_and_identical_control_pass(self) -> None:
        report = compare_hartree_null_delta(*self.fixture())
        self.assertTrue(report["passed"])
        self.assertEqual(report["delta_vh"]["max_abs_ha"], 0.0)
        self.assertEqual(report["matrix"]["max_abs_ha"], 0.0)

    def test_nonzero_first_delta_fails(self) -> None:
        fixture = list(self.fixture())
        fixture[0] = _replace_matrix_value(fixture[0], 1, "delta_vh", 1.0e-4)
        report = compare_hartree_null_delta(*fixture)
        self.assertFalse(report["passed"])
        self.assertFalse(report["delta_vh"]["passed"])

    def test_setup_side_effect_in_exx_fails(self) -> None:
        fixture = list(self.fixture())
        fixture[3] = _replace_matrix_value(fixture[3], 1, "exx", 1.0e-4)
        report = compare_hartree_null_delta(*fixture)
        self.assertFalse(report["passed"])
        self.assertFalse(report["matrix"]["passed"])

    def test_non_hartree_contract_difference_is_rejected(self) -> None:
        fixture = list(self.fixture())
        for index in (3, 4, 5):
            fixture[index] = fixture[index].replace(
                "# symmetry exx_on_gw_on_rpa_on\n",
                "# symmetry exx_off_gw_off_rpa_off\n",
            )
        with self.assertRaisesRegex(
            HartreeNullDeltaComparisonError, "non-Hartree contracts differ"
        ):
            compare_hartree_null_delta(*fixture)


if __name__ == "__main__":
    unittest.main()
