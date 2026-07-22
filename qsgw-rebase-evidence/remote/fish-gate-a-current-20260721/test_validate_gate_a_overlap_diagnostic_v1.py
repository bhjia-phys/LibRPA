#!/usr/bin/env python3

from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import validate_gate_a_overlap_diagnostic_v1 as validator


def accepted_report() -> dict[str, object]:
    components = {
        name: {"max_abs_diff": 1e-10}
        for name in validator.EXPECTED_COMPONENTS
    }
    return {
        "components": components,
        "contract": {
            "differences": {
                "qsgw_min_iter": {"old": 1, "new": 2},
                "qsgw_max_iter": {"old": 1, "new": 2},
            }
        },
        "state": {
            "max_eigenvalue_abs_diff_ha": 1e-10,
            "max_alignment_unitarity_residual": 1e-14,
            "max_rotation_relative_residual": 1e-13,
            "max_velocity_relative_residual": 0.0,
            "max_wfc_relative_residual": 1e-13,
        },
    }


class OverlapDiagnosticTests(unittest.TestCase):
    def test_accepts_equivalent_prefixes(self) -> None:
        result = validator.validate(accepted_report(), 2e-9)
        self.assertTrue(result["passed"])
        self.assertEqual(
            result["conclusion"],
            "overlap_alias_does_not_explain_rejected_prefix",
        )

    def test_rejects_large_component_difference(self) -> None:
        report = accepted_report()
        report["components"]["sigma_c_iw"]["max_abs_diff"] = 3e-9
        with self.assertRaisesRegex(validator.DiagnosticError, "sigma_c_iw"):
            validator.validate(report, 2e-9)

    def test_rejects_unexpected_contract_difference(self) -> None:
        report = accepted_report()
        report["contract"]["differences"]["use_shrink_abfs"] = {
            "old": 0,
            "new": 1,
        }
        with self.assertRaisesRegex(validator.DiagnosticError, "contract"):
            validator.validate(report, 2e-9)

    def test_rejects_state_rotation_change(self) -> None:
        report = copy.deepcopy(accepted_report())
        report["state"]["max_rotation_relative_residual"] = 2e-10
        with self.assertRaisesRegex(validator.DiagnosticError, "rotation"):
            validator.validate(report, 2e-9)


if __name__ == "__main__":
    unittest.main(verbosity=2)
