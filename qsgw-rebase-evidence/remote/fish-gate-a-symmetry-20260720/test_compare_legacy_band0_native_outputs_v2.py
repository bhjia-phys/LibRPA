#!/usr/bin/env python3
from __future__ import annotations

import unittest

from compare_legacy_band0_native_outputs_v2 import add_acceptance_semantics


def make_report(
    *,
    h0_max_abs: float = 0.0,
    h0_relative: float = 0.0,
    oracle_hermiticity: float = 0.0,
    reproduced_hermiticity: float = 0.0,
    sigcrf_passed: bool = True,
    text_passed: bool = True,
) -> dict[str, object]:
    return {
        "schema": "v1",
        "h0": {
            "files": [
                {
                    "oracle_hermiticity_max_abs_ha": oracle_hermiticity,
                    "reproduced_hermiticity_max_abs_ha": reproduced_hermiticity,
                }
            ],
            "max_abs_ha": h0_max_abs,
            "max_relative_frobenius": h0_relative,
            "passed": False,
        },
        "sigcrf": {"passed": sigcrf_passed},
        "text_outputs": {"passed": text_passed},
        "passed": False,
    }


class AcceptanceSemanticsTests(unittest.TestCase):
    def test_legacy_invariant_gap_does_not_reject_exact_reproduction(self) -> None:
        report = add_acceptance_semantics(
            make_report(
                h0_max_abs=0.0,
                h0_relative=0.0,
                oracle_hermiticity=8.0e-8,
                reproduced_hermiticity=8.0e-8,
            )
        )
        self.assertTrue(report["historical_reproduction_passed"])
        self.assertTrue(report["passed"])
        self.assertFalse(report["absolute_invariants_passed"])
        self.assertFalse(report["goal_thresholds_passed"])
        self.assertTrue(report["legacy_oracle_invariant_gap"])

    def test_parity_failure_is_still_rejected(self) -> None:
        report = add_acceptance_semantics(make_report(h0_max_abs=2.0e-6))
        self.assertFalse(report["historical_reproduction_passed"])
        self.assertFalse(report["passed"])

    def test_all_goal_thresholds_can_pass(self) -> None:
        report = add_acceptance_semantics(
            make_report(
                h0_max_abs=5.0e-7,
                h0_relative=5.0e-9,
                oracle_hermiticity=5.0e-11,
                reproduced_hermiticity=6.0e-11,
            )
        )
        self.assertTrue(report["historical_reproduction_passed"])
        self.assertTrue(report["absolute_invariants_passed"])
        self.assertTrue(report["goal_thresholds_passed"])
        self.assertFalse(report["legacy_oracle_invariant_gap"])


if __name__ == "__main__":
    unittest.main()
