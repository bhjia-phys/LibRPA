#!/usr/bin/env python3
"""Tests for matched-pair binding in symmetry runner v6."""

from __future__ import annotations

import unittest
from pathlib import Path


class CurrentSymmetrySideV6RunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = (
            Path(__file__)
            .with_name("gate-a23-current-symmetry-side-v6.sh")
            .read_text(encoding="utf-8")
        )

    def test_validates_printed_beta_numerically(self) -> None:
        self.assertIn('$2 == "qsgw_mixing_beta"', self.source)
        self.assertIn("delta = ($3 + 0.0) - 0.2", self.source)
        self.assertIn("delta > 1.0e-15", self.source)
        self.assertIn("count != 1", self.source)
        self.assertNotIn(
            "grep -Fqx '# qsgw_mixing_beta 0.2'", self.source
        )

    def test_binds_matched_physical_structure_and_bz_bundle(self) -> None:
        self.assertIn("pair-physical-bundles-20260723-v2", self.source)
        self.assertIn("expected_pair_output_sha", self.source)
        self.assertIn("expected_pair_validation_sha", self.source)
        self.assertIn('test -e "$pair_root/PAIR_COMPLETE"', self.source)
        self.assertIn(
            "physical_lattice_source=matching_abacus_input_STRU", self.source
        )
        self.assertIn(
            "cartesian_kvector_source=fractional_bz_times_physical_reciprocal_lattice",
            self.source,
        )

    def test_retains_required_modes_and_observers(self) -> None:
        for token in (
            "run_mode no-mix-miniter2 2 none",
            "run_mode linear-beta-0.2-miniter5 5 linear",
            "validate_qsgw_trace_closure.py",
            "validate_qsgw_fixed_basis.py",
            "validate_qsgw_initial_state.py",
            "current-electron-count.txt",
        ):
            self.assertIn(token, self.source)

    def test_is_versioned_and_does_not_claim_a2_or_a3(self) -> None:
        self.assertIn("gate-a23-current-symmetry-side-v6.sh", self.source)
        self.assertIn("gate=current_symmetry_side_v6", self.source)
        self.assertIn("acceptance=false_pending_full_bz_control", self.source)
        self.assertNotIn("A2_ACCEPTANCE", self.source)
        self.assertNotIn("A3_ACCEPTANCE", self.source)
        self.assertNotIn("exact847", self.source.lower())
        self.assertNotIn("legacy", self.source.lower())


if __name__ == "__main__":
    unittest.main()
