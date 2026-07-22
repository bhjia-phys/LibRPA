#!/usr/bin/env python3
"""Regression tests for the physical structure-plus-BZ symmetry runner v4."""

from __future__ import annotations

import unittest
from pathlib import Path


class CurrentSymmetrySideV4RunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = (
            Path(__file__)
            .with_name("gate-a23-current-symmetry-side-v4.sh")
            .read_text(encoding="utf-8")
        )

    def test_binds_physical_structure_and_bz_bundle_v3(self) -> None:
        self.assertIn("symmetry-physical-bundle-20260723-v3", self.source)
        for digest in (
            "d0da1dd120ca834e3ed81a7dd5e93b7c10ace30c79d53eb6d2fa08a27acc40c7",
            "01104a58dbcbcae8129331f7afd72bf866cd6d07c64226de35c4b7682489f4ba",
            "23c28adf8163e11339c72c0352501d2a595edae89ed5e352e3173a675af7e164",
            "5ef0f130be39886a8a55ac72130ae9c090f897c0858753638efa2ab578e7d358",
        ):
            self.assertIn(digest, self.source)

    def test_requires_physical_structure_and_cartesian_kvector_provenance(self) -> None:
        self.assertIn(
            "physical_lattice_source=matching_abacus_input_STRU", self.source
        )
        self.assertIn(
            "cartesian_kvector_source=correct_fractional_bz_coordinates_times_physical_reciprocal_lattice",
            self.source,
        )
        self.assertIn("shared_gw_source_changes=none", self.source)

    def test_runs_required_modes_with_all_state_observers(self) -> None:
        for token in (
            "run_mode no-mix-miniter2 2 none",
            "run_mode linear-beta-0.2-miniter5 5 linear",
            "qsgw_mixing_beta = 0.2",
            "validate_qsgw_trace_closure.py",
            "validate_qsgw_fixed_basis.py",
            "validate_qsgw_initial_state.py",
            "current-electron-count.txt",
        ):
            self.assertIn(token, self.source)

    def test_is_versioned_and_pending_full_bz_comparison(self) -> None:
        self.assertIn("gate-a23-current-symmetry-side-v4.sh", self.source)
        self.assertIn("gate=current_symmetry_side_v4", self.source)
        self.assertIn("acceptance=false_pending_full_bz_control", self.source)
        self.assertNotIn("A2_ACCEPTANCE", self.source)
        self.assertNotIn("A3_ACCEPTANCE", self.source)
        self.assertNotIn("exact847", self.source.lower())
        self.assertNotIn("legacy", self.source.lower())


if __name__ == "__main__":
    unittest.main()
