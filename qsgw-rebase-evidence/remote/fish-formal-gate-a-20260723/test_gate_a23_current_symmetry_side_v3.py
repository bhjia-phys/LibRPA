#!/usr/bin/env python3
"""Regression tests for the provenance-complete symmetry runner v3."""

from __future__ import annotations

import unittest
from pathlib import Path


class CurrentSymmetrySideV3RunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = (
            Path(__file__)
            .with_name("gate-a23-current-symmetry-side-v3.sh")
            .read_text(encoding="utf-8")
        )

    def test_binds_provenance_complete_bundle_v2(self) -> None:
        self.assertIn(
            "symmetry-physical-bundle-20260723-v2", self.source
        )
        for digest in (
            "9d2c7e63572e26ee2bb59d72c145da2c6ab9babf2c6e2c3bf352de8be5abab8d",
            "a2fe0a055c7da236c5a17ad2a9c5769d7e2c549c08d40a260a2166b8d225963b",
            "f8f97072567fbb819c406c0d7776d0209486c245b4e0140bdff7da34508b6c79",
            "70f69e02fb749229448b6139bc01f944789fbc21c23d10e10b7a7b56d952f340",
        ):
            self.assertIn(digest, self.source)

    def test_checks_complete_bundle_provenance_contract(self) -> None:
        for exact in (
            "grid=4x4x4",
            "use_shrink_abfs=false",
            "physical_lattice_source=matching_abacus_input_STRU",
            "shared_gw_source_changes=none",
            "symmetry=on",
            "headwing=off",
            "hartree=off",
            "band_update=off",
        ):
            self.assertIn(exact, self.source)

    def test_retains_both_iteration_modes_and_all_observers(self) -> None:
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

    def test_is_versioned_and_does_not_overclaim(self) -> None:
        self.assertIn("gate-a23-current-symmetry-side-v3.sh", self.source)
        self.assertIn("gate=current_symmetry_side_v3", self.source)
        self.assertIn("acceptance=false_pending_full_bz_control", self.source)
        self.assertNotIn("A2_ACCEPTANCE", self.source)
        self.assertNotIn("A3_ACCEPTANCE", self.source)
        self.assertNotIn("exact847", self.source.lower())
        self.assertNotIn("legacy", self.source.lower())


if __name__ == "__main__":
    unittest.main()
