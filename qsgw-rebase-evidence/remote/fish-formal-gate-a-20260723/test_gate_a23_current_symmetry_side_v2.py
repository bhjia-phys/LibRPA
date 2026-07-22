#!/usr/bin/env python3
"""Static contract tests for corrected current-QSGW symmetry production."""

from __future__ import annotations

import unittest
from pathlib import Path


class CurrentSymmetrySideV2RunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = (
            Path(__file__)
            .with_name("gate-a23-current-symmetry-side-v2.sh")
            .read_text(encoding="utf-8")
        )

    def test_binds_corrected_read_only_bundle_and_manifests(self) -> None:
        self.assertIn(
            "abacus-pinned-dd421665-si-k444-symmetry-physical-bundle-20260723-v1",
            self.source,
        )
        for digest in (
            "96ac96fdd196b2d6a53034724c4999ab31acaee3db621829ede836e3b7079eaf",
            "b044ca10b8c5516c3b3bea1f001d6ec1b721042fc000ae7ad4e455a55ad82be8",
            "f8f97072567fbb819c406c0d7776d0209486c245b4e0140bdff7da34508b6c79",
            "70f69e02fb749229448b6139bc01f944789fbc21c23d10e10b7a7b56d952f340",
            "545e595eab871410b4e06e46c6d4394d82f6d46fa46eb08e572a0546763a3ed4",
        ):
            self.assertIn(digest, self.source)
        self.assertIn('test -z "$(find "$bundle" -type l -print -quit)"', self.source)
        self.assertIn('test -z "$(find "$bundle" -perm /222 -print -quit)"', self.source)

    def test_requires_physical_lattice_provenance_and_zero_shared_gw_changes(self) -> None:
        self.assertIn(
            "physical_lattice_source=matching_abacus_input_STRU", self.source
        )
        self.assertIn("shared_gw_source_changes=none", self.source)
        self.assertNotIn("symmetry-bundle-20260720-v4\n", self.source)

    def test_runs_both_required_iteration_sequences(self) -> None:
        self.assertIn("run_mode no-mix-miniter2 2 none", self.source)
        self.assertIn("run_mode linear-beta-0.2-miniter5 5 linear", self.source)
        self.assertIn("qsgw_mixing_beta = 0.2", self.source)
        self.assertIn('"0:$target_iter"', self.source)
        self.assertIn("qsgw_write_iteration_matrices = true", self.source)

    def test_enables_all_symmetry_paths_and_keeps_later_stages_off(self) -> None:
        for token in (
            "use_abacus_exx_symmetry = true",
            "use_abacus_gw_symmetry = true",
            "use_symmetry_exx = true",
            "use_symmetry_gw = true",
            "use_symmetry_rpa = true",
            "qsgw_update_hartree = false",
            "qsgw_iterative_headwing = false",
        ):
            self.assertIn(token, self.source)
        self.assertNotIn("task = qsgw_band", self.source)

    def test_validates_closure_fixed_basis_initial_state_and_electron_count(self) -> None:
        for token in (
            "validate_qsgw_trace_closure.py",
            "validate_qsgw_fixed_basis.py",
            "validate_qsgw_initial_state.py",
            "current-electron-count.txt",
            "current-v6-to-v5-normalization.json",
        ):
            self.assertIn(token, self.source)

    def test_remains_pending_until_full_bz_comparison(self) -> None:
        self.assertIn("gate=current_symmetry_side_v2", self.source)
        self.assertIn("acceptance=false_pending_full_bz_control", self.source)
        self.assertIn("SYMMETRY_SIDE_COMPLETE", self.source)
        self.assertNotIn("A2_ACCEPTANCE", self.source)
        self.assertNotIn("A3_ACCEPTANCE", self.source)
        self.assertNotIn("exact847", self.source.lower())
        self.assertNotIn("legacy", self.source.lower())


if __name__ == "__main__":
    unittest.main()
