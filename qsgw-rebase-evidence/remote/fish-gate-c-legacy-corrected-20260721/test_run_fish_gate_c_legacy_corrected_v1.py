#!/usr/bin/env python3

from __future__ import annotations

import unittest
from pathlib import Path


RUNNER = Path(__file__).resolve().parent / "run_fish_gate_c_legacy_corrected_v1.sh"


class CorrectedLegacyParityRunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = RUNNER.read_text(encoding="utf-8")

    def test_binds_candidate_legacy_build_bundle_and_contract_hashes(self) -> None:
        required = (
            "CANDIDATE_GATE0_PROVENANCE_SHA256",
            "CANDIDATE_EXE_SHA256",
            "LEGACY_BUILD_PROVENANCE_SHA256",
            "LEGACY_BUILD_OUTPUT_SHA256",
            "LEGACY_EXE_SHA256",
            "FULLBZ_BUNDLE_PROVENANCE_SHA256",
            "FULLBZ_BUNDLE_OUTPUT_SHA256",
            "FULLBZ_DATASET_MANIFEST_SHA256",
            "GRID_BASE_CONTRACT_SHA256",
            "GRID_HARTREE_TRUNCATED_CONTRACT_SHA256",
            "RUNNER_SHA256",
        )
        for value in required:
            with self.subTest(value=value):
                self.assertIn(value, self.source)

    def test_requires_pinned_full_bz_same_dataset(self) -> None:
        for value in (
            "producer_commit=dd4216653386d32f79e3219f3ea5dd2d229c1c5a",
            "scf_kpoints=64",
            "full_bz_kpoints=64",
            "symmetry=-1_full_bz",
            "n_scf_kpoints 64",
            "KS_eigenvector_*.dat",
        ):
            with self.subTest(value=value):
                self.assertIn(value, self.source)

    def test_runs_legacy_and_current_null_delta_controls(self) -> None:
        required = (
            "run_legacy legacy-control-off 1 0",
            "run_legacy legacy-control-on 1 1",
            "compare_legacy_hartree_null_delta_v1.py",
            "run_current current-control-off 1 false",
            "run_current current-parity-on 2 true",
            "compare_qsgw_hartree_null_delta_v1.py",
            "legacy_null_delta_side_effect_guard=true",
            "current_null_delta_side_effect_guard=true",
        )
        for value in required:
            with self.subTest(value=value):
                self.assertIn(value, self.source)

    def test_runs_two_round_parity_and_independent_contraction(self) -> None:
        required = (
            "run_legacy legacy-parity-on 2 1",
            "compare_qsgw_legacy_hartree_v4_current_v6.py",
            "--iterations 0:2",
            "--expected-legacy-beta 0.2",
            "--expected-current-beta 0.2",
            "validate_qsgw_hartree_contraction_v1.py",
            "--expected-normalization legacy_extra_inverse_nk",
            "--coulomb-prefix coulomb_cut_",
            "current_independent_contraction=true",
        )
        for value in required:
            with self.subTest(value=value):
                self.assertIn(value, self.source)

    def test_runner_has_no_remote_or_scheduler_commands(self) -> None:
        for forbidden in ("ssh ", "scp ", "sbatch ", "srun "):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, self.source)


if __name__ == "__main__":
    unittest.main()
