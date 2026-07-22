#!/usr/bin/env python3
"""Static contract tests for pinned current-QSGW symmetry-side production."""

from __future__ import annotations

import unittest
from pathlib import Path


class CurrentSymmetrySideRunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = (
            Path(__file__)
            .with_name("gate-a23-current-symmetry-side-v1.sh")
            .read_text(encoding="utf-8")
        )

    def test_binds_clean_committed_runner_and_gate0_executable(self) -> None:
        for token in (
            "RUNNER_COMMIT",
            "RUNNER_SOURCE",
            "RUNNER_SHA256",
            "GATE0_PROVENANCE_SHA256",
            "CANDIDATE_EXE_SHA256",
        ):
            self.assertIn(token, self.source)
        self.assertIn('git -C "$RUNNER_SOURCE" rev-parse HEAD', self.source)
        self.assertIn('git -C "$RUNNER_SOURCE" status --porcelain', self.source)
        self.assertIn("4f9ab0cfc90f54910158ab01a877581b080f136e", self.source)
        self.assertIn("67b9888dac0d09870361398165d0b3c1acc931ff", self.source)

    def test_binds_pinned_symmetry_bundle_and_all_manifests(self) -> None:
        self.assertIn(
            "abacus-pinned-dd421665-si-k444-symmetry-bundle-20260720-v4",
            self.source,
        )
        for token in (
            "expected_bundle_provenance_sha",
            "expected_bundle_output_sha",
            "expected_dataset_manifest_sha",
            "expected_contract_sha",
            "expected_vxc_manifest_sha",
            "sha256sum --check --quiet OUTPUT_SHA256SUMS.txt",
            "sha256sum --check --quiet DATASET_SHA256SUMS.txt",
        ):
            self.assertIn(token, self.source)
        self.assertIn('test -z "$(find "$bundle" -type l -print -quit)"', self.source)
        self.assertIn('test -z "$(find "$bundle" -perm /222 -print -quit)"', self.source)

    def test_runs_only_current_symmetry_with_both_required_mixers(self) -> None:
        self.assertIn("run_mode no-mix-miniter2 2 none", self.source)
        self.assertIn("run_mode linear-beta-0.2-miniter5 5 linear", self.source)
        self.assertIn("qsgw_mixing_beta = 0.2", self.source)
        self.assertIn("use_abacus_exx_symmetry = true", self.source)
        self.assertIn("use_abacus_gw_symmetry = true", self.source)
        self.assertIn("use_symmetry_exx = true", self.source)
        self.assertIn("use_symmetry_gw = true", self.source)
        self.assertIn("use_symmetry_rpa = true", self.source)
        self.assertNotIn("exact847", self.source.lower())
        self.assertNotIn("legacy", self.source.lower())

    def test_keeps_headwing_hartree_and_band_out_of_this_gate(self) -> None:
        self.assertIn("qsgw_iterative_headwing = false", self.source)
        self.assertIn("qsgw_update_hartree = false", self.source)
        self.assertIn("# headwing disabled_stage1", self.source)
        self.assertIn("# hartree disabled_stage1", self.source)
        self.assertIn("# band disabled_stage1", self.source)
        self.assertNotIn("task = qsgw_band", self.source)

    def test_uses_matching_parallelism_and_deterministic_reduction(self) -> None:
        self.assertIn("mpi_ranks=4", self.source)
        self.assertIn("omp_threads=12", self.source)
        self.assertIn("LIBRI_DETERMINISTIC_REDUCTION=1", self.source)
        self.assertIn('mpirun -np "$mpi_ranks"', self.source)

    def test_validates_every_iteration_and_state_invariant(self) -> None:
        self.assertIn("qsgw_write_iteration_matrices = true", self.source)
        self.assertIn(
            'cp "$closure_source" "$tool_dir/validate_qsgw_trace_closure_v3.py"',
            self.source,
        )
        self.assertIn("validate_qsgw_trace_closure.py", self.source)
        self.assertIn("validate_qsgw_fixed_basis.py", self.source)
        self.assertIn("validate_qsgw_initial_state.py", self.source)
        self.assertIn("current-electron-count.txt", self.source)
        self.assertIn("qsgw_matrices.dat", self.source)
        self.assertIn("qsgw_eigenvalues.dat", self.source)
        self.assertIn("qsgw_iterations.dat", self.source)
        self.assertIn('"0:$target_iter"', self.source)

    def test_does_not_claim_a2_or_a3_before_fullbz_comparison(self) -> None:
        self.assertIn("SYMMETRY_SIDE_COMPLETE", self.source)
        self.assertIn("acceptance=false_pending_full_bz_control", self.source)
        self.assertNotIn("A2_ACCEPTANCE", self.source)
        self.assertNotIn("A3_ACCEPTANCE", self.source)
        self.assertNotIn('touch "$run_root/GREEN_CONFIRMED"', self.source)
        self.assertNotIn('>"$run_root/GREEN_CONFIRMED"', self.source)


if __name__ == "__main__":
    unittest.main()
