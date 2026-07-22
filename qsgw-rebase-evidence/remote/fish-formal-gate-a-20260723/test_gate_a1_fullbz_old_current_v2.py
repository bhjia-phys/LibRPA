from pathlib import Path
import unittest


RUNNER = Path(__file__).with_name("gate-a1-fullbz-old-current-v2.sh")


class FishFormalGateA1RunnerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = RUNNER.read_text(encoding="utf-8")

    def test_binds_only_the_valid_legacy_full_bz_oracle(self):
        self.assertIn(
            "expected_old_commit=e08f4a130df7661e9ac355b9be45fb2bf9c3ed01",
            self.source,
        )
        self.assertIn("legacy_symmetry=off_full_bz", self.source)
        self.assertIn("--expected-legacy-symmetry off", self.source)
        self.assertNotIn("exact847", self.source)
        self.assertNotIn("legacy_symmetry=on", self.source)

    def test_binds_rebased_candidate_and_accepted_gate0(self):
        self.assertIn(
            "expected_upstream_commit=67b9888dac0d09870361398165d0b3c1acc931ff",
            self.source,
        )
        self.assertIn(
            "expected_candidate_commit=4f9ab0cfc90f54910158ab01a877581b080f136e",
            self.source,
        )
        self.assertIn("GATE0_PROVENANCE_SHA256", self.source)
        self.assertIn("CANDIDATE_EXE_SHA256", self.source)
        self.assertIn("gate=fish_gate0_current_v2", self.source)

    def test_requires_accepted_gate1_before_numerics(self):
        self.assertIn("GATE1_PROVENANCE_SHA256", self.source)
        self.assertIn(
            "librpa-qsgw-gate1-current-postcheck-20260723-36d74369-v1",
            self.source,
        )
        self.assertIn(
            "gate=fish_gate1_current_g0w0_ab_recovery_v2", self.source
        )
        self.assertIn('test -e "$candidate_gate1/GREEN_CONFIRMED"', self.source)
        self.assertIn('test ! -e "$candidate_gate1/FAILED"', self.source)
        self.assertIn(
            'sha256sum --check --quiet OUTPUT_SHA256SUMS.txt', self.source
        )

    def test_binds_runner_checkout_and_writes_failure_marker(self):
        self.assertIn("RUNNER_SOURCE", self.source)
        self.assertIn('git -C "$RUNNER_SOURCE" rev-parse HEAD', self.source)
        self.assertIn('git -C "$RUNNER_SOURCE" status --porcelain', self.source)
        self.assertIn('>"$run_root/FAILED"', self.source)
        self.assertIn('touch "$run_root/GREEN_CONFIRMED"', self.source)
        self.assertIn("run_succeeded=1", self.source)

    def test_runs_both_required_iteration_and_mixing_modes(self):
        self.assertIn("run_mode no-mix-miniter2 2 1 none 0.2", self.source)
        self.assertIn(
            "run_mode linear-beta-0.2-miniter5 5 0.2 linear 0.2",
            self.source,
        )
        self.assertIn('--iterations "0:$target_iter"', self.source)
        self.assertIn("qsgw_update_hartree = false", self.source)
        self.assertIn("replace_w_head = false", self.source)

    def test_runs_directly_on_fish_with_fixed_parallelism(self):
        self.assertIn("mpi_ranks=4", self.source)
        self.assertIn("omp_threads=12", self.source)
        self.assertIn('mpirun -np "$mpi_ranks"', self.source)
        self.assertIn("execution_surface=fish_direct", self.source)
        self.assertNotIn("#SBATCH", self.source)
        self.assertNotIn("SLURM_", self.source)

    def test_preserves_and_relocates_original_bundle_manifest(self):
        self.assertIn("expected_common_output_manifest_sha", self.source)
        self.assertIn("common-input-OUTPUT_SHA256SUMS.txt", self.source)
        self.assertIn("common-input-OUTPUT_SHA256SUMS.relocated.txt", self.source)
        self.assertIn("sha256sum --check --quiet DATASET_SHA256SUMS.txt", self.source)

    def test_requires_full_matrix_and_state_observers(self):
        for observer in (
            "compare_qsgw_legacy_v4_current_v6.py",
            "validate_qsgw_trace_closure.py",
            "validate_qsgw_fixed_basis.py",
            "validate_qsgw_initial_state.py",
            "current-electron-count.txt",
        ):
            self.assertIn(observer, self.source)
        self.assertIn("matrix_state_and_invariants=passed", self.source)


if __name__ == "__main__":
    unittest.main()
