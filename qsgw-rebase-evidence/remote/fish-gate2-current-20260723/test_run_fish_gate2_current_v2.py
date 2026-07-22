from pathlib import Path
import unittest


RUNNER = Path(__file__).with_name("run_fish_gate2_current_v2.sh")


class FishGate2CurrentRunnerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = RUNNER.read_text(encoding="utf-8")

    def test_binds_current_upstream_candidate_and_executable(self):
        self.assertIn(
            "expected_upstream_commit=67b9888dac0d09870361398165d0b3c1acc931ff",
            self.source,
        )
        self.assertIn(
            "expected_candidate_commit=4f9ab0cfc90f54910158ab01a877581b080f136e",
            self.source,
        )
        self.assertIn(
            "expected_candidate_exe_sha=77ab964e15f0cdfee05ad54cf5b9da4ad9e4e3ac1f6990c4b50e8f0a55a29b47",
            self.source,
        )

    def test_binds_accepted_current_gate0_and_gate1(self):
        for value in (
            "librpa-qsgw-gate0-20260723-4f9ab0cf-v1",
            "librpa-qsgw-gate1-current-postcheck-20260723-36d74369-v1",
            "librpa-qsgw-gate1-current-20260723-4f9ab0cf-g0w0-v2",
            "gate=fish_gate1_current_g0w0_ab_recovery_v2",
            "expected_gate1_provenance_sha=880115845b7cb5a983885613a1b2325607bb269836f681b0af5051b8f9d74d0d",
            "expected_gate1_manifest_sha=74a3453c46df34e7b0ec8430260cd47182e6d3d01a409f4a313dc976948a127f",
            "expected_gate1_source_manifest_sha=22840e6cf84d18526f05283231c47eb0a1784c58764f4387a7391c1f50aa5155",
        ):
            self.assertIn(value, self.source)

    def test_runs_first_self_energy_with_fixed_contract(self):
        for value in (
            "qsgw_min_iter = 1",
            "qsgw_max_iter = 1",
            "qsgw_mixer = none",
            "qsgw_update_hartree = false",
            "replace_w_head = false",
            "use_symmetry_exx = true",
            "use_symmetry_gw = true",
            "use_symmetry_rpa = true",
            "--iteration 1",
            "--channel 0",
            "--source kgrid",
        ):
            self.assertIn(value, self.source)

    def test_uses_current_only_and_no_invalid_legacy_oracle(self):
        self.assertNotIn("exact847", self.source)
        self.assertNotIn("legacy_symmetry", self.source)
        self.assertIn("candidate_qsgw_first_self_energy_vs_accepted_upstream_g0w0", self.source)

    def test_runs_directly_on_fish_and_records_terminal_state(self):
        self.assertIn("mpi_ranks=1", self.source)
        self.assertIn("omp_threads=32", self.source)
        self.assertIn('mpirun -np "$mpi_ranks"', self.source)
        self.assertNotIn("#SBATCH", self.source)
        self.assertIn('>"$run_root/FAILED"', self.source)
        self.assertIn('touch "$run_root/GREEN_CONFIRMED"', self.source)

    def test_binds_clean_runner_checkout_and_observers(self):
        self.assertIn('git -C "$RUNNER_SOURCE" rev-parse HEAD', self.source)
        self.assertIn('git -C "$RUNNER_SOURCE" status --porcelain', self.source)
        self.assertIn("compare_qsgw_iter1_g0w0_v1.py", self.source)
        self.assertIn("validate_qsgw_iter1_v6.py", self.source)
        self.assertIn("SOURCE_RUN_SHA256SUMS.txt", self.source)
        self.assertIn("OUTPUT_SHA256SUMS.txt", self.source)


if __name__ == "__main__":
    unittest.main()
