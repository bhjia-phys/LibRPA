from pathlib import Path
import unittest


RUNNER = Path(__file__).with_name("recover_fish_gate1_current_v2.sh")


class RecoverGate1CurrentV2Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.text = RUNNER.read_text(encoding="utf-8")

    def test_binds_failed_current_source_run(self):
        expected = (
            "source_run=/home/bhj/ai-runs/librpa-qsgw-gate1-current-20260723-4f9ab0cf-g0w0-v2",
            "expected_source_runner_commit=25ec31772809968cde2079622991679ca37f213f",
            "expected_source_runner_sha=9e028d1eef9993a212c9073c5a70ae00746328d60eb9ad71906b5396a632f624",
            "expected_source_failed_sha=493d3d18e70a5c10ce9f3eaef8f36075615a3767916f25e9d3732bcf6b6fec07",
            'test -e "$source_run/FAILED"',
            'test ! -e "$source_run/GREEN_CONFIRMED"',
        )
        for value in expected:
            self.assertIn(value, self.text)

    def test_binds_current_gate0_and_executables(self):
        expected = (
            "expected_upstream_commit=67b9888dac0d09870361398165d0b3c1acc931ff",
            "expected_candidate_commit=4f9ab0cfc90f54910158ab01a877581b080f136e",
            "expected_gate0_provenance_sha=62a4026017c26b28f9c9503fb4c71a265345369db2a709b0f811a7000b5fd424",
            "expected_upstream_exe_sha=c2705015e2219c548ce6d6cfce93d32072b14391fbd651aab4e38c0bb125737c",
            "expected_candidate_exe_sha=77ab964e15f0cdfee05ad54cf5b9da4ad9e4e3ac1f6990c4b50e8f0a55a29b47",
        )
        for value in expected:
            self.assertIn(value, self.text)

    def test_preserves_source_failure_and_uses_versioned_postcheck(self):
        self.assertIn("SOURCE_RUN_SHA256SUMS.txt", self.text)
        self.assertIn("source_run_status=rejected_observer_threshold_only", self.text)
        self.assertIn("gate=fish_gate1_current_g0w0_ab_recovery_v2", self.text)
        self.assertNotIn('touch "$source_run/GREEN_CONFIRMED"', self.text)

    def test_uses_documented_tolerances_and_observed_values(self):
        self.assertIn("--max-abs-tolerance-ha 2e-10", self.text)
        self.assertIn("--relative-frobenius-tolerance 2e-10", self.text)
        self.assertIn("sigc_max_abs_difference_ha=1.1588952445590924e-10", self.text)
        self.assertIn("sigc_relative_frobenius_difference=5.661776804668715e-11", self.text)
        self.assertIn("energy_qp_max_abs_tolerance_ha=1e-9", self.text)

    def test_writes_immutable_terminal_markers(self):
        self.assertIn('>"$recovery_root/FAILED"', self.text)
        self.assertIn('touch "$recovery_root/GREEN_CONFIRMED"', self.text)
        self.assertIn("recovery_succeeded=1", self.text)


if __name__ == "__main__":
    unittest.main()
