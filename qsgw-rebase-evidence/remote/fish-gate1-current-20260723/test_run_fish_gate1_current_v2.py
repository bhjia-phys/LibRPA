from pathlib import Path
import unittest


RUNNER = Path(__file__).with_name("run_fish_gate1_current_v2.sh")


class Gate1RunnerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.text = RUNNER.read_text(encoding="utf-8")

    def test_binds_current_upstream_candidate_and_gate0(self):
        expected = (
            "expected_upstream_commit=67b9888dac0d09870361398165d0b3c1acc931ff",
            "expected_candidate_commit=4f9ab0cfc90f54910158ab01a877581b080f136e",
            "expected_gate0_provenance_sha=62a4026017c26b28f9c9503fb4c71a265345369db2a709b0f811a7000b5fd424",
            "expected_gate0_manifest_sha=845d309c485d5fd8060a70faf57584aa1e6c44e267595ab15f4a2ec12417c5dd",
            "gate0=/home/bhj/ai-runs/librpa-qsgw-gate0-20260723-4f9ab0cf-v1",
        )
        for value in expected:
            self.assertIn(value, self.text)

    def test_preserves_controlled_g0w0_pair(self):
        expected = (
            "mpi_ranks=1",
            "omp_threads=32",
            "task = g0w0",
            "use_symmetry_exx = true",
            "use_symmetry_gw = true",
            "use_symmetry_rpa = true",
            "replace_w_head = false",
            "--max-abs-tolerance-ha 1e-10",
            "--relative-frobenius-tolerance 1e-10",
            "--max-abs-tolerance-ha 1e-9",
        )
        for value in expected:
            self.assertIn(value, self.text)

    def test_reuses_frozen_assets_without_mutating_them(self):
        self.assertIn(
            "asset_dir=qsgw-rebase-evidence/remote/fish-gate1-current-20260722",
            self.text,
        )
        self.assertIn('ln -s "$path" "$overlay/$(basename "$path")"', self.text)
        self.assertNotIn("cp -r \"$dataset\"", self.text)

    def test_oneapi_setup_is_checked_explicitly(self):
        self.assertIn("set +eu", self.text)
        self.assertIn("oneapi_rc=$?", self.text)
        self.assertIn("test \"$oneapi_rc\" -eq 0", self.text)

    def test_writes_immutable_terminal_markers(self):
        self.assertIn('touch "$run_root/GREEN_CONFIRMED"', self.text)
        self.assertIn('>"$run_root/FAILED"', self.text)
        self.assertIn("gate=fish_gate1_current_g0w0_ab_v2", self.text)


if __name__ == "__main__":
    unittest.main()
