from pathlib import Path
import unittest


RUNNER = Path(__file__).with_name("run_fish_gate0_current_v2.sh")


class FishGate0CurrentV2ContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = RUNNER.read_text(encoding="utf-8")

    def test_pins_current_upstream_and_gate_version(self):
        self.assertIn(
            "upstream_commit=67b9888dac0d09870361398165d0b3c1acc931ff",
            self.source,
        )
        self.assertIn("gate=fish_gate0_current_v2", self.source)
        self.assertIn("FISH_GATE0_CURRENT_V2=PASS", self.source)

    def test_oneapi_setup_survives_errexit_but_checks_return_code(self):
        self.assertIn("set +eu\nsource /opt/intel/oneapi/setvars.sh --force", self.source)
        self.assertIn("oneapi_rc=$?\nset -eu\ntest \"$oneapi_rc\" -eq 0", self.source)

    def test_protected_shared_numerics_are_compared_to_upstream(self):
        for path in (
            "src/core/dielecmodel.cpp",
            "src/core/gw.cpp",
            "src/core/exx.cpp",
            "src/api/compute_g0w0.cpp",
            "driver/tasks/g0w0.cpp",
        ):
            self.assertIn(path, self.source)
        self.assertIn('test ! -s "$run_root/protected-diff.patch"', self.source)

    def test_requires_immutable_candidate_and_runner_hashes(self):
        self.assertIn("CANDIDATE_COMMIT must be the exact clean candidate commit", self.source)
        self.assertIn("RUNNER_SHA256 must identify this exact LF-normalized runner", self.source)
        self.assertIn('test ! -e "$source_root"', self.source)
        self.assertIn('test ! -e "$run_root"', self.source)


if __name__ == "__main__":
    unittest.main()
