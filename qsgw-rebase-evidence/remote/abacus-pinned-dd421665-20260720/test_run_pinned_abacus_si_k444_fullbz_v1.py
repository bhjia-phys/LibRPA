#!/usr/bin/env python3
"""Static contract tests for the pinned Si k444 full-BZ producer runner."""

from __future__ import annotations

import unittest
from pathlib import Path


RUNNER = Path(__file__).with_name("run_pinned_abacus_si_k444_fullbz_v1.slurm")


class FullBzProducerRunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = RUNNER.read_text(encoding="utf-8")

    def test_binds_pinned_source_binary_and_input(self) -> None:
        for value in (
            "dd4216653386d32f79e3219f3ea5dd2d229c1c5a",
            "a2676c36e318da831339cfb147c6027f3b7aed0b40925fa4a6086a947a403b17",
            'cp "$contract/INPUT_scf_fullbz" "$run_dir/INPUT"',
            'cp "$contract/KPT_k444" "$run_dir/KPT"',
            "out_librpa_reader_version",
        ):
            with self.subTest(value=value):
                self.assertIn(value, self.source)

    def test_runs_as_slurm_numerical_job_and_validates_full_bz(self) -> None:
        for value in (
            "#SBATCH",
            "validate_abacus_si_k444_fullbz_output_v1.py",
            "symmetry=-1",
            "scf_kpoints=64",
            "full_bz_kpoints=64",
            "reader_version=0",
        ):
            with self.subTest(value=value):
                self.assertIn(value, self.source)

    def test_freezes_exact_executed_runner(self) -> None:
        self.assertIn('${BASH_SOURCE[0]}', self.source)
        self.assertIn("executed-runner.slurm", self.source)
        self.assertIn("runner_sha256=", self.source)
        self.assertIn("ARTIFACT_SHA256SUMS.txt", self.source)

    def test_has_no_remote_or_nested_scheduler_commands(self) -> None:
        for forbidden in ("ssh ", "scp ", "sbatch ", "srun "):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, self.source)


if __name__ == "__main__":
    unittest.main()
