#!/usr/bin/env python3
"""Static contract tests for the matched p1 Si k444 producer pair."""

from __future__ import annotations

import unittest
from pathlib import Path


RUNNER = Path(__file__).with_name(
    "run_pinned_abacus_si_k444_pair_p1_v1.slurm"
)


class PairedP1ProducerRunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = RUNNER.read_text(encoding="utf-8")

    def test_uses_one_array_with_identical_p1_resources(self) -> None:
        for value in (
            "#SBATCH --partition=p1",
            "#SBATCH --nodes=8",
            "#SBATCH --ntasks-per-node=1",
            "#SBATCH --cpus-per-task=40",
            "#SBATCH --exclusive",
            "#SBATCH --array=0-1",
            'np="${SLURM_NTASKS:-8}"',
            'OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-40}"',
        ):
            self.assertIn(value, self.source)
        self.assertNotIn("48cp3", self.source)

    def test_changes_only_the_symmetry_input_between_array_members(self) -> None:
        for value in (
            '0) mode=symmetry; input_name=INPUT_scf_symmetry; symmetry_value=1 ;;',
            '1) mode=fullbz; input_name=INPUT_scf_fullbz; symmetry_value=-1 ;;',
            'cp "$contract/$input_name" "$run_dir/INPUT"',
            'cp "$contract/KPT_k444" "$run_dir/KPT"',
            'cp "$contract/STRU" "$run_dir/STRU"',
            "changed_factor=abacus_input_symmetry",
            "all_other_producer_inputs=identical",
        ):
            self.assertIn(value, self.source)

    def test_binds_binary_source_and_common_staging_manifest(self) -> None:
        for value in (
            "dd4216653386d32f79e3219f3ea5dd2d229c1c5a",
            "a2676c36e318da831339cfb147c6027f3b7aed0b40925fa4a6086a947a403b17",
            "COMMON_SHA256SUMS.txt",
            "sha256sum --check --quiet COMMON_SHA256SUMS.txt",
            "executed-runner.slurm",
            "runner_sha256=",
        ):
            self.assertIn(value, self.source)

    def test_uses_mode_specific_validators_and_markers(self) -> None:
        for value in (
            "validate_abacus_si_k444_symmetry_output_v1.py",
            "validate_abacus_si_k444_fullbz_output_v1.py",
            "OUTPUT_VALIDATION.json",
            "ARTIFACT_SHA256SUMS.txt",
            'touch "$mode_root/COMPLETE"',
            'test ! -e "$mode_root/FAILED"',
        ):
            self.assertIn(value, self.source)

    def test_has_no_remote_or_nested_scheduler_commands(self) -> None:
        for forbidden in ("ssh ", "scp ", "sbatch ", "srun "):
            self.assertNotIn(forbidden, self.source)


if __name__ == "__main__":
    unittest.main()
