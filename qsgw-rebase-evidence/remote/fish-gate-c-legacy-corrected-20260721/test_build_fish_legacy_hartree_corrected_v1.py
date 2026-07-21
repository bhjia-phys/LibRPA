#!/usr/bin/env python3

from __future__ import annotations

import unittest
from pathlib import Path


RUNNER = Path(__file__).resolve().parent / "build_fish_legacy_hartree_corrected_v1.sh"


class CorrectedLegacyBuildRunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = RUNNER.read_text(encoding="utf-8")

    def test_binds_exact_seed_patch_and_normalized_source(self) -> None:
        required = (
            "6c79c6731d3420b3c9f70c97dea8c63236bca2952216bb67d9e6b9839552c9d1",
            "50215d59480b46bea7e3edb34ac5e453165afcf0a66bc6a48053b17b491b303f",
            "127b8c3927de328f095ce1be287fb492b32c7e6e3bd09fc23526252da56dd234",
            "54a5eae4f96b7a39e1400881ce96824f40b8e4ac026dc819b26e519db3e02ef9",
            "git apply --check",
            "dos2unix",
        )
        for value in required:
            with self.subTest(value=value):
                self.assertIn(value, self.source)

    def test_requires_clean_candidate_and_immutable_output(self) -> None:
        for value in (
            'git -C "$CANDIDATE_SOURCE" rev-parse HEAD',
            'git -C "$CANDIDATE_SOURCE" status --porcelain',
            'test ! -e "$root"',
            'RUNNER_SHA256',
            'OUTPUT_SHA256SUMS.txt',
            'GREEN_CONFIRMED',
        ):
            with self.subTest(value=value):
                self.assertIn(value, self.source)

    def test_preserves_normal_reader_and_dedicated_hartree_map(self) -> None:
        for value in (
            'read_Vq_row(driver_params.input_dir, "coulomb_cut_"',
            "hartree_full_vq_cut, meanfield.get_n_kpoints(), Rlist,",
            "gw_vq_reader=distributed_row_unchanged",
            "reader_state_restored=Vq_cut,n_irk_points,irk_points,irk_weight",
        ):
            with self.subTest(value=value):
                self.assertIn(value, self.source)

    def test_build_runner_has_no_remote_or_scheduler_commands(self) -> None:
        for forbidden in ("ssh ", "scp ", "sbatch ", "srun "):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, self.source)


if __name__ == "__main__":
    unittest.main()
