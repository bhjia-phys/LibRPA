#!/usr/bin/env python3
"""Static tests for the immutable Si k444 full-BZ Hartree bundle assembler."""

from __future__ import annotations

import unittest
from pathlib import Path


RUNNER = Path(__file__).with_name(
    "assemble_pinned_abacus_si_k444_fullbz_hartree_bundle_v1.slurm"
)


class FullBzHartreeBundleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = RUNNER.read_text(encoding="utf-8")

    def test_requires_clean_completed_full_bz_producer(self) -> None:
        for value in (
            'test -e "$producer/COMPLETE"',
            'test ! -e "$producer/FAILED"',
            "OUTPUT_VALIDATION.json",
            'report["bz_sampling"]["n_scf"] == 64',
            'report["bz_sampling"]["full_bz"] is True',
            'report["structure"]["n_symops"] == 0',
        ):
            with self.subTest(value=value):
                self.assertIn(value, self.source)

    def test_freezes_64_state_basis_files_and_exact_contracts(self) -> None:
        for value in (
            "prepare_abacus_qsgw_fullbz_contract_v1.py",
            "extend_qsgw_contract_with_hartree_v1.py",
            "qsgw_input.hartree-full.contract",
            "qsgw_input.hartree-truncated.contract",
            "seq 0 63",
            "seq 1 64",
            "basis state",
            "gauge mf0_state",
        ):
            with self.subTest(value=value):
                self.assertIn(value, self.source)

    def test_records_parity_runner_provenance_contract(self) -> None:
        for value in (
            "producer_commit=dd4216653386d32f79e3219f3ea5dd2d229c1c5a",
            "grid=4x4x4",
            "scf_kpoints=64",
            "full_bz_kpoints=64",
            "symmetry=-1_full_bz",
            "reader_version=0",
            "hartree_contracts=full,truncated",
        ):
            with self.subTest(value=value):
                self.assertIn(value, self.source)

    def test_hashes_and_makes_bundle_immutable(self) -> None:
        for value in (
            "SOURCE_ARTIFACT_SHA256SUMS.txt",
            "SOURCE_COPY_VERIFICATION.txt",
            "DATASET_SHA256SUMS.txt",
            "OUTPUT_SHA256SUMS.txt",
            "chmod a-w",
        ):
            with self.subTest(value=value):
                self.assertIn(value, self.source)

    def test_freezes_exact_executed_runner(self) -> None:
        self.assertIn('${BASH_SOURCE[0]}', self.source)
        self.assertIn("executed-runner.slurm", self.source)
        self.assertIn("runner_sha256=", self.source)

    def test_has_no_remote_or_nested_scheduler_commands(self) -> None:
        for forbidden in ("ssh ", "scp ", "sbatch ", "srun "):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, self.source)


if __name__ == "__main__":
    unittest.main()
