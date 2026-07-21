#!/usr/bin/env python3
"""Static contract tests for the pinned Si k444 bundle assembly runner."""

from __future__ import annotations

import unittest
from pathlib import Path


class BundleAssemblyContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = (
            Path(__file__).with_name(
                "assemble_pinned_abacus_si_k444_symmetry_bundle_v2.slurm"
            )
            .read_text(encoding="utf-8")
        )

    def test_recovers_only_through_completed_postcheck(self) -> None:
        self.assertIn('test -e "$producer/FAILED"', self.source)
        self.assertIn('test ! -e "$producer/COMPLETE"', self.source)
        self.assertIn('test -e "$postcheck/COMPLETE"', self.source)
        self.assertIn('test ! -e "$postcheck/FAILED"', self.source)
        self.assertIn("symmetry-postcheck-20260720-v4", self.source)
        self.assertIn("SOURCE_FAILED_SHA256SUMS.txt", self.source)
        self.assertIn("source_numeric_stage=completed", self.source)
        self.assertIn("source_terminal_marker=FAILED_observer_only", self.source)

    def test_binds_only_producer_native_vxck_files(self) -> None:
        self.assertIn("prepare_abacus_qsgw_ibz_contract_v2.py", self.source)
        self.assertIn('producer_name="vxck${index}_nao.txt"', self.source)
        self.assertNotIn("legacy_name=", self.source)
        self.assertIn("vxc_dataset_schema=producer_native_vxck_only", self.source)
        self.assertIn("derived_vxcs_aliases=none", self.source)

    def test_hashes_source_and_copied_dataset(self) -> None:
        self.assertIn("SOURCE_ARTIFACT_SHA256SUMS.txt", self.source)
        self.assertIn("DATASET_SHA256SUMS.txt", self.source)
        self.assertIn("SOURCE_COPY_VERIFICATION.txt", self.source)
        self.assertIn("copied source artifacts are SHA256-identical", self.source)

    def test_keeps_runtime_compatibility_pending(self) -> None:
        self.assertIn("numerical_runtime_parity_pending", self.source)
        self.assertIn(
            "vxc_legacy_compatibility=pending_same_bundle_runtime_gate",
            self.source,
        )


if __name__ == "__main__":
    unittest.main()
