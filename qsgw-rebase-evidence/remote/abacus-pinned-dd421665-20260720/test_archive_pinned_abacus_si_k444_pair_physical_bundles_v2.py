#!/usr/bin/env python3
"""Static contract tests for the paired physical-bundle transport archive."""

from __future__ import annotations

import unittest
from pathlib import Path


class PairedBundleArchiveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = (
            Path(__file__)
            .with_name("archive_pinned_abacus_si_k444_pair_physical_bundles_v2.slurm")
            .read_text(encoding="utf-8")
        )

    def test_runs_on_p1_and_binds_the_completed_source(self) -> None:
        self.assertIn("#SBATCH --partition=p1", self.source)
        self.assertIn("SLURM_JOB_ID", self.source)
        self.assertIn("abacus-pinned-dd421665-si-k444-pair-physical-bundles-20260723-v2", self.source)
        self.assertIn('test -e "$bundle/PAIR_COMPLETE"', self.source)
        self.assertIn('test ! -e "$bundle/FAILED"', self.source)
        self.assertIn("expected_bundle_output_manifest_sha", self.source)
        self.assertIn("expected_pair_validation_sha", self.source)

    def test_rechecks_all_source_manifests_and_immutability(self) -> None:
        self.assertIn("sha256sum --check --quiet OUTPUT_SHA256SUMS.txt", self.source)
        self.assertIn("sha256sum --check --quiet DATASET_SHA256SUMS.txt", self.source)
        self.assertIn('find "$bundle" -type l', self.source)
        self.assertIn('find "$bundle" -perm /222', self.source)
        self.assertIn('"status": "PASS"', self.source)

    def test_creates_and_checks_a_single_zstd_tar(self) -> None:
        self.assertIn("zstd -T", self.source)
        self.assertIn("tar -C", self.source)
        self.assertIn("ARCHIVE_SHA256SUMS.txt", self.source)
        self.assertIn("archive-members.txt", self.source)
        self.assertIn("tar -xOf -", self.source)
        self.assertIn("source_output_manifest_from_archive.sha256", self.source)

    def test_freezes_terminal_archive_evidence(self) -> None:
        self.assertIn("ARCHIVE_COMPLETE", self.source)
        self.assertIn("FAILED", self.source)
        self.assertIn("OUTPUT_SHA256SUMS.txt", self.source)
        self.assertIn('find "$root" -type f -exec chmod a-w', self.source)
        self.assertIn('find "$root" -depth -type d -exec chmod a-w', self.source)
        self.assertNotIn("ln -s", self.source)
        self.assertNotRegex(self.source, r"(^|\s)rm\s")


if __name__ == "__main__":
    unittest.main()
