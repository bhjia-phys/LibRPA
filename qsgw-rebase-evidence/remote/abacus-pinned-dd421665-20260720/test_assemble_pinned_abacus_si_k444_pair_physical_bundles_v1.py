#!/usr/bin/env python3
"""Static contract tests for matched symmetry/full-BZ physical bundles."""

from __future__ import annotations

import unittest
from pathlib import Path


class PairedPhysicalBundleAssemblerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = (
            Path(__file__)
            .with_name("assemble_pinned_abacus_si_k444_pair_physical_bundles_v1.slurm")
            .read_text(encoding="utf-8")
        )

    def test_runs_only_as_a_slurm_job_and_binds_completed_pair(self) -> None:
        self.assertIn("#SBATCH --partition=p1", self.source)
        self.assertIn("SLURM_JOB_ID", self.source)
        self.assertIn("abacus-pinned-dd421665-si-k444-pair-p1-20260723-v1", self.source)
        self.assertIn("expected_symmetry_artifact_manifest_sha", self.source)
        self.assertIn("expected_fullbz_artifact_manifest_sha", self.source)
        self.assertIn('test -e "$source_symmetry/COMPLETE"', self.source)
        self.assertIn('test -e "$source_fullbz/COMPLETE"', self.source)

    def test_changes_only_the_producer_symmetry_mode(self) -> None:
        self.assertIn("all_other_producer_inputs=identical", self.source)
        self.assertIn("producer_mode=symmetry", self.source)
        self.assertIn("producer_mode=fullbz", self.source)
        self.assertIn("source_commit=dd4216653386d32f79e3219f3ea5dd2d229c1c5a", self.source)
        self.assertIn("binary_sha256=a2676c36e318da831339cfb147c6027f3b7aed0b40925fa4a6086a947a403b17", self.source)

    def test_builds_both_physical_structure_and_bz_overlays(self) -> None:
        self.assertIn("build_abacus_physical_stru_overlay_v1.py", self.source)
        self.assertIn("build_abacus_physical_bz_sampling_overlay_v1.py", self.source)
        self.assertIn("--allow-no-symmetry", self.source)
        self.assertIn("PHYSICAL_STRU_OVERLAY.json", self.source)
        self.assertIn("PHYSICAL_BZ_SAMPLING_OVERLAY.json", self.source)
        self.assertIn("non_cartesian_tokens_unchanged", self.source)

    def test_generates_matching_state_basis_contracts(self) -> None:
        self.assertIn("prepare_abacus_qsgw_ibz_contract_v3.py", self.source)
        self.assertIn("prepare_abacus_qsgw_fullbz_contract_v1.py", self.source)
        self.assertIn("generate_abacus_basis_metadata_v1.py", self.source)
        self.assertIn("expected_kpoints=8", self.source)
        self.assertIn("expected_kpoints=64", self.source)
        self.assertIn("grep -Fqx 'basis state'", self.source)
        self.assertIn("grep -Fqx 'gauge mf0_state'", self.source)

    def test_copies_only_selected_inputs_and_hashes_every_copy(self) -> None:
        self.assertIn("copy_selected", self.source)
        self.assertIn("SOURCE_SELECTED_SHA256SUMS.txt", self.source)
        self.assertIn("SOURCE_COPY_VERIFICATION.txt", self.source)
        self.assertIn('cp --reflink=auto "$source_path" "$target_path"', self.source)
        self.assertNotIn("ln -s", self.source)
        self.assertNotRegex(self.source, r"(^|\s)rm\s")

    def test_validates_pair_gap_and_freezes_terminal_artifacts(self) -> None:
        self.assertIn("PAIR_VALIDATION.json", self.source)
        self.assertIn("gap_difference_ev", self.source)
        self.assertIn("1.0e-6", self.source)
        self.assertIn("DATASET_SHA256SUMS.txt", self.source)
        self.assertIn("OUTPUT_SHA256SUMS.txt", self.source)
        self.assertIn("PAIR_COMPLETE", self.source)
        self.assertIn("FAILED", self.source)
        self.assertIn('find "$root" -type f -exec chmod a-w', self.source)
        self.assertIn('find "$root" -depth -type d -exec chmod a-w', self.source)


if __name__ == "__main__":
    unittest.main()
