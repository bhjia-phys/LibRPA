#!/usr/bin/env python3
"""Static contract tests for the corrected fish symmetry bundle assembler."""

from __future__ import annotations

import unittest
from pathlib import Path


SOURCE = Path(__file__).with_name(
    "assemble_fish_si_k444_symmetry_physical_bundle_v1.sh"
)


class SymmetryPhysicalBundleAssemblerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = SOURCE.read_text(encoding="utf-8")

    def test_binds_clean_committed_runner_and_frozen_v4_source(self) -> None:
        for token in ("RUNNER_COMMIT", "RUNNER_SOURCE", "RUNNER_SHA256"):
            self.assertIn(token, self.source)
        self.assertIn('git -C "$RUNNER_SOURCE" status --porcelain', self.source)
        self.assertIn(
            "abacus-pinned-dd421665-si-k444-symmetry-bundle-20260720-v4",
            self.source,
        )
        self.assertIn("expected_source_dataset_manifest_sha", self.source)
        self.assertIn("expected_source_output_manifest_sha", self.source)
        self.assertIn("sha256sum --check --quiet DATASET_SHA256SUMS.txt", self.source)
        self.assertIn("sha256sum --check --quiet OUTPUT_SHA256SUMS.txt", self.source)

    def test_copies_physical_files_without_mutating_source_or_using_symlinks(self) -> None:
        self.assertIn('test ! -e "$root"', self.source)
        self.assertIn('cp --reflink=auto "$path" "$dataset/$name"', self.source)
        self.assertIn("SOURCE_REUSE_VERIFICATION.txt", self.source)
        self.assertIn('test -z "$(find "$root" -type l -print -quit)"', self.source)
        self.assertNotIn("ln -s", self.source)
        self.assertNotRegex(self.source, r"(^|\s)rm\s")

    def test_rebuilds_only_physical_structure_and_derived_contract_files(self) -> None:
        for name in (
            "stru_out",
            "qsgw_input.contract",
            "qsgw_vxc_scf.manifest",
            "qsgw_input_contract.summary.json",
        ):
            self.assertIn(name, self.source)
        self.assertIn("build_abacus_physical_stru_overlay_v1.py", self.source)
        self.assertIn("PHYSICAL_STRU_OVERLAY.json", self.source)
        self.assertIn('report["symmetry_operation_count"] == 48', self.source)
        self.assertIn('report["atom_symmetry_max_abs"] <= 1.0e-10', self.source)

    def test_regenerates_state_basis_contract_and_validates_stage1_scope(self) -> None:
        self.assertIn("prepare_abacus_qsgw_ibz_contract_v3.py", self.source)
        self.assertIn("generate_abacus_basis_metadata_v1.py", self.source)
        self.assertIn("grep -Fqx 'basis state'", self.source)
        self.assertIn("grep -Fqx 'gauge mf0_state'", self.source)
        self.assertIn("grep -Fqx 'n_scf_kpoints 8'", self.source)
        self.assertIn("grep -Fqx 'headwing_update none'", self.source)
        self.assertIn("grep -Fqx 'hartree_update off'", self.source)
        self.assertIn("grep -Fqx 'band_update off'", self.source)

    def test_writes_manifests_provenance_and_read_only_terminal_state(self) -> None:
        for token in (
            "DATASET_SHA256SUMS.txt",
            "OUTPUT_SHA256SUMS.txt",
            "PROVENANCE.txt",
            "COMPLETE",
            "FAILED",
            "find \"$root\" -type f -exec chmod a-w",
            "find \"$root\" -depth -type d -exec chmod a-w",
        ):
            self.assertIn(token, self.source)
        self.assertIn("physical_lattice_source=matching_abacus_input_STRU", self.source)
        self.assertIn("shared_gw_source_changes=none", self.source)


if __name__ == "__main__":
    unittest.main()
