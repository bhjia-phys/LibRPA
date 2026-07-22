#!/usr/bin/env python3
"""Static contract tests for the state-basis Si k444 bundle runner."""

from __future__ import annotations

import unittest
from pathlib import Path


class StateBasisBundleAssemblyContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = (
            Path(__file__).with_name(
                "assemble_pinned_abacus_si_k444_symmetry_bundle_v4.slurm"
            )
            .read_text(encoding="utf-8")
        )

    def test_recovers_only_through_completed_postcheck(self) -> None:
        self.assertIn('test -e "$producer/FAILED"', self.source)
        self.assertIn('test ! -e "$producer/COMPLETE"', self.source)
        self.assertIn('test -e "$postcheck/COMPLETE"', self.source)
        self.assertIn('test ! -e "$postcheck/FAILED"', self.source)
        self.assertIn("symmetry-postcheck-20260720-v4", self.source)
        self.assertIn("source_numeric_stage=completed", self.source)

    def test_requires_state_basis_source_audit(self) -> None:
        self.assertIn("audit_native_vxck_state_basis_v1.py", self.source)
        self.assertIn("native_vxck_state_basis_v1.json", self.source)
        self.assertIn('report["producer"]["matrix_basis"] == "ks_state"', self.source)
        self.assertIn(
            'report["producer"]["matrix_shape"] == "n_bands_x_n_bands"',
            self.source,
        )
        self.assertIn('report["legacy"]["basis_transform"] == "none"', self.source)
        self.assertIn('report["candidate"]["basis_transform"] == "none"', self.source)

    def test_generates_state_basis_manifest_v3(self) -> None:
        self.assertIn("prepare_abacus_qsgw_ibz_contract_v3.py", self.source)
        self.assertNotIn(
            'python3 -B "$tools_dir/prepare_abacus_qsgw_ibz_contract_v2.py"',
            self.source,
        )
        self.assertIn("grep -Fqx 'basis state'", self.source)
        self.assertIn("grep -Fqx 'gauge mf0_state'", self.source)
        self.assertIn('"vxc_matrix_basis": "ks_state"', self.source)
        self.assertIn('"vxc_basis_transform": "none"', self.source)

    def test_freezes_only_native_vxck_and_records_basis_contract(self) -> None:
        self.assertIn('producer_name="vxck${index}_nao.txt"', self.source)
        self.assertNotIn("legacy_name=", self.source)
        self.assertIn("vxc_dataset_schema=producer_native_vxck_only", self.source)
        self.assertIn("vxc_matrix_basis=ks_state", self.source)
        self.assertIn("vxc_basis_transform=none", self.source)
        self.assertIn("derived_vxcs_aliases=none", self.source)
        self.assertIn("pinned_abacus_si_k444_symmetry_bundle_v4", self.source)

    def test_copies_only_regular_frozen_input_files(self) -> None:
        self.assertIn(
            'find "$producer/producer-inputs-v2" -maxdepth 1 -type f -print0',
            self.source,
        )
        self.assertNotIn(
            'cp --reflink=auto "$producer/producer-inputs-v2/"*',
            self.source,
        )

    def test_hashes_source_and_copied_dataset(self) -> None:
        self.assertIn("SOURCE_ARTIFACT_SHA256SUMS.txt", self.source)
        self.assertIn("DATASET_SHA256SUMS.txt", self.source)
        self.assertIn("SOURCE_COPY_VERIFICATION.txt", self.source)
        self.assertIn("copied source artifacts are SHA256-identical", self.source)

    def test_freezes_charge_restart_for_the_nscf_extension(self) -> None:
        self.assertGreaterEqual(
            self.source.count("ABACUS-CHARGE-DENSITY.restart"), 2
        )
        self.assertIn("scf_charge_restart=frozen_in_dataset", self.source)


if __name__ == "__main__":
    unittest.main()
