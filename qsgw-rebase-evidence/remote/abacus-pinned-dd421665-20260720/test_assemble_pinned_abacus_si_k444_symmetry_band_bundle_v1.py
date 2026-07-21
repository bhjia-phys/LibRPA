#!/usr/bin/env python3
"""Static contract tests for the combined SCF plus band dataset assembler."""

from __future__ import annotations

import unittest
from pathlib import Path


class CombinedBandBundleAssemblyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = (
            Path(__file__).with_name(
                "assemble_pinned_abacus_si_k444_symmetry_band_bundle_v1.slurm"
            )
            .read_text(encoding="utf-8")
        )

    def test_requires_completed_readonly_scf_and_nscf_bundles(self) -> None:
        self.assertIn("symmetry-bundle-20260720-v3", self.source)
        self.assertIn("si-band-nscf-20260720-v1", self.source)
        self.assertGreaterEqual(self.source.count("/COMPLETE"), 2)
        self.assertGreaterEqual(self.source.count("-perm /222"), 3)
        self.assertIn("sha256sum --check --quiet OUTPUT_SHA256SUMS.txt", self.source)
        self.assertIn("sha256sum --check --quiet ARTIFACT_SHA256SUMS.txt", self.source)

    def test_runs_validated_preprocessor_and_band_contract_generator(self) -> None:
        self.assertIn("preprocess_abacus_band_for_librpa_v2.py", self.source)
        self.assertIn("prepare_abacus_qsgw_band_contract_v2.py", self.source)
        self.assertIn("NSCF_OUTPUT_VALIDATION.json", self.source)
        self.assertIn("qsgw_vxc_band.manifest", self.source)
        self.assertIn("qsgw_band_input.contract", self.source)

    def test_freezes_exact_143_point_reader_file_sets(self) -> None:
        self.assertIn("band_KS_eigenvalue_k_*.txt", self.source)
        self.assertIn("band_KS_eigenvector_k_*.txt", self.source)
        self.assertIn("band_vxc_k_*.txt", self.source)
        self.assertIn("band_vxck*_nao.txt", self.source)
        self.assertGreaterEqual(self.source.count("-eq 143"), 4)
        self.assertIn('"qsgw_full_vxc_copy": "byte_identical"', self.source)
        self.assertIn('"band_vxc_basis": "state"', self.source)

    def test_preserves_disabled_contract_and_adds_band_contract(self) -> None:
        self.assertIn('test -f "$dataset/qsgw_input.contract"', self.source)
        self.assertIn("grep -Fqx 'band_update off'", self.source)
        self.assertIn("grep -Fqx 'n_band_kpoints 0'", self.source)
        self.assertIn("grep -Fqx 'band_update operator_fourier'", self.source)
        self.assertIn("grep -Fqx 'n_band_kpoints 143'", self.source)

    def test_writes_qsgw_runtime_inputs_with_matching_contracts(self) -> None:
        self.assertIn("librpa-qsgw-grid.in", self.source)
        self.assertIn("librpa-qsgw-band.in", self.source)
        self.assertIn("task = qsgw_band", self.source)
        self.assertIn("qsgw_input_contract = qsgw_band_input.contract", self.source)
        self.assertIn("replace_w_head = false", self.source)
        self.assertIn("use_pyatb = false", self.source)
        self.assertNotIn("qsgw_iterative_headwing", self.source)
        self.assertIn("qsgw_update_hartree = false", self.source)

    def test_hashes_and_makes_final_bundle_immutable(self) -> None:
        self.assertIn("DATASET_SHA256SUMS.txt", self.source)
        self.assertIn("OUTPUT_SHA256SUMS.txt", self.source)
        self.assertIn('touch "$root/COMPLETE"', self.source)
        self.assertIn('find "$root" -type f -exec chmod a-w', self.source)
        self.assertIn('test -z "$(find "$root" -type l -print -quit)"', self.source)


if __name__ == "__main__":
    unittest.main()
