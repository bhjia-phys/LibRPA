#!/usr/bin/env python3
"""Static tests for the physical structure and BZ symmetry bundle v3."""

from __future__ import annotations

import unittest
from pathlib import Path


class SymmetryPhysicalBundleV3Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = (
            Path(__file__)
            .with_name("assemble_fish_si_k444_symmetry_physical_bundle_v3.sh")
            .read_text(encoding="utf-8")
        )

    def test_uses_new_immutable_v3_root(self) -> None:
        self.assertIn("symmetry-physical-bundle-20260723-v3", self.source)
        self.assertIn(
            "assemble_fish_si_k444_symmetry_physical_bundle_v3.sh", self.source
        )
        self.assertIn('test ! -e "$root"', self.source)

    def test_rebuilds_structure_then_bz_cartesian_vectors(self) -> None:
        stru_call = self.source.index("build_abacus_physical_stru_overlay_v1.py")
        bz_call = self.source.index("build_abacus_physical_bz_sampling_overlay_v1.py")
        self.assertLess(stru_call, bz_call)
        self.assertIn("PHYSICAL_STRU_OVERLAY.json", self.source)
        self.assertIn("PHYSICAL_BZ_SAMPLING_OVERLAY.json", self.source)
        self.assertIn('report["non_cartesian_tokens_unchanged"] is True', self.source)
        self.assertIn('report["fractional_recovery_max_abs"] <= 1.0e-12', self.source)
        self.assertIn('report["grid"] == [4, 4, 4]', self.source)

    def test_treats_bz_sampling_as_derived_and_rebinds_contract(self) -> None:
        self.assertIn("stru_out|bz_sampling_out|basis_wfc_out", self.source)
        self.assertIn('"bz_sampling_out",', self.source)
        self.assertIn("prepare_abacus_qsgw_ibz_contract_v3.py", self.source)
        self.assertIn("DATASET_SHA256SUMS.txt", self.source)
        self.assertIn("OUTPUT_SHA256SUMS.txt", self.source)

    def test_records_physical_kvector_provenance_and_stage1_scope(self) -> None:
        for token in (
            "acceptance_scope=corrected_physical_stru_and_bz_stage1_input",
            "cartesian_kvector_source=correct_fractional_bz_coordinates_times_physical_reciprocal_lattice",
            "source_bz_sampling_out_sha256=",
            "physical_bz_sampling_out_sha256=",
            "grid=4x4x4",
            "use_shrink_abfs=false",
            "headwing=off",
            "hartree=off",
            "band_update=off",
            "shared_gw_source_changes=none",
        ):
            self.assertIn(token, self.source)

    def test_preserves_source_and_freezes_output(self) -> None:
        self.assertNotRegex(self.source, r"(^|\s)rm\s")
        self.assertNotIn("ln -s", self.source)
        self.assertIn("SOURCE_REUSE_VERIFICATION.txt", self.source)
        self.assertIn('find "$root" -type f -exec chmod a-w', self.source)
        self.assertIn('find "$root" -depth -type d -exec chmod a-w', self.source)
        self.assertIn("COMPLETE", self.source)
        self.assertIn("FAILED", self.source)


if __name__ == "__main__":
    unittest.main()
