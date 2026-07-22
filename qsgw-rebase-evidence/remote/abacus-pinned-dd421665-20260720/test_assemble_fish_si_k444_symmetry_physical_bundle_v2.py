#!/usr/bin/env python3
"""Regression tests for the provenance-complete symmetry bundle v2."""

from __future__ import annotations

import unittest
from pathlib import Path


class SymmetryPhysicalBundleV2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = (
            Path(__file__)
            .with_name("assemble_fish_si_k444_symmetry_physical_bundle_v2.sh")
            .read_text(encoding="utf-8")
        )

    def test_uses_new_immutable_root_and_runner_identity(self) -> None:
        self.assertIn(
            "symmetry-physical-bundle-20260723-v2", self.source
        )
        self.assertIn(
            "assemble_fish_si_k444_symmetry_physical_bundle_v2.sh", self.source
        )
        self.assertIn('test ! -e "$root"', self.source)
        self.assertNotIn(
            "root=$base/abacus-pinned-dd421665-si-k444-symmetry-physical-bundle-20260723-v1",
            self.source,
        )

    def test_records_every_symmetry_runner_preflight_fact(self) -> None:
        for exact_line in (
            "grid=4x4x4",
            "scf_kpoints=8",
            "full_bz_kpoints=64",
            "use_shrink_abfs=false",
            "symmetry=on",
            "headwing=off",
            "hartree=off",
            "band_update=off",
            "physical_lattice_source=matching_abacus_input_STRU",
            "shared_gw_source_changes=none",
        ):
            self.assertIn(exact_line, self.source)

    def test_retains_all_v1_data_and_integrity_guards(self) -> None:
        for token in (
            "build_abacus_physical_stru_overlay_v1.py",
            "prepare_abacus_qsgw_ibz_contract_v3.py",
            "SOURCE_REUSE_VERIFICATION.txt",
            "DATASET_SHA256SUMS.txt",
            "OUTPUT_SHA256SUMS.txt",
            "PHYSICAL_STRU_OVERLAY.json",
            "COMPLETE",
            "FAILED",
        ):
            self.assertIn(token, self.source)
        self.assertNotRegex(self.source, r"(^|\s)rm\s")
        self.assertNotIn("ln -s", self.source)


if __name__ == "__main__":
    unittest.main()
