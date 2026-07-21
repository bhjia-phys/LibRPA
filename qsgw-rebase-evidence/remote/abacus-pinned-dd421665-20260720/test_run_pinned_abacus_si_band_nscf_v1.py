#!/usr/bin/env python3
"""Static contract tests for the pinned ABACUS NSCF Slurm runner."""

from __future__ import annotations

import unittest
from pathlib import Path


class PinnedNscfSlurmContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = (
            Path(__file__).with_name("run_pinned_abacus_si_band_nscf_v1.slurm")
            .read_text(encoding="utf-8")
        )

    def test_consumes_only_completed_immutable_scf_bundle_v3(self) -> None:
        self.assertIn("symmetry-bundle-20260720-v3", self.source)
        self.assertIn('test -e "$scf_bundle/COMPLETE"', self.source)
        self.assertIn('test ! -e "$scf_bundle/FAILED"', self.source)
        self.assertIn("sha256sum --check --quiet OUTPUT_SHA256SUMS.txt", self.source)
        self.assertIn('test -z "$(find "$scf_bundle" -perm /222 -print -quit)"', self.source)

    def test_stages_frozen_charge_into_fresh_abacus_output(self) -> None:
        self.assertIn("ABACUS-CHARGE-DENSITY.restart", self.source)
        self.assertIn('test ! -e "$run_dir/OUT.ABACUS"', self.source)
        self.assertIn('mkdir "$run_dir/OUT.ABACUS"', self.source)
        self.assertIn('cp "$source_charge" "$staged_charge"', self.source)
        self.assertIn("charge_sha256_before", self.source)
        self.assertIn("charge_sha256_after", self.source)
        self.assertIn("SCF charge restart changed during NSCF", self.source)

    def test_uses_pinned_inputs_binary_and_validator(self) -> None:
        self.assertIn("INPUT_nscf_band", self.source)
        self.assertIn("KPT_band", self.source)
        self.assertIn(
            "a2676c36e318da831339cfb147c6027f3b7aed0b40925fa4a6086a947a403b17",
            self.source,
        )
        self.assertIn("validate_abacus_si_band_nscf_output_v1.py", self.source)
        self.assertIn("--expected-charge-sha256", self.source)
        self.assertIn('assert report["n_kpoints"] == 143', self.source)

    def test_marks_complete_only_after_hashing_and_makes_bundle_immutable(self) -> None:
        validation = self.source.index("NSCF_OUTPUT_VALIDATION.json")
        manifest = self.source.index("ARTIFACT_SHA256SUMS.txt")
        complete = self.source.index('touch "$root/COMPLETE"')
        immutable = self.source.index('find "$root" -type f -exec chmod a-w')
        self.assertLess(validation, manifest)
        self.assertLess(manifest, complete)
        self.assertLess(complete, immutable)
        self.assertIn('test -z "$(find "$root" -type l -print -quit)"', self.source)


if __name__ == "__main__":
    unittest.main()
