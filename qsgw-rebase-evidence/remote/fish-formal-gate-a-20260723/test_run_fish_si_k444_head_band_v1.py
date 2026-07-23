#!/usr/bin/env python3

from __future__ import annotations

import unittest
from pathlib import Path


RUNNER = Path(__file__).with_name("run_fish_si_k444_head_band_v1.sh")


class FishRunnerContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.text = RUNNER.read_text(encoding="ascii")

    def test_enforces_requested_physics_scope(self) -> None:
        self.assertIn("replace_w_head = true", self.text)
        self.assertIn("option_dielect_func = 4", self.text)
        self.assertIn("qsgw_update_hartree = false", self.text)
        self.assertIn("qsgw_mixer = none", self.text)
        self.assertNotIn("option_dielect_func = 3", self.text)
        self.assertNotIn("qsgw_mixer = linear", self.text)
        self.assertNotIn("qsgw_mixer = pulay", self.text)

    def test_requires_full_bz_k444_head_and_band_contract(self) -> None:
        self.assertGreaterEqual(
            self.text.count("grep -Fqx 'n_scf_kpoints 64'"), 3
        )
        self.assertIn("grep -Fqx 'n_headwing_kpoints 64'", self.text)
        self.assertIn("grep -Fqx 'n_band_kpoints 201'", self.text)
        self.assertIn("headwing_grid scf", self.text)
        self.assertIn("headwing_update fixed_basis_rotation", self.text)
        self.assertIn("band_update operator_fourier", self.text)

    def test_acceptance_is_only_multiround_bands_and_gap(self) -> None:
        self.assertIn("compare_qsgw_band_iterations_v1.py", self.text)
        self.assertIn("--occupied-bands 4", self.text)
        self.assertIn("--energy-tolerance-ev 1e-4", self.text)
        self.assertIn("--gap-tolerance-ev 2e-4", self.text)
        self.assertNotIn("residual_tolerance", self.text)
        self.assertNotIn("mixing_coefficient", self.text)


if __name__ == "__main__":
    unittest.main()
