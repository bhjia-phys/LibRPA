#!/usr/bin/env python3

from __future__ import annotations

import unittest
from pathlib import Path


SOURCE = Path(__file__).with_name(
    "assemble_pinned_abacus_si_k444_symmetry_band_hartree_bundle_v1.slurm"
)


class HartreeBundleAssemblerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = SOURCE.read_text(encoding="ascii")

    def test_extends_both_grid_and_band_contracts_for_both_coulomb_routes(self):
        for name in (
            "qsgw_input.hartree-full.contract",
            "qsgw_input.hartree-truncated.contract",
            "qsgw_band_input.hartree-full.contract",
            "qsgw_band_input.hartree-truncated.contract",
        ):
            self.assertIn(name, self.source)
        self.assertIn("--hartree-coulomb full", self.source)
        self.assertIn("--hartree-coulomb truncated", self.source)

    def test_contract_roles_match_the_unshrunk_driver_route(self):
        self.assertIn("hartree_ri_coefficients", self.source)
        self.assertIn("Cs_data_", self.source)
        self.assertIn("coulomb_mat_", self.source)
        self.assertIn("coulomb_cut_", self.source)
        self.assertIn("basis_aux_out", self.source)
        self.assertNotIn("--use-shrink-abfs", self.source)

    def test_templates_keep_headwing_off_and_bind_hartree_explicitly(self):
        self.assertEqual(self.source.count("replace_w_head = false"), 2)
        self.assertEqual(self.source.count("qsgw_update_hartree = true"), 2)
        self.assertEqual(
            self.source.count("qsgw_hartree_normalization = weighted_occupations"),
            2,
        )
        self.assertIn("task = qsgw_band", self.source)
        self.assertIn("qsgw_export_hamiltonian_for_pyatb = true", self.source)

    def test_bundle_is_immutable_and_never_uses_symlink_inputs(self):
        self.assertIn("copy_mode=physical_copy_reflink_allowed_no_symlinks", self.source)
        self.assertIn('test -z "$(find "$root" -type l -print -quit)"', self.source)
        self.assertIn('find "$root" -type f -exec chmod a-w {} +', self.source)
        self.assertNotIn("ln -s", self.source)


if __name__ == "__main__":
    unittest.main()
