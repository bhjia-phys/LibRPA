#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent
MODULE_PATH = HERE / "prepare_abacus_qsgw_fullbz_contract_hartree_v32.py"
SPEC = importlib.util.spec_from_file_location("hartree_contract", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def touch(root: Path, *names: str) -> None:
    for name in names:
        (root / name).write_text(name + "\n", encoding="ascii")


def grouped_roles(records):
    result = {}
    for role, path in records:
        result.setdefault(role, []).append(path.name)
    return result


class HartreeContractRoleTest(unittest.TestCase):
    def test_full_basis_roles_match_full_reader(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            touch(
                root,
                "Cs_data_0.txt",
                "Cs_shrinked_data_0.txt",
                "coulomb_cut_0.txt",
                "basis_aux_out",
            )
            roles = grouped_roles(
                MODULE.hartree_role_files(root, False, "truncated")
            )
            self.assertEqual(
                roles["hartree_ri_coefficients"], ["Cs_data_0.txt"]
            )
            self.assertEqual(
                roles["hartree_aux_basis"], ["basis_aux_out"]
            )

    def test_shrink_roles_fall_back_to_shrink_ri_basis_sources(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            touch(
                root,
                "Cs_data_0.txt",
                "Cs_shrinked_data_0.txt",
                "Cs_shrinked_data_1.txt",
                "coulomb_cut_0.txt",
                "basis_aux_out",
            )
            roles = grouped_roles(
                MODULE.hartree_role_files(root, True, "truncated")
            )
            expected = ["Cs_shrinked_data_0.txt", "Cs_shrinked_data_1.txt"]
            self.assertEqual(roles["hartree_ri_coefficients"], expected)
            self.assertEqual(roles["hartree_aux_basis"], expected)

    def test_explicit_shrink_auxiliary_basis_takes_precedence(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            touch(
                root,
                "Cs_shrinked_data_0.txt",
                "coulomb_mat_0.txt",
                "basis_aux_shrink_out",
            )
            roles = grouped_roles(
                MODULE.hartree_role_files(root, True, "full")
            )
            self.assertEqual(
                roles["hartree_aux_basis"], ["basis_aux_shrink_out"]
            )
            self.assertEqual(
                roles["hartree_coulomb"], ["coulomb_mat_0.txt"]
            )

    def test_missing_selected_ri_set_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            touch(root, "Cs_data_0.txt", "coulomb_cut_0.txt")
            with self.assertRaisesRegex(ValueError, "RI or Coulomb"):
                MODULE.hartree_role_files(root, True, "truncated")


if __name__ == "__main__":
    unittest.main()
