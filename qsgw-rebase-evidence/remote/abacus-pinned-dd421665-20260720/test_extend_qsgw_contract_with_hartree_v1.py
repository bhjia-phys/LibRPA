#!/usr/bin/env python3

from __future__ import annotations

import unittest

from extend_qsgw_contract_with_hartree_v1 import (
    ContractRecord,
    extend_contract,
    parse_contract,
    select_hartree_records,
)


SHA = "a" * 64


def base_contract(*, band=False, shrink=False):
    static = [
        "stru_out",
        "basis_wfc_out",
        "basis_aux_out",
        "Cs_data_0.txt",
        "Cs_data_1.txt",
        "coulomb_mat_0.txt",
        "coulomb_mat_1.txt",
        "coulomb_cut_0.txt",
        "coulomb_cut_1.txt",
    ]
    if shrink:
        static += [
            "basis_aux_shrink_out",
            "Cs_shrinked_data_0.txt",
            "Cs_shrinked_data_1.txt",
        ]
    records = [
        f"mf0_eigenvalues {SHA} band_out",
        f"mf0_wavefunctions {SHA} KS_eigenvector_0.dat",
        f"scf_kpoints {SHA} bz_sampling_out",
        f"vxc_scf_manifest {SHA} qsgw_vxc_scf.manifest",
    ]
    records += [f"reader_static {SHA} {name}" for name in static]
    if band:
        records += [
            f"band_kpoints {SHA} band_kpath_info",
            f"band_mf0_eigenvalues {SHA} band_KS_eigenvalue_k_00001.txt",
            f"band_mf0_wavefunctions {SHA} band_KS_eigenvector_k_00001.txt",
            f"vxc_band_manifest {SHA} qsgw_vxc_band.manifest",
        ]
    return "\n".join(
        [
            "# librpa-qsgw-input-contract-v1",
            "producer abacus",
            "internal_energy_units hartree",
            "mf0_basis state_coefficients_in_nao",
            "mf0_gauge producer_state",
            "n_spins 1",
            "n_bands 2",
            "n_aos 2",
            "n_scf_kpoints 1",
            "n_headwing_kpoints 0",
            f"n_band_kpoints {1 if band else 0}",
            "headwing_grid disabled",
            "headwing_update none",
            "hartree_update off",
            f"band_update {'operator_fourier' if band else 'off'}",
            "role sha256 file",
            *records,
        ]
    ) + "\n"


class HartreeContractExtensionTests(unittest.TestCase):
    def test_full_hartree_extension_preserves_band_contract(self):
        document = parse_contract(base_contract(band=True))
        additions = select_hartree_records(
            document, use_shrink_abfs=False, hartree_coulomb="full"
        )
        output = extend_contract(document, additions)
        parsed = parse_contract(output)

        self.assertIn("hartree_update delta_density", output)
        self.assertIn("band_update operator_fourier", output)
        self.assertEqual(parsed.value("n_band_kpoints"), "1")
        self.assertEqual(
            [record.file for record in additions if record.role == "hartree_coulomb"],
            ["coulomb_mat_0.txt", "coulomb_mat_1.txt"],
        )
        self.assertEqual(
            [record.file for record in additions if record.role == "hartree_aux_basis"],
            ["basis_aux_out"],
        )

    def test_truncated_shrink_extension_matches_reader_precedence(self):
        document = parse_contract(base_contract(shrink=True))
        additions = select_hartree_records(
            document, use_shrink_abfs=True, hartree_coulomb="truncated"
        )
        by_role = {}
        for record in additions:
            by_role.setdefault(record.role, []).append(record.file)
        self.assertEqual(
            by_role["hartree_ri_coefficients"],
            ["Cs_shrinked_data_0.txt", "Cs_shrinked_data_1.txt"],
        )
        self.assertEqual(
            by_role["hartree_coulomb"],
            ["coulomb_cut_0.txt", "coulomb_cut_1.txt"],
        )
        self.assertEqual(by_role["hartree_aux_basis"], ["basis_aux_shrink_out"])

    def test_extension_rejects_a_role_not_bound_by_reader_static(self):
        document = parse_contract(base_contract())
        additions = (
            ContractRecord("hartree_ri_coefficients", SHA, "Cs_data_0.txt"),
            ContractRecord("hartree_coulomb", SHA, "coulomb_mat_0.txt"),
            ContractRecord("hartree_aux_basis", SHA, "not_reader_static"),
        )
        with self.assertRaisesRegex(ValueError, "reader_static"):
            extend_contract(document, additions)

    def test_existing_hartree_contract_is_not_extended_twice(self):
        text = base_contract().replace(
            "hartree_update off", "hartree_update delta_density"
        )
        document = parse_contract(text)
        with self.assertRaisesRegex(ValueError, "hartree_update off"):
            select_hartree_records(
                document, use_shrink_abfs=False, hartree_coulomb="full"
            )

    def test_malformed_hash_and_unsafe_path_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "invalid contract file record"):
            parse_contract(base_contract().replace(SHA, "bad", 1))
        with self.assertRaisesRegex(ValueError, "invalid contract file record"):
            parse_contract(base_contract().replace("band_out", "../band_out", 1))


if __name__ == "__main__":
    unittest.main()
