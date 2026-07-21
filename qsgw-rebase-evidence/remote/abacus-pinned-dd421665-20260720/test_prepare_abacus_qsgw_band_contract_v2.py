#!/usr/bin/env python3
"""Unit tests for extending a frozen SCF contract with band inputs."""

from __future__ import annotations

import unittest

import prepare_abacus_qsgw_band_contract_v2 as contract


SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64


def scf_contract_text() -> str:
    return "\n".join(
        [
            "# librpa-qsgw-input-contract-v1",
            "producer abacus",
            "internal_energy_units hartree",
            "mf0_basis state_coefficients_in_nao",
            "mf0_gauge producer_state",
            "n_spins 1",
            "n_bands 2",
            "n_aos 3",
            "n_scf_kpoints 1",
            "n_headwing_kpoints 0",
            "n_band_kpoints 0",
            "headwing_grid disabled",
            "headwing_update none",
            "hartree_update off",
            "band_update off",
            "role sha256 file",
            f"mf0_eigenvalues {SHA_A} band_out",
            f"mf0_wavefunctions {SHA_B} KS_eigenvector_0.dat",
            f"scf_kpoints {SHA_C} bz_sampling_out",
            f"vxc_scf_manifest {SHA_A} qsgw_vxc_scf.manifest",
            f"reader_static {SHA_B} stru_out",
            "",
        ]
    )


class BandContractExtensionTests(unittest.TestCase):
    def test_band_header_order_is_basis_states_spin_kpoints(self) -> None:
        header = contract.parse_band_kpath_info_text(
            "3 2 1 2\n0 0 0\n0.5 0 0.5\n", "band_kpath_info"
        )
        self.assertEqual(header.n_basis, 3)
        self.assertEqual(header.n_states, 2)
        self.assertEqual(header.n_spins, 1)
        self.assertEqual(header.kpoints, [(0.0, 0.0, 0.0), (0.5, 0.0, 0.5)])

    def test_rendered_contract_preserves_scf_roles_and_adds_exact_band_roles(self) -> None:
        parsed = contract.parse_contract_text(scf_contract_text(), "scf-contract")
        header = contract.parse_band_kpath_info_text(
            "3 2 1 2\n0 0 0\n0.5 0 0.5\n", "band_kpath_info"
        )
        rendered = contract.render_band_contract(
            parsed,
            header,
            band_kpoints=contract.ContractRecord("band_kpoints", SHA_A, "band_kpath_info"),
            eigenvalues=[
                contract.ContractRecord(
                    "band_mf0_eigenvalues", SHA_A, "band_KS_eigenvalue_k_00001.txt"
                ),
                contract.ContractRecord(
                    "band_mf0_eigenvalues", SHA_B, "band_KS_eigenvalue_k_00002.txt"
                ),
            ],
            wavefunctions=[
                contract.ContractRecord(
                    "band_mf0_wavefunctions", SHA_A, "band_KS_eigenvector_k_00001.txt"
                ),
                contract.ContractRecord(
                    "band_mf0_wavefunctions", SHA_B, "band_KS_eigenvector_k_00002.txt"
                ),
            ],
            vxc_manifest=contract.ContractRecord(
                "vxc_band_manifest", SHA_C, "qsgw_vxc_band.manifest"
            ),
        )

        self.assertIn("n_band_kpoints 2", rendered)
        self.assertIn("band_update operator_fourier", rendered)
        self.assertIn(f"mf0_eigenvalues {SHA_A} band_out", rendered)
        lines = rendered.splitlines()
        self.assertEqual(
            sum(line.startswith("band_mf0_eigenvalues ") for line in lines), 2
        )
        self.assertEqual(
            sum(line.startswith("band_mf0_wavefunctions ") for line in lines), 2
        )
        self.assertEqual(sum(line.startswith("band_kpoints ") for line in lines), 1)
        self.assertEqual(
            sum(line.startswith("vxc_band_manifest ") for line in lines), 1
        )

    def test_rejects_dimension_mismatch_hidden_by_equal_si_dimensions(self) -> None:
        parsed = contract.parse_contract_text(scf_contract_text(), "scf-contract")
        wrong_header = contract.parse_band_kpath_info_text(
            "2 3 1 1\n0 0 0\n", "wrong-band-kpath"
        )
        with self.assertRaisesRegex(ValueError, "dimensions"):
            contract.render_band_contract(
                parsed,
                wrong_header,
                contract.ContractRecord("band_kpoints", SHA_A, "band_kpath_info"),
                [contract.ContractRecord("band_mf0_eigenvalues", SHA_A, "e.txt")],
                [contract.ContractRecord("band_mf0_wavefunctions", SHA_B, "w.txt")],
                contract.ContractRecord("vxc_band_manifest", SHA_C, "v.manifest"),
            )

    def test_rejects_scf_contract_that_already_declares_band_files(self) -> None:
        text = scf_contract_text() + f"band_kpoints {SHA_A} band_kpath_info\n"
        with self.assertRaisesRegex(ValueError, "disabled SCF contract"):
            contract.parse_contract_text(text, "contaminated-scf-contract")


if __name__ == "__main__":
    unittest.main()
