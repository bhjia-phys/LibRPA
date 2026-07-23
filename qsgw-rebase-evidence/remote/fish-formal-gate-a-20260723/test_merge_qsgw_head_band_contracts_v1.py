#!/usr/bin/env python3

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from merge_qsgw_head_band_contracts_v1 import (
    ContractError,
    merge_contracts,
    parse_contract,
)


DIGEST_A = "a" * 64
DIGEST_B = "b" * 64
DIGEST_C = "c" * 64


def contract_text(
    head_grid: str,
    head_update: str,
    n_head: int,
    band_update: str,
    n_band: int,
    extra_records: str,
) -> str:
    return (
        "# librpa-qsgw-input-contract-v1\n"
        "producer abacus\n"
        "internal_energy_units hartree\n"
        "mf0_basis state_coefficients_in_nao\n"
        "mf0_gauge producer_state\n"
        "n_spins 1\n"
        "n_bands 8\n"
        "n_aos 8\n"
        "n_scf_kpoints 64\n"
        f"n_headwing_kpoints {n_head}\n"
        f"n_band_kpoints {n_band}\n"
        f"headwing_grid {head_grid}\n"
        f"headwing_update {head_update}\n"
        "hartree_update off\n"
        f"band_update {band_update}\n"
        "role sha256 file\n"
        f"mf0_eigenvalues {DIGEST_A} band_out\n"
        f"{extra_records}"
    )


class ContractMergeTest(unittest.TestCase):
    def test_combines_head_and_band_roles(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            head_path = root / "head.contract"
            band_path = root / "band.contract"
            head_path.write_text(
                contract_text(
                    "scf",
                    "fixed_basis_rotation",
                    64,
                    "off",
                    0,
                    f"velocity_mf0 {DIGEST_B} velocity_matrix\n",
                ),
                encoding="ascii",
            )
            band_path.write_text(
                contract_text(
                    "disabled",
                    "none",
                    0,
                    "operator_fourier",
                    21,
                    (
                        f"band_mf0_eigenvalues {DIGEST_C} "
                        "band_KS_eigenvalue_1.txt\n"
                    ),
                ),
                encoding="ascii",
            )
            merged = merge_contracts(
                parse_contract(head_path), parse_contract(band_path)
            )
            self.assertIn("n_headwing_kpoints 64\n", merged)
            self.assertIn("n_band_kpoints 21\n", merged)
            self.assertIn("headwing_grid scf\n", merged)
            self.assertIn("headwing_update fixed_basis_rotation\n", merged)
            self.assertIn("band_update operator_fourier\n", merged)
            self.assertEqual(
                sum(
                    line.startswith("mf0_eigenvalues ")
                    for line in merged.splitlines()
                ),
                1,
            )
            self.assertIn("velocity_mf0 ", merged)
            self.assertIn("band_mf0_eigenvalues ", merged)

    def test_rejects_identity_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            head_path = root / "head.contract"
            band_path = root / "band.contract"
            head_path.write_text(
                contract_text(
                    "scf",
                    "fixed_basis_rotation",
                    64,
                    "off",
                    0,
                    f"velocity_mf0 {DIGEST_B} velocity_matrix\n",
                ),
                encoding="ascii",
            )
            band_path.write_text(
                contract_text(
                    "disabled",
                    "none",
                    0,
                    "operator_fourier",
                    21,
                    (
                        f"band_mf0_eigenvalues {DIGEST_C} "
                        "band_KS_eigenvalue_1.txt\n"
                    ),
                ).replace("n_bands 8\n", "n_bands 9\n"),
                encoding="ascii",
            )
            with self.assertRaises(ContractError):
                merge_contracts(
                    parse_contract(head_path), parse_contract(band_path)
                )


if __name__ == "__main__":
    unittest.main()
