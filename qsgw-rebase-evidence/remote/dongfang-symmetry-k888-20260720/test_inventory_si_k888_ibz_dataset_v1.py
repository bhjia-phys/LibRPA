#!/usr/bin/env python3

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from inventory_si_k888_ibz_dataset_v1 import collect_inventory


def write(path: Path, content: str = "fixture\n") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="ascii")


def build_stru() -> str:
    prefix = ["0"] * 18 + ["0"]
    kpoints = ["0.0"] * (29 * 3)
    mapping = [str(index % 29 + 1) for index in range(512)]
    return " ".join(prefix + ["8", "8", "8"] + kpoints + mapping) + "\n"


def build_fixture(root: Path) -> None:
    write(root / "band_out", "29 1 44 44 0.0\n")
    write(root / "stru_out", build_stru())
    write(root / "basis_wfc_out", "fixture\n")
    write(root / "basis_aux_out", "fixture\n")
    write(root / "band_kpath_info", "44 44 1 143\n")
    write(root / "pyatb_librpa_df" / "band_out", "512 1 44 44 0.0\n")
    write(root / "pyatb_librpa_df" / "k_path_info", "44 44 1 512\n")
    write(root / "pyatb_librpa_df" / "velocity_matrix")
    for index in range(29):
        write(root / "scf_librpa_root" / f"KS_eigenvector_{index}.dat")
        write(root / f"vxcs1k{index + 1}_nao.txt")
    for index in range(512):
        write(root / "pyatb_librpa_df" / f"KS_eigenvector_{index}.dat")
    for index in range(1, 144):
        write(root / f"band_KS_eigenvalue_k_{index:05d}.txt")
        write(root / f"band_KS_eigenvector_k_{index:05d}.txt")
        write(root / f"band_vxck{index}_nao.txt")
    for name in (
        "Cs_data_0.txt",
        "Cs_shrinked_data_0.txt",
        "shrink_sinvS_0.txt",
        "coulomb_mat_0.txt",
        "coulomb_cut_0.txt",
    ):
        write(root / name)


class InventoryTests(unittest.TestCase):
    def test_complete_fixture_passes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            build_fixture(root)
            inventory = collect_inventory(root)
            self.assertTrue(inventory["passed"])
            self.assertEqual(
                inventory["groups"]["headwing_wavefunctions"]["count"], 512
            )
            self.assertEqual(inventory["groups"]["band_wavefunctions"]["count"], 143)
            self.assertEqual(
                inventory["stru_sampling"]["stru_out"]["mapping_count"], 512
            )

    def test_missing_headwing_member_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            build_fixture(root)
            (root / "pyatb_librpa_df" / "KS_eigenvector_511.dat").unlink()
            inventory = collect_inventory(root)
            self.assertFalse(inventory["passed"])
            failed = {
                item["name"]
                for item in inventory["assertions"]
                if not item["passed"]
            }
            self.assertIn("headwing_wavefunctions_indices", failed)


if __name__ == "__main__":
    unittest.main()
