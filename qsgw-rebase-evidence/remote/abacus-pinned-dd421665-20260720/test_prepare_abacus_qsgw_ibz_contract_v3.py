#!/usr/bin/env python3
"""Unit tests for the state-basis ABACUS QSGW contract generator."""

from __future__ import annotations

import unittest
from pathlib import Path
from unittest import mock

import prepare_abacus_qsgw_ibz_contract_v3 as contract


class CapturingOutput:
    def __init__(self) -> None:
        self.text = ""

    def write_text(self, text: str, encoding: str) -> None:
        self.text = text


class NativeVxcStateContractTests(unittest.TestCase):
    def test_reads_native_comment_dimensions(self) -> None:
        content = "\n".join(
            [
                "# ionic step 0",
                "# rows 44",
                "# columns 44",
                "Row 1",
                " (1.0,0.0)",
            ]
        )
        with mock.patch.object(Path, "read_text", return_value=content):
            self.assertEqual(
                contract.read_abacus_native_vxc_dimension(Path("vxck1_nao.txt")),
                44,
            )

    def test_manifest_binds_native_vxck_as_fixed_state_matrices(self) -> None:
        matrices = [Path("vxck1_nao.txt"), Path("vxck2_nao.txt")]
        output = CapturingOutput()
        with (
            mock.patch.object(Path, "glob", return_value=[]),
            mock.patch.object(contract, "indexed_files", return_value=matrices),
            mock.patch.object(
                contract, "read_abacus_native_vxc_dimension", return_value=3
            ),
            mock.patch.object(contract, "sha256_file", return_value="a" * 64),
        ):
            selected = contract.write_vxc_manifest(
                Path("dataset"),
                output,
                [(0.0, 0.0, 0.0), (0.5, 0.0, 0.0)],
                n_spins=1,
                n_bands=3,
            )

        self.assertEqual(selected, matrices)
        self.assertIn("basis state", output.text)
        self.assertIn("gauge mf0_state", output.text)
        self.assertIn("1 1 0 0 0 3 3", output.text)
        self.assertIn("vxck1_nao.txt", output.text)
        self.assertIn("vxck2_nao.txt", output.text)
        self.assertNotIn("vxcs1k", output.text)

    def test_manifest_rejects_matrix_dimension_different_from_n_bands(self) -> None:
        with (
            mock.patch.object(Path, "glob", return_value=[]),
            mock.patch.object(
                contract,
                "indexed_files",
                return_value=[Path("vxck1_nao.txt")],
            ),
            mock.patch.object(
                contract, "read_abacus_native_vxc_dimension", return_value=4
            ),
        ):
            with self.assertRaisesRegex(ValueError, "state dimension"):
                contract.write_vxc_manifest(
                    Path("dataset"),
                    CapturingOutput(),
                    [(0.0, 0.0, 0.0)],
                    n_spins=1,
                    n_bands=3,
                )

    def test_manifest_rejects_derived_legacy_aliases(self) -> None:
        with mock.patch.object(
            Path, "glob", return_value=[Path("vxcs1k1_nao.txt")]
        ):
            with self.assertRaisesRegex(ValueError, "refuses derived vxcs1k aliases"):
                contract.write_vxc_manifest(
                    Path("dataset"),
                    CapturingOutput(),
                    [(0.0, 0.0, 0.0)],
                    n_spins=1,
                    n_bands=1,
                )


if __name__ == "__main__":
    unittest.main()
