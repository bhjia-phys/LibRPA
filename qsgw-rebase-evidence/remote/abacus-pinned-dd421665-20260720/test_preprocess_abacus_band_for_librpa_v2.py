#!/usr/bin/env python3
"""Unit tests for deterministic ABACUS-to-LibRPA band preprocessing."""

from __future__ import annotations

import struct
import unittest

import preprocess_abacus_band_for_librpa_v2 as preprocess
from validate_abacus_si_band_nscf_output_v1 import DiagonalVxcData, WfcData


class DeterministicBandPreprocessorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.wfc = WfcData(
            k_index=1,
            k_cartesian=(0.0, 0.0, 0.0),
            n_bands=2,
            n_basis=3,
            eigenvalues_ry=[-1.0, 0.5],
            occupations=[2.0, 0.0],
            coefficients=[
                complex(1.0, 0.0),
                complex(0.0, 1.0),
                complex(-0.5, 0.25),
                complex(0.1, -0.2),
                complex(0.3, -0.4),
                complex(0.5, -0.6),
            ],
        )

    def test_eigenvector_bytes_are_little_endian_band_major_complex(self) -> None:
        encoded = preprocess.encode_eigenvectors(self.wfc)
        expected = b"".join(
            struct.pack("<dd", value.real, value.imag)
            for value in self.wfc.coefficients
        )
        self.assertEqual(encoded, expected)
        self.assertEqual(len(encoded), 2 * 3 * 16)

    def test_kpath_header_is_basis_then_states_with_distinct_dimensions(self) -> None:
        text = preprocess.render_band_kpath_info(
            n_basis=3,
            n_states=2,
            n_spins=1,
            kpoints=[(0.0, 0.0, 0.0), (0.5, 0.0, 0.5)],
        )
        self.assertEqual(text.splitlines()[0].split(), ["3", "2", "1", "2"])

    def test_eigenvalue_output_uses_fixed_hartree_and_ev_conversion(self) -> None:
        self.assertEqual(preprocess.HA2EV, 27.211396)
        rows = preprocess.render_eigenvalue_text(self.wfc, spin_index=1).splitlines()
        first = rows[0].split()
        second = rows[1].split()
        self.assertEqual(first[:2], ["1", "1"])
        self.assertAlmostEqual(float(first[3]), -0.5)
        self.assertAlmostEqual(float(first[4]), -0.5 * 27.211396)
        self.assertAlmostEqual(float(second[3]), 0.25)

    def test_g0w0_diagonal_vxc_preserves_hartree_values(self) -> None:
        diagonal = DiagonalVxcData(
            n_kpoints=2,
            n_spins=1,
            n_bands=2,
            values_ha=[-0.25, 0.1, -0.2, 0.15],
        )
        rows = preprocess.render_diagonal_vxc_text(diagonal, kpoint_index=2)
        values = [float(row.split()[2]) for row in rows.splitlines()]
        self.assertEqual(values, [-0.2, 0.15])

    def test_band_manifest_marks_native_vxck_as_state_basis(self) -> None:
        text = preprocess.render_band_vxc_manifest(
            kpoints=[(0.0, 0.0, 0.0), (0.5, 0.0, 0.5)],
            n_spins=1,
            n_bands=2,
            records=[
                ("band_vxck1_nao.txt", "a" * 64),
                ("band_vxck2_nao.txt", "b" * 64),
            ],
        )
        self.assertIn("kind band", text)
        self.assertIn("units Ry", text)
        self.assertIn("basis state", text)
        self.assertIn("gauge mf0_state", text)
        self.assertIn("1 2 0.5 0 0.5 2 2", text)

    def test_source_paths_use_out_wfc_and_native_vxck(self) -> None:
        paths = preprocess.resolve_source_paths("OUT.ABACUS", 2)
        self.assertEqual(paths.wfc[0].as_posix(), "OUT.ABACUS/WFC/wfk1_nao.txt")
        self.assertEqual(paths.wfc[1].as_posix(), "OUT.ABACUS/WFC/wfk2_nao.txt")
        self.assertEqual(paths.vxc[0].as_posix(), "OUT.ABACUS/vxck1_nao.txt")
        self.assertEqual(paths.vxc[1].as_posix(), "OUT.ABACUS/vxck2_nao.txt")


if __name__ == "__main__":
    unittest.main()
