#!/usr/bin/env python3
"""Unit tests for the pinned ABACUS NSCF band-output validator."""

from __future__ import annotations

import unittest

import validate_abacus_si_band_nscf_output_v1 as validator


def sample_wfc_text() -> str:
    return "\n".join(
        [
            "1 (index of k points)",
            "0.0 0.0 0.0",
            "2 (number of bands)",
            "3 (number of orbitals)",
            "1 (band)",
            "-1.0 (Ry)",
            "2.0 (Occupations)",
            "1.0 0.0 0.0 1.0 -0.5 0.25",
            "2 (band)",
            "0.5 (Ry)",
            "0.0 (Occupations)",
            "0.1 -0.2 0.3 -0.4 0.5 -0.6",
            "",
        ]
    )


class NscfBandOutputValidatorTests(unittest.TestCase):
    def test_parses_dynamic_kpoint_table(self) -> None:
        text = "\n".join(
            [
                "nkstot now = 3",
                "K-POINTS DIRECT COORDINATES",
                "KPOINTS DIRECT_X DIRECT_Y DIRECT_Z WEIGHT",
                "1 0.0 0.0 0.0 1.0",
                "2 0.25 0.0 0.25 1.0",
                "3 0.5 0.0 0.5 1.0",
            ]
        )
        self.assertEqual(
            validator.parse_kpt_info_text(text, "KPT.info"),
            [(0.0, 0.0, 0.0), (0.25, 0.0, 0.25), (0.5, 0.0, 0.5)],
        )

        with self.assertRaisesRegex(ValueError, "contiguous"):
            validator.parse_kpt_info_text(
                text.replace("2 0.25", "4 0.25"), "bad-KPT.info"
            )

    def test_parses_band_major_complex_wfc_with_distinct_dimensions(self) -> None:
        parsed = validator.parse_wfc_text(sample_wfc_text(), "wfk1_nao.txt")
        self.assertEqual(parsed.k_index, 1)
        self.assertEqual(parsed.n_bands, 2)
        self.assertEqual(parsed.n_basis, 3)
        self.assertEqual(parsed.eigenvalues_ry, [-1.0, 0.5])
        self.assertEqual(parsed.occupations, [2.0, 0.0])
        self.assertEqual(len(parsed.coefficients), 6)
        self.assertEqual(parsed.coefficients[1], 1j)
        self.assertEqual(parsed.coefficients[-1], complex(0.5, -0.6))

    def test_rejects_nonfinite_or_incomplete_wfc(self) -> None:
        with self.assertRaisesRegex(ValueError, "non-finite"):
            validator.parse_wfc_text(
                sample_wfc_text().replace("0.5 -0.6", "nan -0.6"),
                "nonfinite-wfc",
            )
        with self.assertRaisesRegex(ValueError, "coefficient count"):
            validator.parse_wfc_text(
                sample_wfc_text().replace(" 0.5 -0.6", ""),
                "short-wfc",
            )

    def test_parses_hartree_diagonal_vxc_output(self) -> None:
        text = "\n".join(
            [
                "2",
                "1",
                "2",
                "-0.25 -6.802849",
                "0.10 2.721140",
                "-0.20 -5.442279",
                "0.15 4.081709",
            ]
        )
        parsed = validator.parse_vxc_out_text(text, "vxc_out.dat")
        self.assertEqual((parsed.n_kpoints, parsed.n_spins, parsed.n_bands), (2, 1, 2))
        self.assertEqual(parsed.values_ha, [-0.25, 0.1, -0.2, 0.15])

    def test_native_vxc_dimension_is_state_dimension(self) -> None:
        text = "\n".join(
            [
                "# rows 2",
                "# columns 2",
                "Row 1",
                " (2.0,0.0) (1.0,0.5)",
                "Row 2",
                " (4.0,0.0)",
            ]
        )
        self.assertEqual(
            validator.parse_native_vxc_text(text, "vxck1_nao.txt"),
            (2, 3),
        )

    def test_indexed_names_must_be_exact_and_contiguous(self) -> None:
        self.assertEqual(
            validator.indexed_names(
                ["wfk2_nao.txt", "ignored", "wfk1_nao.txt"],
                r"wfk(\d+)_nao\.txt",
                "WFC",
            ),
            [(1, "wfk1_nao.txt"), (2, "wfk2_nao.txt")],
        )
        with self.assertRaisesRegex(ValueError, "contiguous"):
            validator.indexed_names(
                ["wfk1_nao.txt", "wfk3_nao.txt"],
                r"wfk(\d+)_nao\.txt",
                "WFC",
            )


if __name__ == "__main__":
    unittest.main()
