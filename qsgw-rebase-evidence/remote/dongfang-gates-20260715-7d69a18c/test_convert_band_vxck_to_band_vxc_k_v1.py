#!/usr/bin/env python3
"""Tests for the band_vxck -> band_vxc_k converter."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from convert_band_vxck_to_band_vxc_k_v1 import (  # noqa: E402
    ConvertError,
    convert_file,
    parse_band_vxck,
)


def _vxck(values: list[list[complex]], n: int) -> str:
    lines = [
        "#------------------------------------------------------------------------\n",
        "# ionic step 0\n",
        "# filename OUT.ABACUS/vxck1_nao.txt\n",
        "# gamma only 0\n",
        f"# rows {n}\n",
        f"# columns {n}\n",
        "#------------------------------------------------------------------------\n",
    ]
    for index, row in enumerate(values, 1):
        lines.append(f"Row {index}\n")
        for start in range(0, len(row), 8):
            chunk = row[start : start + 8]
            lines.append(
                " " + " ".join(
                    f"({value.real:.8e},{value.imag:.8e})" for value in chunk
                )
                + "\n"
            )
    return "".join(lines)


TRIANGLE = [
    [complex(-8.00126971e-01, 0.0), complex(1.0e-16, 0.0)],
    [complex(-3.76140374e-01, 0.0)],
]


class ParseTests(unittest.TestCase):
    def test_triangle_layout(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "band_vxck1_nao.txt"
            path.write_text(_vxck(TRIANGLE, 2), encoding="ascii")
            diagonal, n_rows, n_columns = parse_band_vxck(path)
            self.assertEqual((n_rows, n_columns), (2, 2))
            self.assertEqual(diagonal, [-8.00126971e-01, -3.76140374e-01])

    def test_wrong_row_length_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "band_vxck1_nao.txt"
            path.write_text(
                _vxck(
                    [TRIANGLE[0], TRIANGLE[1] + [complex(1.0, 0.0)]], 2
                ),
                encoding="ascii",
            )
            with self.assertRaisesRegex(ConvertError, "triangular layout"):
                parse_band_vxck(path)

    def test_imaginary_diagonal_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "band_vxck1_nao.txt"
            bad = [
                [complex(-8.00126971e-01, 1.0e-10), complex(1.0e-16, 0.0)],
                [complex(-3.76140374e-01, 0.0)],
            ]
            path.write_text(_vxck(bad, 2), encoding="ascii")
            with self.assertRaisesRegex(ConvertError, "imaginary part"):
                parse_band_vxck(path)


class ConvertTests(unittest.TestCase):
    def test_diagonal_ry_to_ha(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "band_vxck1_nao.txt"
            target = Path(temporary) / "band_vxc_k_00001.txt"
            source.write_text(_vxck(TRIANGLE, 2), encoding="ascii")
            result = convert_file(source, target)
            self.assertEqual(result["n_bands"], 2)
            self.assertEqual(result["rows_written"], 2)
            lines = result["content"].splitlines()
            self.assertEqual(len(lines), 2)
            fields = lines[0].split()
            self.assertEqual((fields[0], fields[1]), ("1", "1"))
            self.assertAlmostEqual(float(fields[2]), -4.000634855e-01, places=10)
            fields = lines[1].split()
            self.assertAlmostEqual(float(fields[2]), -1.88070187e-01, places=10)

    def test_reference_pair_matches_may_value(self) -> None:
        # May 2026 reference: diag Row1 -8.00126971e-01 Ry -> -4.0006348548059606E-01 Ha
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "band_vxck1_nao.txt"
            target = Path(temporary) / "band_vxc_k_00001.txt"
            source.write_text(
                "# header\n# rows 2\n# columns 2\nRow 1\n"
                " (-8.00126971e-01,7.40038581e-32) (1.0e-16,0.0e+00)\n"
                "Row 2\n (-3.76140374e-01,0.0e+00)\n",
                encoding="ascii",
            )
            result = convert_file(source, target)
            value = float(result["content"].splitlines()[0].split()[2])
            self.assertAlmostEqual(value, -4.0006348548059606e-01, places=9)


if __name__ == "__main__":
    unittest.main()
