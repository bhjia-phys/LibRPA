#!/usr/bin/env python3
"""Tests for the band eigenvalue occupation normalization helper."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from normalize_band_eigenvalue_weights_v1 import (  # noqa: E402
    NormalizationError,
    normalize_file,
)


SAMPLE = (
    "       1       1      9.9502487600000005E-03     -5.2661469000000002E-02     -1.4329915732401235E+00\n"
    "       1       2      0.0000000000000000E+00      2.7644296950000002E-01      7.5223964180523808E+00\n"
)


class NormalizeTests(unittest.TestCase):
    def _write(self, directory: Path, content: str = SAMPLE) -> Path:
        path = directory / "band_KS_eigenvalue_k_00001.txt"
        path.write_text(content, encoding="ascii")
        return path

    def test_weighted_to_per_state(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = self._write(Path(temporary))
            result = normalize_file(path, 201)
            self.assertTrue(result["changed"])
            self.assertEqual(result["rows"], 2)
            lines = result["content"].splitlines()
            fields = lines[0].split()
            self.assertAlmostEqual(float(fields[2]), 2.0, places=8)
            # non-occupation columns preserved
            self.assertEqual(fields[3], "-5.2661469000000002E-02")
            self.assertEqual(fields[4], "-1.4329915732401235E+00")
            # zero weight stays zero
            self.assertEqual(lines[1].split()[2].split("E")[0], "0.0000000000000000")

    def test_already_per_state_scales_up(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            content = (
                "       1       1      2.0000000000000000E+00     -5.2661469000000002E-02     -1.4329915732401235E+00\n"
            )
            path = self._write(Path(temporary), content)
            result = normalize_file(path, 201)
            fields = result["content"].splitlines()[0].split()
            self.assertAlmostEqual(float(fields[2]), 402.0, places=9)
            self.assertGreater(result["max_per_state_weight"], 400.0)

    def test_invalid_kpoints_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = self._write(Path(temporary))
            with self.assertRaises(NormalizationError):
                normalize_file(path, 0)

    def test_wrong_column_count_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = self._write(Path(temporary), "1 1 0.5\n")
            with self.assertRaisesRegex(NormalizationError, "5 columns"):
                normalize_file(path, 201)

    def test_empty_file_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = self._write(Path(temporary), "")
            with self.assertRaisesRegex(NormalizationError, "no occupation rows"):
                normalize_file(path, 201)


if __name__ == "__main__":
    unittest.main()
