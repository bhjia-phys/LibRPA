#!/usr/bin/env python3
"""Unit tests for the pinned ABACUS Si k444 output observer."""

from __future__ import annotations

import unittest

import validate_abacus_si_k444_symmetry_output_v1 as observer


class NativeVxcParserTests(unittest.TestCase):
    def test_parses_modern_upper_triangle_schema(self) -> None:
        lines = [
            "#------------------------------------------------------------------------",
            "# ionic step 0",
            "# filename OUT.ABACUS/vxck1_nao.txt",
            "# gamma only 0",
            "# rows 3",
            "# columns 3",
            "#------------------------------------------------------------------------",
            "Row 1",
            " (1.0,0.0) (2.0,-3.0)",
            " (4.0,5.0)",
            "Row 2",
            " (6.0,0.0) (7.0,-8.0)",
            "Row 3",
            " (9.0,0.0)",
        ]

        summary = observer.parse_native_vxc_lines(lines, "vxck1_nao.txt", 3)

        self.assertEqual(summary["rows"], 3)
        self.assertEqual(summary["columns"], 3)
        self.assertEqual(summary["row_count"], 3)
        self.assertEqual(summary["upper_triangle_entries"], 6)

    def test_rejects_wrong_row_entry_count(self) -> None:
        lines = [
            "# rows 2",
            "# columns 2",
            "Row 1",
            " (1.0,0.0)",
            "Row 2",
            " (2.0,0.0)",
        ]

        with self.assertRaisesRegex(ValueError, "expected 2 complex entries"):
            observer.parse_native_vxc_lines(lines, "bad-vxck.txt", 2)

    def test_rejects_nonfinite_entry(self) -> None:
        lines = [
            "# rows 1",
            "# columns 1",
            "Row 1",
            " (nan,0.0)",
        ]

        with self.assertRaisesRegex(ValueError, "non-finite complex entry"):
            observer.parse_native_vxc_lines(lines, "bad-vxck.txt", 1)


if __name__ == "__main__":
    unittest.main()
