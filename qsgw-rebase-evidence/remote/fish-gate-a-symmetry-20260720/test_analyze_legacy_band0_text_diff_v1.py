#!/usr/bin/env python3
from __future__ import annotations

import unittest

from analyze_legacy_band0_text_diff_v1 import (
    band_edges,
    compare_rows,
    parse_band_text,
)


REFERENCE = """
1 0.0 0.0 0.0 1.0 -1.0 0.0 1.0
2 0.5 0.0 0.0 1.0 -0.5 0.0 0.8
"""

OBSERVED = """
1 0.0 0.0 0.0 1.0 -0.9 0.0 1.2
2 0.5 0.0 0.0 1.0 -0.5 0.0 0.9
"""


class BandTextAnalysisTests(unittest.TestCase):
    def test_parser_and_edges(self) -> None:
        rows = parse_band_text(REFERENCE, "reference")
        self.assertEqual(len(rows), 2)
        self.assertEqual(len(rows[0].energies_ev), 2)
        edges = band_edges(rows)
        self.assertEqual(edges["vbm_ev"], -0.5)
        self.assertEqual(edges["cbm_ev"], 0.8)
        self.assertAlmostEqual(edges["gap_ev"], 1.3)

    def test_difference_location_and_gap(self) -> None:
        report = compare_rows(
            parse_band_text(REFERENCE, "reference"),
            parse_band_text(OBSERVED, "observed"),
        )
        self.assertAlmostEqual(report["energy_max_abs_diff_ev"], 0.2)
        self.assertEqual(report["energy_max_location"], {"kpoint": 1, "band": 2})
        self.assertAlmostEqual(report["gap_abs_diff_ev"], 0.1)
        self.assertAlmostEqual(report["per_band"][0]["max_abs_diff_ev"], 0.1)


if __name__ == "__main__":
    unittest.main()
