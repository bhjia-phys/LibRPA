#!/usr/bin/env python3

from __future__ import annotations

import math
import unittest

from compare_qsgw_same_contract_traces_v1 import summarize_physical_blocks


def rows(values: list[complex]) -> dict[tuple[object, ...], tuple[float, complex]]:
    return {
        (1, 0, "raw_h", 0, 0, -1, row, 0): (0.0, value)
        for row, value in enumerate(values)
    }


class PhysicalBlockMetricsTest(unittest.TestCase):
    def test_exact_block(self) -> None:
        metrics = summarize_physical_blocks(rows([1.0, 2.0]), rows([1.0, 2.0]))
        self.assertEqual(metrics["raw_h"]["max_abs_diff_ha"], 0.0)
        self.assertEqual(metrics["raw_h"]["max_relative_frobenius"], 0.0)

    def test_relative_frobenius(self) -> None:
        metrics = summarize_physical_blocks(rows([3.0, 4.0]), rows([3.0, 4.5]))
        self.assertAlmostEqual(metrics["raw_h"]["max_abs_diff_ha"], 0.5)
        self.assertAlmostEqual(metrics["raw_h"]["max_relative_frobenius"], 0.1)

    def test_zero_reference_exact(self) -> None:
        metrics = summarize_physical_blocks(rows([0.0]), rows([0.0]))
        self.assertEqual(metrics["raw_h"]["max_relative_frobenius"], 0.0)

    def test_zero_reference_difference_is_infinite(self) -> None:
        metrics = summarize_physical_blocks(rows([0.0]), rows([1.0e-12]))
        self.assertTrue(math.isinf(metrics["raw_h"]["max_relative_frobenius"]))

    def test_layout_mismatch_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "row keys differ"):
            summarize_physical_blocks(rows([1.0]), rows([1.0, 2.0]))

    def test_frequency_difference_within_tolerance(self) -> None:
        old = rows([1.0])
        new = rows([1.0])
        key = next(iter(new))
        new[key] = (5.0e-11, new[key][1])
        metrics = summarize_physical_blocks(old, new)
        self.assertEqual(metrics["raw_h"]["max_abs_diff_ha"], 0.0)


if __name__ == "__main__":
    unittest.main()
