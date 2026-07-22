#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
MODULE_PATH = HERE / "compare_exact847_checkpoints_v1.py"
SPEC = importlib.util.spec_from_file_location("checkpoint_under_test", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class Exact847CheckpointComparisonTests(unittest.TestCase):
    def test_lower_triangle_noise_does_not_break_scheme_a_parity(self):
        reference = np.asarray(
            [[[[1.0, 0.2 + 0.3j], [9.0 - 2.0j, 2.0]]]], dtype=complex
        )
        observed = reference.copy()
        observed[0, 0, 1, 0] = -7.0 + 8.0j
        report = MODULE.analyze(reference, observed)
        self.assertGreater(report["raw_full_matrix"]["max_abs_ha"], 1.0)
        self.assertEqual(
            report["upper_triangle_hermitized"]["max_abs_ha"], 0.0
        )
        self.assertEqual(report["eigenvalue_max_abs_ha"], 0.0)
        self.assertTrue(report["numerical_parity_passed"])

    def test_upper_triangle_difference_fails_parity(self):
        reference = np.eye(2, dtype=complex).reshape(1, 1, 2, 2)
        observed = reference.copy()
        observed[0, 0, 0, 1] = 1.0e-3
        report = MODULE.analyze(reference, observed)
        self.assertFalse(report["numerical_parity_passed"])
        self.assertAlmostEqual(
            report["upper_triangle_hermitized"]["max_abs_ha"], 1.0e-3
        )


if __name__ == "__main__":
    unittest.main()
