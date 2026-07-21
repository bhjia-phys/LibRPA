#!/usr/bin/env python3

from __future__ import annotations

import unittest

import numpy as np

from audit_gate_a1_mixing_frontier_v1 import weighted_fill


class WeightedFillTest(unittest.TestCase):
    def test_insulator_with_nonuniform_weights_conserves_charge(self) -> None:
        eigenvalues = np.array([[[-1.0, 1.0], [-0.5, 2.0]]])
        result = weighted_fill(eigenvalues, np.array([0.75, 0.25]), 2.0)
        self.assertAlmostEqual(result["electron_count"], 2.0)
        self.assertAlmostEqual(result["gap_ha"], 1.5)
        self.assertFalse(result["metallic"])
        self.assertEqual(result["partial_groups"], [])

    def test_partial_frontier_is_metallic_and_conserves_charge(self) -> None:
        eigenvalues = np.array([[[-1.0, 0.0], [-0.5, 2.0]]])
        result = weighted_fill(eigenvalues, np.array([0.75, 0.25]), 2.5)
        self.assertAlmostEqual(result["electron_count"], 2.5)
        self.assertAlmostEqual(result["gap_ha"], 0.0)
        self.assertTrue(result["metallic"])
        self.assertEqual(len(result["partial_groups"]), 1)
        self.assertAlmostEqual(result["partial_groups"][0]["fraction"], 1.0 / 3.0)


if __name__ == "__main__":
    unittest.main()
