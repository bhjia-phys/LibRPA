#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
MODULE_PATH = HERE / "compare_exact847_component_dump_v1.py"
SPEC = importlib.util.spec_from_file_location("comparison_under_test", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class Exact847ComponentComparisonTests(unittest.TestCase):
    def test_matching_components_have_zero_difference(self):
        base = np.asarray([[[[1.0, 2.0j], [-2.0j, 3.0]]]], dtype=complex)
        static = {component: base.copy() for component in MODULE.STATIC_COMPONENTS}
        frequencies = np.asarray([0.1, 0.5])
        sigma = np.stack([base, 2.0 * base])
        report = MODULE.analyze(
            static, static, frequencies, frequencies, sigma, sigma
        )
        self.assertTrue(report["diagnostic_complete"])
        self.assertEqual(
            report["static_components"]["vc"]["raw"]["max_abs_ha"], 0.0
        )
        self.assertEqual(report["sigma_c_iw"]["overall"]["max_abs_ha"], 0.0)

    def test_localizes_vc_and_sigma_differences(self):
        zero = np.zeros((1, 1, 2, 2), dtype=complex)
        legacy_static = {
            component: zero.copy() for component in MODULE.STATIC_COMPONENTS
        }
        candidate_static = {
            component: zero.copy() for component in MODULE.STATIC_COMPONENTS
        }
        candidate_static["vc"][0, 0, 0, 1] = 2.0 + 3.0j
        legacy_sigma = np.zeros((1, 1, 1, 2, 2), dtype=complex)
        candidate_sigma = legacy_sigma.copy()
        candidate_sigma[0, 0, 0, 1, 0] = 4.0
        report = MODULE.analyze(
            legacy_static,
            candidate_static,
            np.asarray([0.25]),
            np.asarray([0.25]),
            legacy_sigma,
            candidate_sigma,
        )
        self.assertAlmostEqual(
            report["static_components"]["vc"]["raw"]["max_abs_ha"],
            abs(2.0 + 3.0j),
        )
        self.assertEqual(report["sigma_c_iw"]["overall"]["max_abs_ha"], 4.0)

    def test_parses_candidate_sigma_rows(self):
        rows = ["# qsgw_contract_version 6"]
        for row in range(2):
            for column in range(2):
                rows.append(
                    "1 0 sigma_c_iw 0 0 0 2.5e-1 "
                    f"{row} {column} {row + column + 1}.0 0.0"
                )
        frequencies, values = MODULE.read_candidate_sigma_text(
            "\n".join(rows) + "\n", "fixture", 1, 1, 1, 1, 2
        )
        self.assertEqual(frequencies[0], 0.25)
        self.assertEqual(values[0, 0, 0, 1, 1], 3.0)


if __name__ == "__main__":
    unittest.main()
