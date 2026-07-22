#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import math
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent
MODULE_PATH = HERE / "summarize_qsgw_trace_components_v1.py"
SPEC = importlib.util.spec_from_file_location("summary_under_test", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)
FIXTURES = HERE / "fixtures"


class TraceSummaryTests(unittest.TestCase):
    def test_groups_by_iteration_channel_and_component(self):
        report = MODULE.summarize(FIXTURES / "component-summary-valid.dat", {0, 1})
        groups = {
            (row["iteration"], row["component"]): row
            for row in report["groups"]
        }
        self.assertEqual(groups[(0, "h0")]["count"], 2)
        self.assertEqual(groups[(0, "h0")]["max_abs"], 5.0)
        self.assertTrue(
            math.isclose(groups[(0, "h0")]["frobenius"], math.sqrt(29.0))
        )
        self.assertEqual(groups[(1, "vc")]["max_abs"], 10.0)
        self.assertEqual(groups[(1, "vc")]["max_location"]["kpoint"], 2)

    def test_rejects_bad_rows(self):
        with self.assertRaisesRegex(ValueError, "expected 11 columns"):
            MODULE.summarize(FIXTURES / "component-summary-invalid.dat", {0})


if __name__ == "__main__":
    unittest.main()
