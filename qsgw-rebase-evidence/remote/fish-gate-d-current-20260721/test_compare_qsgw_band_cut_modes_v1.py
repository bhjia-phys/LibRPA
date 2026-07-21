#!/usr/bin/env python3

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path


REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "regression_tests" / "backend" / "comparisons"))

from compare_qsgw_band_cut_modes_v1 import (
    CutModeComparisonError,
    compare_cut_modes,
)
from test_validate_qsgw_band_v6 import make_fixture
from validate_qsgw_band_v6 import validate_band_run


class CutModeComparisonTests(unittest.TestCase):
    def setUp(self):
        fixed_root = os.environ.get("LIBRPA_QSGW_CUT_TEST_TMP")
        if fixed_root:
            self.root = Path(fixed_root)
            self.temporary = None
        else:
            self.temporary = tempfile.TemporaryDirectory()
            self.root = Path(self.temporary.name)
            for mode in (0, 1, 2):
                (self.root / "mode{}".format(mode)).mkdir()
        for mode in (0, 1, 2):
            run = self.root / "mode{}".format(mode)
            paths = make_fixture(run, mode=mode)
            report = validate_band_run(
                *paths[:3],
                run,
                paths[3],
                expected_iterations=1,
                expected_cut_mode=mode,
                expected_unoccupied_keep=0,
                expected_shift_ha=20.0,
            )
            (run / "band-validation.json").write_text(
                json.dumps(report, indent=2, sort_keys=True) + "\n",
                encoding="ascii",
            )

    def tearDown(self):
        if self.temporary is not None:
            self.temporary.cleanup()

    def compare(self):
        return compare_cut_modes(
            self.root / "mode0",
            self.root / "mode1",
            self.root / "mode2",
        )

    def test_controlled_cut_modes_pass(self):
        report = self.compare()
        self.assertTrue(report["passed"])
        self.assertTrue(report["byte_equal_cut_independent_tables"]["ks"])

    def test_cut_independent_exx_change_is_rejected(self):
        path = self.root / "mode2" / "qsgw_matrices.dat"
        text = path.read_text(encoding="ascii")
        changed = text.replace(
            "-2.00000000000000011e-01",
            "-1.90000000000000002e-01",
            1,
        )
        self.assertNotEqual(changed, text)
        path.write_text(changed, encoding="ascii")
        with self.assertRaisesRegex(CutModeComparisonError, "invariant"):
            self.compare()


if __name__ == "__main__":
    unittest.main()
