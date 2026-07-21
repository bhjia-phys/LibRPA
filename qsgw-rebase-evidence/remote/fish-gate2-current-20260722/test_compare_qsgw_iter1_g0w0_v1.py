#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import os
import unittest
from pathlib import Path

import gate2_test_fixture_v1 as fixture


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
CMP_QSGW = Path(os.environ.get(
    "CMP_QSGW", REPO / "regression_tests/backend/comparisons/cmp_qsgw.py"
))
spec = importlib.util.spec_from_file_location(
    "compare_qsgw_iter1_g0w0_v1",
    HERE / "compare_qsgw_iter1_g0w0_v1.py",
)
assert spec is not None and spec.loader is not None
observer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(observer)


class ComparatorTests(unittest.TestCase):
    def run_compare(self, paths):
        return observer.compare(
            trace=paths["matrix"],
            g0w0_directory=paths["g0w0"],
            input_contract=paths["contract"],
            cmp_qsgw_path=CMP_QSGW,
        )

    def test_exact_pair_passes(self):
        with fixture.scratch_directory(HERE) as directory:
            report = self.run_compare(fixture.write_fixture(directory))
        self.assertTrue(report["passed"])
        self.assertEqual(report["block_count"], 2)
        self.assertEqual(report["qsgw_contract_version"], 6)

    def test_small_difference_passes(self):
        with fixture.scratch_directory(HERE) as directory:
            paths = fixture.write_fixture(directory)
            fixture.write_sigc(
                paths["g0w0"] / "Sigc_fk_mn_kgrid_ispin_0_ik_0_ifreq_0.bin",
                [[0.01 + 0.02j + 5e-12, 0j], [0j, 0.03 + 0.04j]],
            )
            report = self.run_compare(paths)
        self.assertTrue(report["passed"])

    def test_large_difference_fails(self):
        with fixture.scratch_directory(HERE) as directory:
            paths = fixture.write_fixture(directory)
            fixture.write_sigc(
                paths["g0w0"] / "Sigc_fk_mn_kgrid_ispin_0_ik_0_ifreq_0.bin",
                [[0.01 + 0.02j + 2e-10, 0j], [0j, 0.03 + 0.04j]],
            )
            report = self.run_compare(paths)
        self.assertFalse(report["passed"])

    def test_contract_mismatch_is_rejected(self):
        with fixture.scratch_directory(HERE) as directory:
            paths = fixture.write_fixture(directory, symmetry="exx_off_gw_off_rpa_off")
            with self.assertRaises(observer.ComparisonError):
                self.run_compare(paths)

    def test_missing_g0w0_block_is_rejected(self):
        with fixture.scratch_directory(HERE) as directory:
            paths = fixture.write_fixture(directory)
            next(paths["g0w0"].glob("*ifreq_1.bin")).unlink()
            with self.assertRaises(observer.ComparisonError):
                self.run_compare(paths)


if __name__ == "__main__":
    unittest.main(verbosity=2)
