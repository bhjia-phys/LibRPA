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
    "validate_qsgw_iter1_v6",
    HERE / "validate_qsgw_iter1_v6.py",
)
assert spec is not None and spec.loader is not None
observer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(observer)


class ValidatorTests(unittest.TestCase):
    def run_validate(self, paths):
        return observer.validate(
            matrix_trace=paths["matrix"],
            eigenvalue_trace=paths["eigenvalues"],
            iteration_trace=paths["iterations"],
            band_out=paths["band_out"],
            input_contract=paths["contract"],
            cmp_qsgw_path=CMP_QSGW,
        )

    def test_valid_first_update_passes(self):
        with fixture.scratch_directory(HERE) as directory:
            report = self.run_validate(fixture.write_fixture(directory))
        self.assertTrue(report["passed"])
        self.assertEqual(report["iterations"], [0, 1])
        self.assertEqual(report["qsgw_contract_version"], 6)

    def test_raw_h_closure_corruption_fails(self):
        with fixture.scratch_directory(HERE) as directory:
            report = self.run_validate(
                fixture.write_fixture(directory, raw_shift=1e-5)
            )
        self.assertFalse(report["passed"])
        self.assertGreater(report["raw_h_closure_max_abs_ha"], 1e-6)

    def test_symmetry_contract_mismatch_is_rejected(self):
        with fixture.scratch_directory(HERE) as directory:
            paths = fixture.write_fixture(
                directory, symmetry="exx_off_gw_off_rpa_off"
            )
            with self.assertRaises(observer.ValidationError):
                self.run_validate(paths)

    def test_input_contract_hash_mismatch_is_rejected(self):
        with fixture.scratch_directory(HERE) as directory:
            paths = fixture.write_fixture(directory)
            paths["contract"].write_text("changed\n", encoding="utf-8")
            with self.assertRaises(observer.ValidationError):
                self.run_validate(paths)


if __name__ == "__main__":
    unittest.main(verbosity=2)
