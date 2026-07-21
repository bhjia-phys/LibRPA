#!/usr/bin/env python3
from __future__ import annotations

import unittest

import numpy as np

from compare_legacy_h0_candidate_trace_v1 import (
    compare_iteration,
    hermitize_legacy_upper,
    parse_iterations,
)


def synthetic_case() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    legacy = np.asarray(
        [[[[1.0 + 0.0j, 0.1 + 0.02j],
           [0.1 - 0.02000005j, 2.0 + 0.0j]]]],
        dtype=np.complex128,
    )
    candidate = hermitize_legacy_upper(legacy.copy())
    rotation = np.asarray([[[[1.0, 0.0], [0.0, 1.0]]]], dtype=np.complex128)
    eigenvalues = np.linalg.eigvalsh(legacy[0, 0], UPLO="U").reshape(1, 1, 2)
    return legacy, candidate, rotation, eigenvalues


class LegacyCandidateComparisonTests(unittest.TestCase):
    def test_iteration_parser(self) -> None:
        self.assertEqual(parse_iterations("1:3"), [1, 2, 3])
        self.assertEqual(parse_iterations("3,1,3"), [1, 3])

    def test_legacy_raw_gap_is_separate_from_candidate_acceptance(self) -> None:
        legacy, candidate, rotation, eigenvalues = synthetic_case()
        report = compare_iteration(
            legacy, candidate, rotation, eigenvalues, occupied_bands=1
        )
        self.assertTrue(report["legacy_oracle_invariant_gap"])
        self.assertTrue(report["parity_passed"])
        self.assertTrue(report["candidate_invariants_passed"])
        self.assertTrue(report["passed"])

    def test_matrix_parity_failure_is_rejected(self) -> None:
        legacy, candidate, rotation, eigenvalues = synthetic_case()
        candidate[0, 0, 0, 0] += 2.0e-6
        report = compare_iteration(
            legacy, candidate, rotation, eigenvalues, occupied_bands=1
        )
        self.assertFalse(report["parity_passed"])
        self.assertFalse(report["passed"])

    def test_rotation_invariant_failure_is_rejected(self) -> None:
        legacy, candidate, rotation, eigenvalues = synthetic_case()
        rotation[0, 0, 0, 0] = 1.01
        report = compare_iteration(
            legacy, candidate, rotation, eigenvalues, occupied_bands=1
        )
        self.assertTrue(report["parity_passed"])
        self.assertFalse(report["candidate_invariants_passed"])
        self.assertFalse(report["passed"])


if __name__ == "__main__":
    unittest.main()
