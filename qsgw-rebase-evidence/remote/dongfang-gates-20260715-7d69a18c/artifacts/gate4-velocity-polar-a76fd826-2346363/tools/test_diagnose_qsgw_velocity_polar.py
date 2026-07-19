#!/usr/bin/env python3

import sys
import unittest
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from diagnose_qsgw_velocity_polar import (
    nearest_unitary,
    transform_velocity,
    unitarity_max_abs,
)


class PolarTransformTests(unittest.TestCase):
    def test_exact_unitary_is_unchanged(self) -> None:
        inverse_sqrt_two = 1.0 / np.sqrt(2.0)
        transform = np.array(
            [
                [inverse_sqrt_two, 1j * inverse_sqrt_two],
                [1j * inverse_sqrt_two, inverse_sqrt_two],
            ],
            dtype=np.complex128,
        )

        polar, singular_values = nearest_unitary(transform)

        self.assertTrue(np.allclose(polar, transform, atol=1.0e-15, rtol=0.0))
        self.assertTrue(np.allclose(singular_values, 1.0, atol=1.0e-15, rtol=0.0))
        self.assertLess(unitarity_max_abs(polar), 1.0e-15)

    def test_projection_removes_small_nonunitary_perturbation(self) -> None:
        transform = np.array(
            [[1.0 + 2.0e-10, 3.0e-11j], [4.0e-11j, 1.0 - 1.0e-10]],
            dtype=np.complex128,
        )

        polar, singular_values = nearest_unitary(transform)

        self.assertGreater(unitarity_max_abs(transform), 1.0e-10)
        self.assertLess(unitarity_max_abs(polar), 1.0e-14)
        self.assertGreater(float(np.max(np.abs(singular_values - 1.0))), 1.0e-10)

    def test_velocity_transform_uses_qsgw_storage_convention(self) -> None:
        transform = np.array(
            [[0.0, 1.0j], [1.0j, 0.0]], dtype=np.complex128
        )
        reference_velocity = np.array(
            [[1.0, 2.0 + 3.0j], [2.0 - 3.0j, 4.0]],
            dtype=np.complex128,
        )
        source_velocity = (
            transform.conj() @ reference_velocity @ transform.T
        )

        restored = transform_velocity(transform, source_velocity)

        self.assertTrue(
            np.allclose(restored, reference_velocity, atol=1.0e-15, rtol=0.0)
        )


if __name__ == "__main__":
    unittest.main()
