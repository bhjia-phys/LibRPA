#!/usr/bin/env python3
from __future__ import annotations

import struct
import tempfile
import unittest
from pathlib import Path

from compare_legacy_band0_native_outputs_v1 import (
    hermiticity_max_abs,
    read_matz_binary,
    read_sigcrf_binary,
    vector_metrics,
)


class NativeParserTests(unittest.TestCase):
    def test_matz_round_trip(self) -> None:
        values = [1 + 0j, 2 - 3j, 2 + 3j, 4 + 0j]
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "matrix.bin"
            payload = bytearray(struct.pack("<ii", 2, 2))
            for value in values:
                payload.extend(struct.pack("<dd", value.real, value.imag))
            path.write_bytes(payload)
            rows, columns, observed = read_matz_binary(path)
        self.assertEqual((rows, columns), (2, 2))
        self.assertEqual(observed, values)
        self.assertEqual(hermiticity_max_abs(rows, columns, observed), 0.0)

    def test_sigcrf_round_trip(self) -> None:
        key = (3, 0, 1, 2, 1)
        values = [1.25 - 0.5j, -2.0 + 0.75j]
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "sigc.dat"
            payload = bytearray(struct.pack("<Q", 1))
            payload.extend(struct.pack("<QQQQQ", *key))
            for value in values:
                payload.extend(struct.pack("<dd", value.real, value.imag))
            path.write_bytes(payload)
            observed = read_sigcrf_binary(path)
        self.assertEqual(observed, {key: values})

    def test_vector_metrics(self) -> None:
        metrics = vector_metrics([1 + 0j, 2 + 0j], [1 + 0j, 2.000001 + 0j])
        self.assertAlmostEqual(metrics["max_abs"], 1.0e-6, places=14)
        self.assertGreater(metrics["relative_frobenius"], 0.0)


if __name__ == "__main__":
    unittest.main()
