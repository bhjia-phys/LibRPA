#!/usr/bin/env python3
"""Tests for rebuilding ABACUS BZ Cartesian vectors on the physical lattice."""

from __future__ import annotations

import importlib.util
import math
import unittest
from pathlib import Path


SOURCE = Path(__file__).with_name("build_abacus_physical_bz_sampling_overlay_v1.py")


def load_module():
    spec = importlib.util.spec_from_file_location("physical_bz_overlay", SOURCE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {SOURCE}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def physical_stru(reciprocal_scale: float = 1.0) -> str:
    value = reciprocal_scale * math.pi / 5.1
    return "\n".join(
        (
            "0 5.1 5.1",
            "5.1 0 5.1",
            "5.1 5.1 0",
            f"{-value:.17g} {value:.17g} {value:.17g}",
            f"{value:.17g} {-value:.17g} {value:.17g}",
            f"{value:.17g} {value:.17g} {-value:.17g}",
            "2",
            "0 0 0 1",
            "2.55 2.55 2.55 1",
            "1 row",
            "1 0 0 0 1 0 0 0 1 0 0 0",
            "",
        )
    )


def source_bz(second_fraction: float = 0.25) -> str:
    return "\n".join(
        (
            "4 4 4",
            "2 2",
            "1 0.5 0 0 0 0 0 0 1 1",
            (
                "2 0.5 "
                f"{second_fraction} {second_fraction} {second_fraction} "
                "0.8312292878243472 0.8312292878243472 0.8312292878243472 2 2"
            ),
            "",
        )
    )


class PhysicalBzSamplingOverlayTests(unittest.TestCase):
    def test_rebuilds_only_cartesian_kvectors(self) -> None:
        module = load_module()
        output, report = module.build_text(physical_stru(), source_bz())
        self.assertTrue(report["passed"])
        self.assertEqual(report["grid"], [4, 4, 4])
        self.assertEqual(report["scf_kpoint_count"], 2)
        self.assertTrue(report["non_cartesian_tokens_unchanged"])
        self.assertLess(report["fractional_recovery_max_abs"], 1.0e-12)
        self.assertGreater(report["cartesian_kvector_max_abs_change"], 0.6)

        source_rows = [line.split() for line in source_bz().splitlines()]
        output_rows = [line.split() for line in output.splitlines()]
        for source_row, output_row in zip(source_rows[2:], output_rows[2:]):
            self.assertEqual(source_row[:5], output_row[:5])
            self.assertEqual(source_row[8:], output_row[8:])
        expected = math.pi / 5.1 / 4.0
        self.assertAlmostEqual(float(output_rows[3][5]), expected, places=14)
        self.assertAlmostEqual(float(output_rows[3][6]), expected, places=14)
        self.assertAlmostEqual(float(output_rows[3][7]), expected, places=14)

    def test_rejects_fractional_kpoint_off_grid(self) -> None:
        module = load_module()
        with self.assertRaisesRegex(ValueError, "BvK grid"):
            module.build_text(physical_stru(), source_bz(second_fraction=0.2))

    def test_rejects_structure_with_inconsistent_reciprocal_lattice(self) -> None:
        module = load_module()
        with self.assertRaisesRegex(ValueError, "reciprocal lattice closure"):
            module.build_text(physical_stru(reciprocal_scale=2.0), source_bz())


if __name__ == "__main__":
    unittest.main()
