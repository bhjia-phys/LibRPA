#!/usr/bin/env python3
"""Tests for rebuilding physical LibRPA lattice rows from ABACUS STRU."""

from __future__ import annotations

import importlib.util
import math
import tempfile
import unittest
from pathlib import Path


SOURCE = Path(__file__).with_name("build_abacus_physical_stru_overlay_v1.py")


def load_module():
    spec = importlib.util.spec_from_file_location("physical_stru_overlay", SOURCE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {SOURCE}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


INPUT_STRU = """ATOMIC_SPECIES
Si 1.000 Si.upf

LATTICE_CONSTANT
10.2

LATTICE_VECTORS
0.0 0.5 0.5
0.5 0.0 0.5
0.5 0.5 0.0

ATOMIC_POSITIONS
Direct

Si
0.0
2
0.0 0.0 0.0 0 0 0
0.25 0.25 0.25 0 0 0
"""


def raw_stru(second_atom: float = 2.55, translation: float = 0.25) -> str:
    return "\n".join(
        (
            "0.0 0.9448634388871776 0.9448634388871776",
            "0.9448634388871776 0.0 0.9448634388871776",
            "0.9448634388871776 0.9448634388871776 0.0",
            "-3.324917151297372 3.324917151297372 3.324917151297372",
            "3.324917151297372 -3.324917151297372 3.324917151297372",
            "3.324917151297372 3.324917151297372 -3.324917151297372",
            "2",
            "0.0 0.0 0.0 1",
            f"{second_atom} {second_atom} {second_atom} 1",
            "2 row",
            "1 0 0 0 1 0 0 0 1 0.0 0.0 0.0",
            f"-1 0 0 0 -1 0 0 0 -1 {translation} {translation} {translation}",
            "",
        )
    )


class BuildPhysicalStruOverlayTests(unittest.TestCase):
    def test_rebuilds_lattice_and_reciprocal_while_preserving_payload(self) -> None:
        module = load_module()
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temporary:
            root = Path(temporary)
            input_stru = root / "STRU"
            source = root / "stru_out.raw"
            output = root / "stru_out"
            report = root / "report.json"
            input_stru.write_text(INPUT_STRU, encoding="utf-8")
            source.write_text(raw_stru(), encoding="utf-8")

            result = module.build_overlay(input_stru, source, output, report)

            self.assertTrue(result["passed"])
            self.assertEqual(result["symmetry_operation_count"], 2)
            self.assertAlmostEqual(result["lattice_constant_bohr"], 10.2)
            self.assertLess(result["atom_cartesian_max_abs_bohr"], 1.0e-12)
            self.assertLess(result["reciprocal_closure_max_abs"], 1.0e-12)
            self.assertLess(result["atom_symmetry_max_abs"], 1.0e-12)
            source_lines = source.read_text(encoding="utf-8").splitlines()
            output_lines = output.read_text(encoding="utf-8").splitlines()
            self.assertEqual(source_lines[6:], output_lines[6:])
            lattice = [list(map(float, line.split())) for line in output_lines[:3]]
            self.assertEqual(lattice, [[0.0, 5.1, 5.1], [5.1, 0.0, 5.1], [5.1, 5.1, 0.0]])
            reciprocal = [list(map(float, line.split())) for line in output_lines[3:6]]
            expected = math.pi / 5.1
            self.assertAlmostEqual(reciprocal[0][0], -expected, places=14)
            self.assertAlmostEqual(reciprocal[0][1], expected, places=14)
            self.assertTrue(report.is_file())

    def test_rejects_atom_coordinates_inconsistent_with_input_stru(self) -> None:
        module = load_module()
        with self.assertRaisesRegex(ValueError, "Cartesian atom coordinates"):
            module.build_text(INPUT_STRU, raw_stru(second_atom=2.5))

    def test_rejects_symmetry_operations_that_break_atom_mapping(self) -> None:
        module = load_module()
        with self.assertRaisesRegex(ValueError, "atom mapping"):
            module.build_text(INPUT_STRU, raw_stru(translation=0.2))


if __name__ == "__main__":
    unittest.main()
