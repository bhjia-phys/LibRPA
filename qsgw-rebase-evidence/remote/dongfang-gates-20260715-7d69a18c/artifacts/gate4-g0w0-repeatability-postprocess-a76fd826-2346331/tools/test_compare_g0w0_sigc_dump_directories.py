#!/usr/bin/env python3

from contextlib import contextmanager
import json
import shutil
import struct
import subprocess
import sys
import unittest
import uuid
from pathlib import Path


HERE = Path(__file__).resolve().parent
TEST_TMP = HERE / "__test_tmp_directories"
TEST_TMP.mkdir(exist_ok=True)
sys.path.insert(0, str(HERE))

from compare_g0w0_sigc_dump_directories import compare_directories


@contextmanager
def test_directory():
    path = TEST_TMP / uuid.uuid4().hex
    path.mkdir()
    try:
        yield path
    finally:
        shutil.rmtree(path)


def write_matrix(path: Path, values: list[list[complex]]) -> None:
    dimension = len(values)
    payload = []
    for row in values:
        if len(row) != dimension:
            raise ValueError("test matrix must be square")
        for value in row:
            payload.extend((value.real, value.imag))
    path.write_bytes(
        struct.pack("=ii", dimension, 8)
        + struct.pack(f"={len(payload)}d", *payload)
    )


def filename(kpoint: int, frequency: int) -> str:
    return (
        "Sigc_fk_mn_kgrid_ispin_0_ik_"
        f"{kpoint}_ifreq_{frequency}.bin"
    )


class DirectoryComparisonTests(unittest.TestCase):
    def test_exact_directories_pass_and_count_exact_blocks(self) -> None:
        with test_directory() as tmp:
            reference = tmp / "reference"
            candidate = tmp / "candidate"
            reference.mkdir()
            candidate.mkdir()
            matrix = [[1.0 + 2.0j, 0.5j], [-0.5j, 3.0 + 0.0j]]
            for directory in (reference, candidate):
                write_matrix(directory / filename(0, 0), matrix)
                write_matrix(directory / filename(1, 0), matrix)

            report = compare_directories(
                reference,
                candidate,
                max_abs_tolerance_ha=0.0,
                relative_frobenius_tolerance=0.0,
            )

        self.assertTrue(report["passed"])
        self.assertEqual(report["block_count"], 2)
        self.assertEqual(report["exact_block_count"], 2)
        self.assertEqual(report["exact_element_count"], 8)
        self.assertEqual(report["max_abs_difference_ha"], 0.0)

    def test_reports_global_maximum_values_location_and_percentiles(self) -> None:
        with test_directory() as tmp:
            reference = tmp / "reference"
            candidate = tmp / "candidate"
            reference.mkdir()
            candidate.mkdir()
            write_matrix(
                reference / filename(4, 2),
                [[1.0 + 0.0j, 0.0j], [0.0j, 2.0 + 0.0j]],
            )
            write_matrix(
                candidate / filename(4, 2),
                [[1.0 + 0.0j, 3.0e-9 + 4.0e-9j], [0.0j, 2.0 + 0.0j]],
            )

            report = compare_directories(
                reference,
                candidate,
                max_abs_tolerance_ha=1.0e-10,
                relative_frobenius_tolerance=1.0e-10,
            )

        self.assertFalse(report["passed"])
        self.assertEqual(report["max_abs_location"], [0, 4, 2, 0, 1])
        self.assertAlmostEqual(report["max_abs_difference_ha"], 5.0e-9)
        self.assertEqual(report["reference_value_at_max"], {"real": 0.0, "imag": 0.0})
        self.assertEqual(
            report["candidate_value_at_max"],
            {"real": 3.0e-9, "imag": 4.0e-9},
        )
        self.assertEqual(report["exact_element_count"], 3)
        self.assertAlmostEqual(report["element_abs_difference_quantiles_ha"]["q100"], 5.0e-9)

    def test_cli_writes_report_before_returning_threshold_failure(self) -> None:
        with test_directory() as tmp:
            reference = tmp / "reference"
            candidate = tmp / "candidate"
            output = tmp / "comparison.json"
            reference.mkdir()
            candidate.mkdir()
            write_matrix(reference / filename(0, 0), [[1.0 + 0.0j]])
            write_matrix(candidate / filename(0, 0), [[1.0 + 2.0e-9j]])

            completed = subprocess.run(
                [
                    sys.executable,
                    str(HERE / "compare_g0w0_sigc_dump_directories.py"),
                    str(reference),
                    str(candidate),
                    str(output),
                    "--max-abs-tolerance-ha",
                    "1e-10",
                    "--relative-frobenius-tolerance",
                    "1e-10",
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            payload = json.loads(output.read_text(encoding="utf-8"))

        self.assertEqual(completed.returncode, 1)
        self.assertFalse(payload["passed"])


if __name__ == "__main__":
    unittest.main()
