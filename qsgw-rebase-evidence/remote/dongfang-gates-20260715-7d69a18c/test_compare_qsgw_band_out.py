#!/usr/bin/env python3

from contextlib import contextmanager
import json
import shutil
import subprocess
import sys
import unittest
import uuid
from pathlib import Path


HERE = Path(__file__).resolve().parent
TEST_TMP = HERE / "__test_tmp"
TEST_TMP.mkdir(exist_ok=True)
sys.path.insert(0, str(HERE))

from compare_qsgw_band_out import compare_band_out_files, read_band_out


@contextmanager
def test_directory():
    path = TEST_TMP / uuid.uuid4().hex
    path.mkdir()
    try:
        yield path
    finally:
        shutil.rmtree(path)


def write_band_out(
    path: Path,
    *,
    fermi: float = 0.5,
    eigen_shift: float = 0.0,
    occupation_shift: float = 0.0,
) -> None:
    path.write_text(
        "\n".join(
            [
                "2",
                "1",
                "2",
                "2",
                f"{fermi:.16f}",
                "1 1",
                f"1 {2.0 + occupation_shift:.16f} {-0.5 + eigen_shift:.16f} -13.605693",
                "2 0.0000000000000000 0.3000000000000000 8.163416",
                "2 1",
                "1 2.0000000000000000 -0.4000000000000000 -10.884555",
                "2 0.0000000000000000 0.4000000000000000 10.884555",
            ]
        )
        + "\n",
        encoding="ascii",
    )


class BandOutComparisonTests(unittest.TestCase):
    def test_reads_expected_shape_and_rows(self) -> None:
        with test_directory() as tmp:
            path = tmp / "band_out"
            write_band_out(path)

            data = read_band_out(path)

        self.assertEqual(data.n_kpoints, 2)
        self.assertEqual(data.n_spins, 1)
        self.assertEqual(data.n_bands, 2)
        self.assertEqual(len(data.states), 4)
        self.assertEqual(data.states[-1].kpoint, 2)
        self.assertEqual(data.states[-1].spin, 1)
        self.assertEqual(data.states[-1].band, 2)

    def test_exact_files_pass_zero_tolerances(self) -> None:
        with test_directory() as tmp:
            reference = tmp / "reference"
            candidate = tmp / "candidate"
            write_band_out(reference)
            write_band_out(candidate)

            result = compare_band_out_files(
                reference,
                candidate,
                eigenvalue_tolerance_ha=0.0,
                occupation_tolerance=0.0,
                fermi_tolerance_ha=0.0,
            )

        self.assertTrue(result["passed"])
        self.assertTrue(result["byte_identical"])
        self.assertEqual(result["differences"]["eigenvalue_max_abs_ha"], 0.0)
        self.assertEqual(result["differences"]["occupation_max_abs"], 0.0)

    def test_reports_largest_numeric_differences_and_location(self) -> None:
        with test_directory() as tmp:
            reference = tmp / "reference"
            candidate = tmp / "candidate"
            write_band_out(reference)
            write_band_out(
                candidate,
                fermi=0.5003,
                eigen_shift=2.0e-6,
                occupation_shift=3.0e-8,
            )

            result = compare_band_out_files(
                reference,
                candidate,
                eigenvalue_tolerance_ha=1.0e-10,
                occupation_tolerance=1.0e-10,
                fermi_tolerance_ha=1.0e-10,
            )

        self.assertFalse(result["passed"])
        self.assertFalse(result["byte_identical"])
        self.assertAlmostEqual(
            result["differences"]["eigenvalue_max_abs_ha"], 2.0e-6
        )
        self.assertAlmostEqual(result["differences"]["occupation_max_abs"], 3.0e-8)
        self.assertAlmostEqual(result["differences"]["fermi_max_abs_ha"], 3.0e-4)
        self.assertEqual(
            result["differences"]["eigenvalue_max_location"],
            {"kpoint": 1, "spin": 1, "band": 1},
        )

    def test_cli_writes_json_before_returning_failure(self) -> None:
        with test_directory() as tmp:
            reference = tmp / "reference"
            candidate = tmp / "candidate"
            output = tmp / "comparison.json"
            write_band_out(reference)
            write_band_out(candidate, eigen_shift=2.0e-6)

            completed = subprocess.run(
                [
                    sys.executable,
                    str(HERE / "compare_qsgw_band_out.py"),
                    str(reference),
                    str(candidate),
                    str(output),
                    "--eigenvalue-tolerance-ha",
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
