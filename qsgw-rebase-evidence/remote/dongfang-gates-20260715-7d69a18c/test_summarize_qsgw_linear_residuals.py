import json
import pathlib
import tempfile
import unittest

import summarize_qsgw_linear_residuals as observer


def matrix_rows(iteration, component, matrix, lower_override=None):
    rows = []
    for row, values in enumerate(matrix):
        for column, value in enumerate(values):
            if lower_override is not None and row > column:
                value = lower_override
            rows.append(
                f"{iteration} 0 {component} 0 0 -1 0 {row} {column} "
                f"{value.real:.17e} {value.imag:.17e}"
            )
    return rows


def traces():
    h0 = [[1.0 + 0.0j, 0.2 + 0.1j], [0.2 - 0.1j, 2.0 + 0.0j]]
    raw1 = [[1.5 + 0.0j, 0.4 + 0.2j], [0.4 - 0.2j, 1.0 + 0.0j]]
    mixed1 = [[1.1 + 0.0j, 0.24 + 0.12j], [0.24 - 0.12j, 1.8 + 0.0j]]
    raw2 = [[0.9 + 0.0j, 0.3 + 0.15j], [0.3 - 0.15j, 2.2 + 0.0j]]
    mixed2 = [[1.06 + 0.0j, 0.252 + 0.126j], [0.252 - 0.126j, 1.88 + 0.0j]]
    old = []
    current = []
    for iteration, component, matrix in (
        (0, "h0", h0),
        (1, "raw_h", raw1),
        (1, "mixed_h", mixed1),
        (2, "raw_h", raw2),
        (2, "mixed_h", mixed2),
    ):
        lower_override = None if component == "h0" else 99.0 + 7.0j
        old.extend(matrix_rows(iteration, component, matrix, lower_override))
        current.extend(matrix_rows(iteration, component, matrix))
    residual1 = observer.matrix_residual_norms(h0, raw1)
    residual2 = observer.matrix_residual_norms(mixed1, raw2)
    iteration_trace = [
        "# iter max_delta_eV residual_l2_Ha residual_max_Ha efermi_eV "
        "gap_eV electron_count requested_mode applied_mode beta fallback "
        "rcond coefficient_l1 coefficient_count converged coefficients fallback_reason",
        "0 0 0 0 0 1 8 -1 -1 0.2 0 1 0 0 0 none none",
        f"1 1 {residual1[0]} {residual1[1]} 0 2 8 0 0 0.2 0 1 1 1 0 1 none",
        f"2 0.5 {residual2[0]} {residual2[1]} 0 3 8 0 0 0.2 0 1 1 1 0 1 none",
    ]
    return "\n".join(old), "\n".join(current), "\n".join(iteration_trace)


class LinearResidualObserverTests(unittest.TestCase):
    def test_old_current_and_reported_residuals_agree(self):
        old, current, iteration_trace = traces()
        report = observer.compare_residuals(
            old,
            current,
            iteration_trace,
            [0, 1, 2],
            absolute_tolerance=1.0e-12,
            relative_tolerance=1.0e-12,
        )
        self.assertTrue(report["passed"])
        self.assertEqual(len(report["trajectory"]), 3)
        self.assertEqual(report["trajectory"][0]["residual_l2_ha"], 0.0)
        self.assertLessEqual(report["max_old_current_l2_abs_diff_ha"], 1.0e-12)
        self.assertLessEqual(report["max_current_reported_l2_abs_diff_ha"], 1.0e-12)

    def test_reported_residual_mismatch_fails(self):
        old, current, iteration_trace = traces()
        lines = iteration_trace.splitlines()
        fields = lines[2].split()
        fields[2] = "9.0"
        lines[2] = " ".join(fields)
        iteration_trace = "\n".join(lines)
        report = observer.compare_residuals(
            old,
            current,
            iteration_trace,
            [0, 1, 2],
            absolute_tolerance=1.0e-12,
            relative_tolerance=1.0e-12,
        )
        self.assertFalse(report["passed"])

    def test_cli_writes_json_and_csv(self):
        old, current, iteration_trace = traces()
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            old_path = root / "old.dat"
            current_path = root / "current.dat"
            iteration_path = root / "iterations.dat"
            json_path = root / "report.json"
            csv_path = root / "trajectory.csv"
            old_path.write_text(old)
            current_path.write_text(current)
            iteration_path.write_text(iteration_trace)
            exit_code = observer.main(
                [
                    str(old_path),
                    str(current_path),
                    str(iteration_path),
                    str(json_path),
                    str(csv_path),
                    "--iterations",
                    "0:2",
                    "--absolute-tolerance-ha",
                    "1e-12",
                    "--relative-tolerance",
                    "1e-12",
                ]
            )
            self.assertEqual(exit_code, 0)
            self.assertTrue(json.loads(json_path.read_text())["passed"])
            self.assertEqual(len(csv_path.read_text().splitlines()), 4)


if __name__ == "__main__":
    unittest.main()
