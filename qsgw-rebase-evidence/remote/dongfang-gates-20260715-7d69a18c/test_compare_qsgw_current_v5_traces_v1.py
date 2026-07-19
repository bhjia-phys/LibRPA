#!/usr/bin/env python3
"""Tests for compare_qsgw_current_v5_traces_v1.

The tool under test compares two current (contract version 5) QSGW
component traces to verify that two executables/runs of the same
computation agree numerically. These tests lock the pass/fail contract:
identical traces pass, tolerance-level noise passes, genuine differences
fail with exit code 2, and malformed traces fail closed with exit code 1.
"""

from __future__ import annotations

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from compare_qsgw_current_v5_traces_v1 import (  # noqa: E402
    TraceFormatError,
    compare_trace_text,
    main,
    parse_iterations,
)


SHA = "0" * 64


def _header(
    *,
    version: int = 5,
    mixer: str = "linear",
    beta: str = "2.00000000000000011e-01",
    hartree: str = "delta_density",
    sha: str = SHA,
) -> list[str]:
    return [
        f"# qsgw_contract_version {version}",
        "# fixed_basis immutable_mf0",
        "# live_update eigenvalues_wfc",
        "# velocity disabled_stage1",
        "# headwing disabled_stage1",
        "# symmetry unsupported_full_bz_only",
        f"# hartree {hartree}",
        "# hartree_coulomb truncated",
        "# hartree_normalization legacy_extra_inverse_nk",
        "# band disabled_stage1",
        "# qsgw_input_contract synthetic.contract",
        f"# qsgw_input_contract_sha256 {sha}",
        f"# qsgw_mixer {mixer}",
        f"# qsgw_mixing_beta {beta}",
        "# iter channel component spin kpoint frequency_index frequency_Ha "
        "row column real_value imag_value",
    ]


def _row(
    iteration: int,
    component: str,
    row: int,
    column: int,
    real: float,
    imag: float = 0.0,
    *,
    channel: int = 0,
    spin: int = 0,
    kpoint: int = 0,
    frequency_index: int = -1,
    frequency: float = 0.0,
) -> str:
    return (
        f"{iteration} {channel} {component} {spin} {kpoint} "
        f"{frequency_index} {frequency:.17e} {row} {column} "
        f"{real:.17e} {imag:.17e}"
    )


def _matrix(scale: float, imaginary: float = 0.01) -> list[list[complex]]:
    return [
        [
            complex(scale + 0.1 * r + 0.01 * c, imaginary * (r - c))
            for c in range(2)
        ]
        for r in range(2)
    ]


H0 = _matrix(1.0)
VXC = _matrix(0.3)
EXX = _matrix(-0.2)
VC = _matrix(0.02)
DVH = _matrix(0.05)
RAW = _matrix(0.9)
MIXED = _matrix(0.95)
SIGMA = _matrix(0.01, imaginary=0.001)

ITERATION_ZERO_COMPONENTS = (("h0", H0), ("vxc_dft", VXC))
ITERATIVE_COMPONENTS = (
    ("exx", EXX),
    ("vc", VC),
    ("delta_vh", DVH),
    ("raw_h", RAW),
    ("mixed_h", MIXED),
)
SIGMA_FREQUENCIES = ((0, 0.1), (1, 0.3))

# iter0: 2 components x 2x2; iters 1-2: (5 static + 2 frequency blocks) x 2x2.
ROWS_PER_ITERATION_ZERO = 8
ROWS_PER_ITERATION = 28


def _trace(
    *,
    header_kw: dict | None = None,
    overrides: dict | None = None,
    frequency_overrides: dict | None = None,
    skip: tuple = (),
    raw_lines: tuple = (),
) -> str:
    """Build a synthetic v5 trace.

    ``overrides`` maps (iteration, component, frequency_index, row, column)
    to a replacement complex value. ``frequency_overrides`` maps
    (iteration, component, frequency_index) to a replacement frequency.
    ``skip`` holds element identities to omit; ``raw_lines`` are appended
    verbatim (for malformed-row tests).
    """
    lines = _header(**(header_kw or {}))
    overrides = overrides or {}
    frequency_overrides = frequency_overrides or {}
    skipped = set(skip)

    def emit(
        iteration: int,
        component: str,
        matrix: list[list[complex]],
        frequency_index: int = -1,
        frequency: float = 0.0,
    ) -> None:
        actual_frequency = frequency_overrides.get(
            (iteration, component, frequency_index), frequency
        )
        for row in range(2):
            for column in range(2):
                identity = (iteration, component, frequency_index, row, column)
                if identity in skipped:
                    continue
                value = overrides.get(identity, matrix[row][column])
                lines.append(
                    _row(
                        iteration,
                        component,
                        row,
                        column,
                        value.real,
                        value.imag,
                        frequency_index=frequency_index,
                        frequency=actual_frequency,
                    )
                )

    for component, matrix in ITERATION_ZERO_COMPONENTS:
        emit(0, component, matrix)
    for iteration in (1, 2):
        for component, matrix in ITERATIVE_COMPONENTS:
            emit(iteration, component, matrix)
        for frequency_index, frequency in SIGMA_FREQUENCIES:
            emit(iteration, "sigma_c_iw", SIGMA, frequency_index, frequency)
    lines.extend(raw_lines)
    return "\n".join(lines) + "\n"


def _run_cli(
    old_text: str, new_text: str, *extra_args: str
) -> tuple[int, dict]:
    with tempfile.TemporaryDirectory() as temporary:
        old_path = Path(temporary) / "old.txt"
        new_path = Path(temporary) / "new.txt"
        output_path = Path(temporary) / "report.json"
        old_path.write_text(old_text, encoding="utf-8", newline="\n")
        new_path.write_text(new_text, encoding="utf-8", newline="\n")
        with contextlib.redirect_stdout(io.StringIO()):
            exit_code = main(
                [str(old_path), str(new_path), str(output_path), *extra_args]
            )
        report = json.loads(output_path.read_text(encoding="utf-8"))
    return exit_code, report


class ExcludeComponentsTests(unittest.TestCase):
    def test_excluded_components_are_not_compared(self) -> None:
        old_text = _trace()
        new_text = _trace(
            overrides={
                (1, "delta_vh", -1, 0, 0): complex(0.5, 0.5),
                (2, "delta_vh", -1, 1, 1): complex(-0.25, 0.75),
            }
        )
        report = compare_trace_text(old_text, new_text)
        self.assertFalse(report["passed"])
        report = compare_trace_text(
            old_text, new_text, exclude_components=frozenset({"delta_vh"})
        )
        self.assertTrue(report["passed"])
        self.assertEqual(report["excluded_components"], ["delta_vh"])
        self.assertNotIn("delta_vh", report["components"])
        exit_code, cli_report = _run_cli(
            old_text, new_text, "--exclude-components", "delta_vh"
        )
        self.assertEqual(exit_code, 0)
        self.assertTrue(cli_report["passed"])
        self.assertEqual(cli_report["excluded_components"], ["delta_vh"])

    def test_excluded_component_frequency_is_not_compared(self) -> None:
        old_text = _trace()
        new_text = _trace(
            frequency_overrides={(1, "sigma_c_iw", 0): 99.0}
        )
        report = compare_trace_text(
            old_text, new_text, exclude_components=frozenset({"sigma_c_iw"})
        )
        self.assertTrue(report["passed"])
        self.assertEqual(report["frequency_difference_count"], 0)

    def test_exclusion_does_not_hide_other_failures(self) -> None:
        old_text = _trace()
        new_text = _trace(
            overrides={
                (1, "delta_vh", -1, 0, 0): complex(0.5, 0.5),
                (0, "h0", -1, 0, 0): H0[0][0] + 1.0e-11,
            }
        )
        report = compare_trace_text(
            old_text, new_text, exclude_components=frozenset({"delta_vh"})
        )
        self.assertFalse(report["passed"])
        self.assertFalse(report["components"]["h0"]["passed"])
        self.assertNotIn("delta_vh", report["components"])
        exit_code, _report = _run_cli(
            old_text, new_text, "--exclude-components", "delta_vh"
        )
        self.assertEqual(exit_code, 2)


class ParseIterationsTests(unittest.TestCase):
    def test_inclusive_range_and_list(self) -> None:
        self.assertEqual(parse_iterations("0:2"), [0, 1, 2])
        self.assertEqual(parse_iterations("1,3"), [1, 3])


class ComparePassTests(unittest.TestCase):
    def test_identical_traces_pass(self) -> None:
        report = compare_trace_text(_trace(), _trace())
        self.assertTrue(report["passed"])
        self.assertEqual(report["contract_differences"], [])
        self.assertEqual(
            report["row_count"],
            {
                "old": ROWS_PER_ITERATION_ZERO + 2 * ROWS_PER_ITERATION,
                "new": ROWS_PER_ITERATION_ZERO + 2 * ROWS_PER_ITERATION,
            },
        )
        components = report["components"]
        self.assertEqual(components["h0"]["block_count"], 1)
        self.assertEqual(components["exx"]["block_count"], 2)
        self.assertEqual(components["sigma_c_iw"]["block_count"], 4)
        for metric in components.values():
            self.assertTrue(metric["passed"])
            self.assertEqual(metric["max_abs_diff_ha"], 0.0)
            self.assertEqual(metric["max_relative_frobenius"], 0.0)
        exit_code, cli_report = _run_cli(_trace(), _trace())
        self.assertEqual(exit_code, 0)
        self.assertTrue(cli_report["passed"])

    def test_single_element_1e_13_passes(self) -> None:
        old_text = _trace()
        new_text = _trace(overrides={(0, "h0", -1, 0, 0): H0[0][0] + 1.0e-13})
        report = compare_trace_text(old_text, new_text)
        self.assertTrue(report["passed"])
        metric = report["components"]["h0"]
        self.assertTrue(metric["passed"])
        self.assertTrue(0.9e-13 < metric["max_abs_diff_ha"] < 1.1e-13)


class CompareFailTests(unittest.TestCase):
    def test_single_element_1e_11_fails_default_tolerance(self) -> None:
        old_text = _trace()
        new_text = _trace(overrides={(0, "h0", -1, 0, 0): H0[0][0] + 1.0e-11})
        report = compare_trace_text(old_text, new_text)
        self.assertFalse(report["passed"])
        metric = report["components"]["h0"]
        self.assertFalse(metric["passed"])
        self.assertGreater(metric["max_abs_diff_ha"], 1.0e-12)
        exit_code, cli_report = _run_cli(old_text, new_text)
        self.assertEqual(exit_code, 2)
        self.assertFalse(cli_report["passed"])

    def test_relative_frobenius_failure(self) -> None:
        small = {
            (iteration, "delta_vh", -1, row, column): complex(1.0e-6, 0.0)
            for iteration in (1, 2)
            for row in range(2)
            for column in range(2)
        }
        old_text = _trace(overrides=small)
        shifted = dict(small)
        shifted[(1, "delta_vh", -1, 0, 0)] = complex(1.0e-6 + 5.0e-13, 0.0)
        new_text = _trace(overrides=shifted)
        report = compare_trace_text(old_text, new_text)
        metric = report["components"]["delta_vh"]
        self.assertLessEqual(metric["max_abs_diff_ha"], 1.0e-12)
        self.assertGreater(metric["max_relative_frobenius"], 1.0e-8)
        self.assertFalse(metric["passed"])
        self.assertFalse(report["passed"])

    def test_missing_key_fails(self) -> None:
        old_text = _trace()
        new_text = _trace(skip=((1, "exx", -1, 0, 0),))
        report = compare_trace_text(old_text, new_text)
        self.assertFalse(report["passed"])
        key_differences = report["key_differences"]
        self.assertEqual(key_differences["extra"], [])
        self.assertEqual(
            key_differences["missing"], [[1, 0, "exx", 0, 0, -1, 0, 0]]
        )
        exit_code, _cli_report = _run_cli(old_text, new_text)
        self.assertEqual(exit_code, 2)

    def test_frequency_mismatch_fails(self) -> None:
        old_text = _trace()
        new_text = _trace(
            frequency_overrides={(1, "sigma_c_iw", 0): 0.1 + 1.0e-8}
        )
        report = compare_trace_text(old_text, new_text)
        self.assertFalse(report["passed"])
        self.assertEqual(report["frequency_difference_count"], 4)
        first = report["frequency_differences"][0]
        self.assertEqual(first["key"][:3], [1, 0, "sigma_c_iw"])
        self.assertAlmostEqual(first["old_frequency_ha"], 0.1)
        self.assertAlmostEqual(first["new_frequency_ha"], 0.1 + 1.0e-8)

    def test_contract_difference_fails(self) -> None:
        old_text = _trace()
        new_text = _trace(header_kw={"mixer": "none"})
        report = compare_trace_text(old_text, new_text)
        self.assertFalse(report["passed"])
        self.assertEqual(
            report["contract_differences"],
            [{"key": "qsgw_mixer", "old": "linear", "new": "none"}],
        )
        exit_code, _cli_report = _run_cli(old_text, new_text)
        self.assertEqual(exit_code, 2)

    def test_iterations_subset_compares_only_selected(self) -> None:
        old_text = _trace()
        new_text = _trace(
            overrides={(2, "mixed_h", -1, 0, 0): MIXED[0][0] + 1.0e-11}
        )
        subset = compare_trace_text(old_text, new_text, iterations=[0, 1])
        self.assertTrue(subset["passed"])
        self.assertEqual(
            subset["row_count"],
            {
                "old": ROWS_PER_ITERATION_ZERO + ROWS_PER_ITERATION,
                "new": ROWS_PER_ITERATION_ZERO + ROWS_PER_ITERATION,
            },
        )
        full = compare_trace_text(old_text, new_text)
        self.assertFalse(full["passed"])
        self.assertFalse(full["components"]["mixed_h"]["passed"])
        exit_code, cli_report = _run_cli(
            old_text, new_text, "--iterations", "0:1"
        )
        self.assertEqual(exit_code, 0)
        self.assertTrue(cli_report["passed"])


class FormatErrorTests(unittest.TestCase):
    def test_contract_version_4_rejected(self) -> None:
        with self.assertRaisesRegex(
            TraceFormatError, "unsupported qsgw_contract_version"
        ):
            compare_trace_text(_trace(header_kw={"version": 4}), _trace())
        exit_code, report = _run_cli(
            _trace(header_kw={"version": 4}), _trace()
        )
        self.assertEqual(exit_code, 1)
        self.assertFalse(report["passed"])
        self.assertIn("unsupported qsgw_contract_version", report["error"])

    def test_duplicate_key_rejected(self) -> None:
        duplicate = _row(0, "h0", 0, 0, H0[0][0].real, H0[0][0].imag)
        with self.assertRaisesRegex(TraceFormatError, "duplicate row key"):
            compare_trace_text(_trace(raw_lines=(duplicate,)), _trace())
        exit_code, report = _run_cli(_trace(raw_lines=(duplicate,)), _trace())
        self.assertEqual(exit_code, 1)
        self.assertIn("duplicate row key", report["error"])

    def test_nonfinite_value_rejected(self) -> None:
        bad = _trace(
            overrides={(0, "h0", -1, 0, 0): complex(float("nan"), 0.0)}
        )
        with self.assertRaisesRegex(TraceFormatError, "non-finite"):
            compare_trace_text(bad, _trace())
        exit_code, report = _run_cli(bad, _trace())
        self.assertEqual(exit_code, 1)
        self.assertIn("non-finite", report["error"])

    def test_wrong_column_count_rejected(self) -> None:
        short_row = "1 0 exx 0 0 -1 0.0 0 0 1.0"
        with self.assertRaisesRegex(TraceFormatError, "expected 11 columns"):
            compare_trace_text(_trace(raw_lines=(short_row,)), _trace())
        exit_code, report = _run_cli(_trace(raw_lines=(short_row,)), _trace())
        self.assertEqual(exit_code, 1)
        self.assertIn("expected 11 columns", report["error"])


if __name__ == "__main__":
    unittest.main()
