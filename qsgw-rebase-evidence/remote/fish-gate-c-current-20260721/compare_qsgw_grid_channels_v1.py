#!/usr/bin/env python3
"""Compare the SCF-grid channel of qsgw and qsgw_band runs."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np


def _bootstrap_cmp_qsgw() -> None:
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "regression_tests" / "backend" / "comparisons"
        if candidate.is_dir():
            sys.path.insert(0, str(candidate))


_bootstrap_cmp_qsgw()
import cmp_qsgw  # noqa: E402


SCHEMA = "librpa-qsgw-grid-vs-band-channel-v1"
HA2EV = cmp_qsgw.HA2EV
GRID_IGNORED_CONTRACT_KEYS = {
    "band",
    "h_qsgw_cut",
    "qsgw_band0_unoccupied_keep",
    "qsgw_band0_cut_mode",
    "qsgw_band0_cut_shift_ha",
    "qsgw_input_contract",
    "qsgw_input_contract_sha256",
}


class GridChannelComparisonError(ValueError):
    pass


def _validate_contract_pair(grid: dict, band: dict) -> None:
    if grid["qsgw_contract_version"] != 6 or band["qsgw_contract_version"] != 6:
        raise GridChannelComparisonError("grid-channel comparison requires v6")
    if grid["band"] != "disabled_stage1" or grid["h_qsgw_cut"] != "disabled_non_band":
        raise GridChannelComparisonError("the qsgw run is not grid-only")
    if band["band"] != "fixed_reference_operator_fourier_live" or band["h_qsgw_cut"] != "band_postprocess":
        raise GridChannelComparisonError("the qsgw_band run lacks band postprocessing")
    if band["qsgw_band0_cut_mode"] != 0:
        raise GridChannelComparisonError("grid metamorphic comparison requires cut mode 0")
    keys = (set(grid) | set(band)) - GRID_IGNORED_CONTRACT_KEYS
    differing = sorted(key for key in keys if grid.get(key) != band.get(key))
    if differing:
        raise GridChannelComparisonError(
            f"grid-relevant QSGW contracts differ for {differing}"
        )


def _max_abs(value: np.ndarray) -> float:
    return float(np.max(np.abs(value))) if value.size else 0.0


def compare_grid_channels(
    grid_matrix_text: str,
    grid_eigenvalue_text: str,
    grid_summary_text: str,
    band_matrix_text: str,
    band_eigenvalue_text: str,
    band_summary_text: str,
    *,
    matrix_max_abs_tolerance_ha: float = 1.0e-8,
    matrix_relative_tolerance: float = 1.0e-8,
    eigenvalue_tolerance_ha: float = 1.0e-6,
    coordinate_tolerance: float = 1.0e-12,
    summary_energy_tolerance_ev: float = 1.0e-5,
    summary_residual_tolerance_ha: float = 1.0e-8,
    summary_scalar_tolerance: float = 1.0e-10,
    coefficient_tolerance: float = 1.0e-12,
) -> dict:
    tolerances = (
        matrix_max_abs_tolerance_ha,
        matrix_relative_tolerance,
        eigenvalue_tolerance_ha,
        coordinate_tolerance,
        summary_energy_tolerance_ev,
        summary_residual_tolerance_ha,
        summary_scalar_tolerance,
        coefficient_tolerance,
    )
    if any(not math.isfinite(value) or value <= 0.0 for value in tolerances):
        raise GridChannelComparisonError("comparison tolerances must be positive")

    grid_contracts = [
        cmp_qsgw._parse_contract(grid_matrix_text, "grid matrix trace"),
        cmp_qsgw._parse_contract(grid_eigenvalue_text, "grid eigenvalue trace"),
        cmp_qsgw._parse_contract(grid_summary_text, "grid summary trace"),
    ]
    band_contracts = [
        cmp_qsgw._parse_contract(band_matrix_text, "band matrix trace"),
        cmp_qsgw._parse_contract(band_eigenvalue_text, "band eigenvalue trace"),
        cmp_qsgw._parse_contract(band_summary_text, "band summary trace"),
    ]
    if any(item != grid_contracts[0] for item in grid_contracts[1:]):
        raise GridChannelComparisonError("grid trace contracts differ")
    if any(item != band_contracts[0] for item in band_contracts[1:]):
        raise GridChannelComparisonError("band trace contracts differ")
    _validate_contract_pair(grid_contracts[0], band_contracts[0])

    grid_blocks = cmp_qsgw._parse_matrix_trace(grid_matrix_text, "grid matrix trace")
    band_blocks = cmp_qsgw._parse_matrix_trace(band_matrix_text, "band matrix trace")
    cmp_qsgw._validate_matrix_trajectory(
        grid_blocks, grid_contracts[0], "grid matrix trace"
    )
    cmp_qsgw._validate_matrix_trajectory(
        band_blocks, band_contracts[0], "band matrix trace"
    )
    grid_channel = {key: value for key, value in grid_blocks.items() if key[1] == 0}
    band_channel = {key: value for key, value in band_blocks.items() if key[1] == 0}
    if set(grid_channel) != set(band_channel):
        raise GridChannelComparisonError("grid matrix block layouts differ")
    matrix_max_abs = 0.0
    matrix_difference_square = 0.0
    matrix_scale_square = 0.0
    for key in sorted(grid_channel, key=str):
        grid_matrix = np.asarray(grid_channel[key][1], dtype=np.complex128)
        band_matrix = np.asarray(band_channel[key][1], dtype=np.complex128)
        if grid_matrix.shape != band_matrix.shape:
            raise GridChannelComparisonError(f"matrix shape differs at {key}")
        difference = grid_matrix - band_matrix
        matrix_max_abs = max(matrix_max_abs, _max_abs(difference))
        matrix_difference_square += float(np.vdot(difference, difference).real)
        matrix_scale_square += float(np.vdot(grid_matrix, grid_matrix).real)
    matrix_relative = math.sqrt(
        matrix_difference_square / max(1.0, matrix_scale_square)
    )

    grid_eigen = cmp_qsgw._parse_eigenvalue_trace(
        grid_eigenvalue_text, "grid eigenvalue trace"
    )
    band_eigen = cmp_qsgw._parse_eigenvalue_trace(
        band_eigenvalue_text, "band eigenvalue trace"
    )
    cmp_qsgw._validate_eigenvalue_trajectory(
        grid_eigen, grid_contracts[0], "grid eigenvalue trace"
    )
    cmp_qsgw._validate_eigenvalue_trajectory(
        band_eigen, band_contracts[0], "band eigenvalue trace"
    )
    grid_eigen_channel = {key: value for key, value in grid_eigen.items() if key[1] == 0}
    band_eigen_channel = {key: value for key, value in band_eigen.items() if key[1] == 0}
    if set(grid_eigen_channel) != set(band_eigen_channel):
        raise GridChannelComparisonError("grid eigenvalue layouts differ")
    eigenvalue_max_abs_ha = 0.0
    coordinate_max_abs = 0.0
    for key in grid_eigen_channel:
        grid_coordinate, grid_energy = grid_eigen_channel[key]
        band_coordinate, band_energy = band_eigen_channel[key]
        coordinate_max_abs = max(
            coordinate_max_abs,
            max(abs(left - right) for left, right in zip(grid_coordinate, band_coordinate)),
        )
        eigenvalue_max_abs_ha = max(
            eigenvalue_max_abs_ha, abs(grid_energy - band_energy) / HA2EV
        )

    grid_summary = cmp_qsgw._parse_iteration_summary(
        grid_summary_text, "grid summary trace"
    )
    band_summary = cmp_qsgw._parse_iteration_summary(
        band_summary_text, "band summary trace"
    )
    if set(grid_summary) != set(band_summary):
        raise GridChannelComparisonError("summary iteration layouts differ")
    summary_energy_max = 0.0
    summary_residual_max = 0.0
    summary_scalar_max = 0.0
    coefficient_max = 0.0
    energy_fields = ("max_delta_eV", "efermi_eV", "gap_eV")
    residual_fields = ("residual_l2_Ha", "residual_max_Ha")
    scalar_fields = ("electron_count", "beta", "rcond", "coefficient_l1")
    exact_fields = (
        "requested_mode", "applied_mode", "fallback", "coefficient_count",
        "converged", "fallback_reason",
    )
    for iteration in grid_summary:
        left = grid_summary[iteration]
        right = band_summary[iteration]
        if any(left[field] != right[field] for field in exact_fields):
            raise GridChannelComparisonError(
                f"summary discrete fields differ at iteration {iteration}"
            )
        summary_energy_max = max(
            summary_energy_max,
            *(abs(left[field] - right[field]) for field in energy_fields),
        )
        summary_residual_max = max(
            summary_residual_max,
            *(abs(left[field] - right[field]) for field in residual_fields),
        )
        summary_scalar_max = max(
            summary_scalar_max,
            *(abs(left[field] - right[field]) for field in scalar_fields),
        )
        if len(left["coefficients"]) != len(right["coefficients"]):
            raise GridChannelComparisonError("summary coefficient layouts differ")
        coefficient_max = max(
            coefficient_max,
            *(abs(a - b) for a, b in zip(left["coefficients"], right["coefficients"])),
            0.0,
        )

    matrix_passed = (
        matrix_max_abs <= matrix_max_abs_tolerance_ha
        and matrix_relative <= matrix_relative_tolerance
    )
    eigenvalue_passed = (
        eigenvalue_max_abs_ha <= eigenvalue_tolerance_ha
        and coordinate_max_abs <= coordinate_tolerance
    )
    summary_passed = (
        summary_energy_max <= summary_energy_tolerance_ev
        and summary_residual_max <= summary_residual_tolerance_ha
        and summary_scalar_max <= summary_scalar_tolerance
        and coefficient_max <= coefficient_tolerance
    )
    return {
        "schema": SCHEMA,
        "passed": matrix_passed and eigenvalue_passed and summary_passed,
        "acceptance_scope": "qsgw_vs_qsgw_band_grid_channel_metamorphic",
        "matrix": {
            "block_count": len(grid_channel),
            "max_abs_ha": matrix_max_abs,
            "relative_frobenius": matrix_relative,
            "passed": matrix_passed,
        },
        "eigenvalues": {
            "value_count": len(grid_eigen_channel),
            "max_abs_ha": eigenvalue_max_abs_ha,
            "coordinate_max_abs": coordinate_max_abs,
            "passed": eigenvalue_passed,
        },
        "summary": {
            "iteration_count": len(grid_summary),
            "energy_max_abs_ev": summary_energy_max,
            "residual_max_abs_ha": summary_residual_max,
            "scalar_max_abs": summary_scalar_max,
            "coefficient_max_abs": coefficient_max,
            "passed": summary_passed,
        },
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("grid_run", type=Path)
    parser.add_argument("band_run", type=Path)
    parser.add_argument("output", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        report = compare_grid_channels(
            (args.grid_run / "qsgw_matrices.dat").read_text(encoding="utf-8"),
            (args.grid_run / "qsgw_eigenvalues.dat").read_text(encoding="utf-8"),
            (args.grid_run / "qsgw_iterations.dat").read_text(encoding="utf-8"),
            (args.band_run / "qsgw_matrices.dat").read_text(encoding="utf-8"),
            (args.band_run / "qsgw_eigenvalues.dat").read_text(encoding="utf-8"),
            (args.band_run / "qsgw_iterations.dat").read_text(encoding="utf-8"),
        )
    except Exception as error:
        report = {"schema": SCHEMA, "passed": False, "error": str(error)}
        exit_code = 1
    else:
        exit_code = 0 if report["passed"] else 2
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
