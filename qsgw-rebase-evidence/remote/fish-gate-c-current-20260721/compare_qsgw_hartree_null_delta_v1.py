#!/usr/bin/env python3
"""Verify that the first Hartree update has no numerical side effects.

At QSGW iteration one the live mean field is still identical to the immutable
reference, so delta V_H must vanish.  A Hartree-enabled run must therefore be
numerically identical to an otherwise matching Hartree-disabled control through
iteration one.  This metamorphic check catches reader or setup contamination
that is invisible when only the dumped delta V_H matrix is inspected.
"""

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


SCHEMA = "librpa-qsgw-hartree-null-delta-side-effect-v1"
IGNORED_CONTRACT_KEYS = {
    "hartree",
    "hartree_coulomb",
    "hartree_normalization",
    "qsgw_input_contract",
    "qsgw_input_contract_sha256",
}
ENERGY_FIELDS = ("max_delta_eV", "efermi_eV", "gap_eV")
RESIDUAL_FIELDS = ("residual_l2_Ha", "residual_max_Ha")
SCALAR_FIELDS = ("electron_count", "beta", "rcond", "coefficient_l1")
EXACT_FIELDS = (
    "requested_mode",
    "applied_mode",
    "fallback",
    "coefficient_count",
    "converged",
    "fallback_reason",
)


class HartreeNullDeltaComparisonError(ValueError):
    pass


def _max_abs(value: np.ndarray) -> float:
    return float(np.max(np.abs(value))) if value.size else 0.0


def _validate_contract_pair(enabled: dict, disabled: dict) -> None:
    if enabled["qsgw_contract_version"] != 6 or disabled["qsgw_contract_version"] != 6:
        raise HartreeNullDeltaComparisonError("null-delta comparison requires v6 traces")
    if enabled["hartree"] != "delta_density":
        raise HartreeNullDeltaComparisonError("enabled trace does not use delta-density Hartree")
    if disabled["hartree"] != "disabled_stage1":
        raise HartreeNullDeltaComparisonError("control trace does not disable Hartree")
    if "hartree_coulomb" in disabled or "hartree_normalization" in disabled:
        raise HartreeNullDeltaComparisonError(
            "Hartree-disabled control unexpectedly declares Hartree settings"
        )
    keys = (set(enabled) | set(disabled)) - IGNORED_CONTRACT_KEYS
    differing = sorted(key for key in keys if enabled.get(key) != disabled.get(key))
    if differing:
        raise HartreeNullDeltaComparisonError(
            f"non-Hartree contracts differ for {differing}"
        )


def _parse_consistent_contracts(texts: tuple[str, str, str], label: str) -> dict:
    contracts = [
        cmp_qsgw._parse_contract(text, f"{label} {kind} trace")
        for text, kind in zip(texts, ("matrix", "eigenvalue", "summary"))
    ]
    if any(contract != contracts[0] for contract in contracts[1:]):
        raise HartreeNullDeltaComparisonError(f"{label} trace contracts differ")
    return contracts[0]


def compare_hartree_null_delta(
    enabled_matrix_text: str,
    enabled_eigenvalue_text: str,
    enabled_summary_text: str,
    disabled_matrix_text: str,
    disabled_eigenvalue_text: str,
    disabled_summary_text: str,
    *,
    delta_zero_tolerance_ha: float = 1.0e-10,
    matrix_max_abs_tolerance_ha: float = 1.0e-8,
    matrix_relative_tolerance: float = 1.0e-8,
    frequency_tolerance_ha: float = 1.0e-12,
    eigenvalue_tolerance_ha: float = 1.0e-6,
    coordinate_tolerance: float = 1.0e-12,
    summary_energy_tolerance_ev: float = 1.0e-5,
    summary_residual_tolerance_ha: float = 1.0e-8,
    summary_scalar_tolerance: float = 1.0e-10,
    coefficient_tolerance: float = 1.0e-12,
) -> dict:
    tolerances = (
        delta_zero_tolerance_ha,
        matrix_max_abs_tolerance_ha,
        matrix_relative_tolerance,
        frequency_tolerance_ha,
        eigenvalue_tolerance_ha,
        coordinate_tolerance,
        summary_energy_tolerance_ev,
        summary_residual_tolerance_ha,
        summary_scalar_tolerance,
        coefficient_tolerance,
    )
    if any(not math.isfinite(value) or value <= 0.0 for value in tolerances):
        raise HartreeNullDeltaComparisonError("comparison tolerances must be positive")

    enabled_contract = _parse_consistent_contracts(
        (enabled_matrix_text, enabled_eigenvalue_text, enabled_summary_text),
        "Hartree-enabled",
    )
    disabled_contract = _parse_consistent_contracts(
        (disabled_matrix_text, disabled_eigenvalue_text, disabled_summary_text),
        "Hartree-disabled",
    )
    _validate_contract_pair(enabled_contract, disabled_contract)

    enabled_blocks = cmp_qsgw._parse_matrix_trace(
        enabled_matrix_text, "Hartree-enabled matrix trace"
    )
    disabled_blocks = cmp_qsgw._parse_matrix_trace(
        disabled_matrix_text, "Hartree-disabled matrix trace"
    )
    cmp_qsgw._validate_matrix_trajectory(
        enabled_blocks, enabled_contract, "Hartree-enabled matrix trace"
    )
    cmp_qsgw._validate_matrix_trajectory(
        disabled_blocks, disabled_contract, "Hartree-disabled matrix trace"
    )

    enabled_window = {
        key: value
        for key, value in enabled_blocks.items()
        if key[0] <= 1 and key[1] == 0
    }
    disabled_window = {
        key: value
        for key, value in disabled_blocks.items()
        if key[0] <= 1 and key[1] == 0
    }
    delta_blocks = {
        key: value for key, value in enabled_window.items() if key[2] == "delta_vh"
    }
    if not delta_blocks or {key[0] for key in delta_blocks} != {1}:
        raise HartreeNullDeltaComparisonError(
            "Hartree-enabled trace must contain iteration-one delta_vh blocks"
        )
    delta_max_abs = max(
        _max_abs(np.asarray(value[1], dtype=np.complex128))
        for value in delta_blocks.values()
    )
    enabled_comparable = {
        key: value for key, value in enabled_window.items() if key[2] != "delta_vh"
    }
    if set(enabled_comparable) != set(disabled_window):
        missing = sorted(set(disabled_window) - set(enabled_comparable), key=str)
        extra = sorted(set(enabled_comparable) - set(disabled_window), key=str)
        raise HartreeNullDeltaComparisonError(
            f"iteration-one matrix layouts differ; missing={missing}, extra={extra}"
        )

    matrix_max_abs = 0.0
    matrix_difference_square = 0.0
    matrix_scale_square = 0.0
    frequency_max_abs = 0.0
    for key in sorted(enabled_comparable, key=str):
        enabled_frequency, enabled_matrix = enabled_comparable[key]
        disabled_frequency, disabled_matrix = disabled_window[key]
        frequency_max_abs = max(
            frequency_max_abs, abs(enabled_frequency - disabled_frequency)
        )
        left = np.asarray(enabled_matrix, dtype=np.complex128)
        right = np.asarray(disabled_matrix, dtype=np.complex128)
        if left.shape != right.shape:
            raise HartreeNullDeltaComparisonError(f"matrix shape differs at {key}")
        difference = left - right
        matrix_max_abs = max(matrix_max_abs, _max_abs(difference))
        matrix_difference_square += float(np.vdot(difference, difference).real)
        matrix_scale_square += float(np.vdot(left, left).real)
    matrix_relative = math.sqrt(
        matrix_difference_square / max(1.0, matrix_scale_square)
    )

    enabled_eigen = cmp_qsgw._parse_eigenvalue_trace(
        enabled_eigenvalue_text, "Hartree-enabled eigenvalue trace"
    )
    disabled_eigen = cmp_qsgw._parse_eigenvalue_trace(
        disabled_eigenvalue_text, "Hartree-disabled eigenvalue trace"
    )
    cmp_qsgw._validate_eigenvalue_trajectory(
        enabled_eigen, enabled_contract, "Hartree-enabled eigenvalue trace"
    )
    cmp_qsgw._validate_eigenvalue_trajectory(
        disabled_eigen, disabled_contract, "Hartree-disabled eigenvalue trace"
    )
    enabled_eigen_window = {
        key: value for key, value in enabled_eigen.items() if key[0] <= 1 and key[1] == 0
    }
    disabled_eigen_window = {
        key: value for key, value in disabled_eigen.items() if key[0] <= 1 and key[1] == 0
    }
    if set(enabled_eigen_window) != set(disabled_eigen_window):
        raise HartreeNullDeltaComparisonError("iteration-one eigenvalue layouts differ")
    eigenvalue_max_abs_ha = 0.0
    coordinate_max_abs = 0.0
    for key, (enabled_coordinate, enabled_energy) in enabled_eigen_window.items():
        disabled_coordinate, disabled_energy = disabled_eigen_window[key]
        coordinate_max_abs = max(
            coordinate_max_abs,
            max(
                abs(left - right)
                for left, right in zip(enabled_coordinate, disabled_coordinate)
            ),
        )
        eigenvalue_max_abs_ha = max(
            eigenvalue_max_abs_ha,
            abs(enabled_energy - disabled_energy) / cmp_qsgw.HA2EV,
        )

    enabled_summary = cmp_qsgw._parse_iteration_summary(
        enabled_summary_text, "Hartree-enabled summary trace"
    )
    disabled_summary = cmp_qsgw._parse_iteration_summary(
        disabled_summary_text, "Hartree-disabled summary trace"
    )
    enabled_summary_window = {
        iteration: row for iteration, row in enabled_summary.items() if iteration <= 1
    }
    disabled_summary_window = {
        iteration: row for iteration, row in disabled_summary.items() if iteration <= 1
    }
    if set(enabled_summary_window) != {0, 1} or set(disabled_summary_window) != {0, 1}:
        raise HartreeNullDeltaComparisonError(
            "both summary traces must cover exactly iterations zero and one in the comparison window"
        )
    summary_energy_max = 0.0
    summary_residual_max = 0.0
    summary_scalar_max = 0.0
    coefficient_max = 0.0
    for iteration in (0, 1):
        left = enabled_summary_window[iteration]
        right = disabled_summary_window[iteration]
        if any(left[field] != right[field] for field in EXACT_FIELDS):
            raise HartreeNullDeltaComparisonError(
                f"summary discrete fields differ at iteration {iteration}"
            )
        summary_energy_max = max(
            summary_energy_max,
            *(abs(left[field] - right[field]) for field in ENERGY_FIELDS),
        )
        summary_residual_max = max(
            summary_residual_max,
            *(abs(left[field] - right[field]) for field in RESIDUAL_FIELDS),
        )
        summary_scalar_max = max(
            summary_scalar_max,
            *(abs(left[field] - right[field]) for field in SCALAR_FIELDS),
        )
        if len(left["coefficients"]) != len(right["coefficients"]):
            raise HartreeNullDeltaComparisonError(
                "summary coefficient layouts differ"
            )
        coefficient_max = max(
            coefficient_max,
            *(abs(a - b) for a, b in zip(left["coefficients"], right["coefficients"])),
            0.0,
        )

    delta_passed = delta_max_abs <= delta_zero_tolerance_ha
    matrix_passed = (
        matrix_max_abs <= matrix_max_abs_tolerance_ha
        and matrix_relative <= matrix_relative_tolerance
        and frequency_max_abs <= frequency_tolerance_ha
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
        "passed": delta_passed and matrix_passed and eigenvalue_passed and summary_passed,
        "acceptance_scope": "iteration1_zero_hartree_side_effect_guard",
        "delta_vh": {
            "block_count": len(delta_blocks),
            "max_abs_ha": delta_max_abs,
            "passed": delta_passed,
        },
        "matrix": {
            "block_count": len(enabled_comparable),
            "max_abs_ha": matrix_max_abs,
            "relative_frobenius": matrix_relative,
            "frequency_max_abs_ha": frequency_max_abs,
            "passed": matrix_passed,
        },
        "eigenvalues": {
            "value_count": len(enabled_eigen_window),
            "max_abs_ha": eigenvalue_max_abs_ha,
            "coordinate_max_abs": coordinate_max_abs,
            "passed": eigenvalue_passed,
        },
        "summary": {
            "energy_max_abs_ev": summary_energy_max,
            "residual_max_abs_ha": summary_residual_max,
            "scalar_max_abs": summary_scalar_max,
            "coefficient_max_abs": coefficient_max,
            "passed": summary_passed,
        },
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("enabled_run", type=Path)
    parser.add_argument("disabled_run", type=Path)
    parser.add_argument("output", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        report = compare_hartree_null_delta(
            (args.enabled_run / "qsgw_matrices.dat").read_text(encoding="utf-8"),
            (args.enabled_run / "qsgw_eigenvalues.dat").read_text(encoding="utf-8"),
            (args.enabled_run / "qsgw_iterations.dat").read_text(encoding="utf-8"),
            (args.disabled_run / "qsgw_matrices.dat").read_text(encoding="utf-8"),
            (args.disabled_run / "qsgw_eigenvalues.dat").read_text(encoding="utf-8"),
            (args.disabled_run / "qsgw_iterations.dat").read_text(encoding="utf-8"),
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
