#!/usr/bin/env python3
"""Validate current-contract grid-only QSGW Hartree trace invariants."""

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


SCHEMA = "librpa-qsgw-hartree-trace-v6-validation-v1"


class HartreeTraceValidationError(ValueError):
    pass


def _matrix(blocks: dict, iteration: int, component: str, spin: int, kpoint: int):
    key = (iteration, 0, component, spin, kpoint, -1)
    if key not in blocks:
        raise HartreeTraceValidationError(f"missing matrix block {key}")
    return np.asarray(blocks[key][1], dtype=np.complex128)


def _layout(blocks: dict, iteration: int, component: str) -> set[tuple[int, int]]:
    return {
        (key[3], key[4])
        for key in blocks
        if key[0] == iteration
        and key[1] == 0
        and key[2] == component
        and key[5] == -1
    }


def _max_abs(value: np.ndarray) -> float:
    return float(np.max(np.abs(value))) if value.size else 0.0


def _relative(actual: np.ndarray, expected: np.ndarray) -> float:
    difference = float(np.linalg.norm(actual - expected))
    scale = float(np.linalg.norm(expected))
    return difference / scale if scale else difference


def validate_hartree_trace(
    matrix_text: str,
    summary_text: str,
    *,
    expected_iterations: int = 2,
    expected_symmetry: str = "exx_on_gw_on_rpa_on",
    closure_tolerance_ha: float = 1.0e-10,
    closure_relative_tolerance: float = 1.0e-8,
    hermiticity_tolerance_ha: float = 1.0e-10,
    electron_count_tolerance: float = 1.0e-10,
    initial_zero_tolerance_ha: float = 1.0e-10,
    response_nonzero_tolerance_ha: float = 1.0e-12,
) -> dict:
    tolerances = (
        closure_tolerance_ha,
        closure_relative_tolerance,
        hermiticity_tolerance_ha,
        electron_count_tolerance,
        initial_zero_tolerance_ha,
        response_nonzero_tolerance_ha,
    )
    if expected_iterations < 2 or any(
        not math.isfinite(value) or value <= 0.0 for value in tolerances
    ):
        raise HartreeTraceValidationError("invalid Hartree trace controls")
    contract = cmp_qsgw._require_same_contract(
        matrix_text, summary_text, "Hartree matrix/summary trace"
    )
    if contract["qsgw_contract_version"] != 6:
        raise HartreeTraceValidationError("Hartree trace validator requires v6")
    if contract["hartree"] != "delta_density":
        raise HartreeTraceValidationError("Hartree delta_density is not enabled")
    if contract["headwing"] != "disabled_stage1":
        raise HartreeTraceValidationError("Gate C requires head-wing disabled")
    if contract["band"] != "disabled_stage1":
        raise HartreeTraceValidationError(
            "grid-only Hartree closure cannot validate a cut band channel"
        )
    if expected_symmetry and contract["symmetry"] != expected_symmetry:
        raise HartreeTraceValidationError("trace symmetry contract differs")

    blocks = cmp_qsgw._parse_matrix_trace(matrix_text, "Hartree matrix trace")
    cmp_qsgw._validate_matrix_trajectory(blocks, contract, "Hartree matrix trace")
    summaries = cmp_qsgw._parse_iteration_summary(
        summary_text, "Hartree iteration trace"
    )
    iterations = list(range(expected_iterations + 1))
    if sorted(summaries) != iterations or sorted({key[0] for key in blocks}) != iterations:
        raise HartreeTraceValidationError("trace iteration window differs")

    reference_layout = _layout(blocks, 0, "h0")
    if not reference_layout:
        raise HartreeTraceValidationError("empty grid reference layout")
    reference_electron_count = summaries[0]["electron_count"]
    max_summary_charge_drift = 0.0
    max_trace_summary_count_difference = 0.0
    closure_max_abs = 0.0
    closure_relative = 0.0
    hermiticity_max = 0.0
    iteration_delta_max: dict[int, float] = {}
    occupation_counts: dict[int, float] = {}

    for iteration in iterations:
        max_summary_charge_drift = max(
            max_summary_charge_drift,
            abs(summaries[iteration]["electron_count"] - reference_electron_count),
        )
        occupation_layout = _layout(blocks, iteration, "occupation")
        if occupation_layout != reference_layout:
            raise HartreeTraceValidationError("occupation layout differs from h0")
        occupation_count = 0.0
        for spin, kpoint in sorted(reference_layout):
            occupation = _matrix(blocks, iteration, "occupation", spin, kpoint)
            if occupation.shape[0] != 1 or _max_abs(occupation.imag) > 1.0e-14:
                raise HartreeTraceValidationError("invalid occupation matrix")
            occupation_count += float(np.sum(occupation.real))
        occupation_counts[iteration] = occupation_count
        max_trace_summary_count_difference = max(
            max_trace_summary_count_difference,
            abs(occupation_count - summaries[iteration]["electron_count"]),
        )

        if iteration == 0:
            continue
        required = ("exx", "vc", "delta_vh", "raw_h")
        if any(_layout(blocks, iteration, component) != reference_layout for component in required):
            raise HartreeTraceValidationError("Hartree update layout differs from h0")
        delta_max = 0.0
        for spin, kpoint in sorted(reference_layout):
            h0 = _matrix(blocks, 0, "h0", spin, kpoint)
            vxc = _matrix(blocks, 0, "vxc_dft", spin, kpoint)
            exx = _matrix(blocks, iteration, "exx", spin, kpoint)
            vc = _matrix(blocks, iteration, "vc", spin, kpoint)
            delta = _matrix(blocks, iteration, "delta_vh", spin, kpoint)
            raw = _matrix(blocks, iteration, "raw_h", spin, kpoint)
            expected_raw = h0 - vxc + exx + vc + delta
            if raw.shape != expected_raw.shape:
                raise HartreeTraceValidationError("raw_h closure shape differs")
            closure_max_abs = max(
                closure_max_abs, _max_abs(raw - expected_raw)
            )
            closure_relative = max(
                closure_relative, _relative(raw, expected_raw)
            )
            hermiticity_max = max(
                hermiticity_max,
                _max_abs(delta - delta.conj().T),
                _max_abs(raw - raw.conj().T),
            )
            delta_max = max(delta_max, _max_abs(delta))
        iteration_delta_max[iteration] = delta_max

    closure_passed = (
        closure_max_abs <= closure_tolerance_ha
        and closure_relative <= closure_relative_tolerance
    )
    charge_passed = (
        max_summary_charge_drift <= electron_count_tolerance
        and max_trace_summary_count_difference <= electron_count_tolerance
    )
    hermiticity_passed = hermiticity_max <= hermiticity_tolerance_ha
    response_passed = (
        iteration_delta_max[1] <= initial_zero_tolerance_ha
        and max(
            value for iteration, value in iteration_delta_max.items()
            if iteration >= 2
        ) >= response_nonzero_tolerance_ha
    )
    return {
        "schema": SCHEMA,
        "passed": (
            closure_passed and charge_passed and hermiticity_passed and response_passed
        ),
        "acceptance_scope": "current_grid_hartree_trace_invariants",
        "legacy_same_dataset_acceptance": False,
        "iterations": iterations,
        "reference_electron_count": reference_electron_count,
        "occupation_electron_counts": occupation_counts,
        "max_summary_charge_drift": max_summary_charge_drift,
        "max_trace_summary_electron_count_difference": (
            max_trace_summary_count_difference
        ),
        "raw_closure_max_abs_ha": closure_max_abs,
        "raw_closure_relative_frobenius": closure_relative,
        "delta_vh_hermiticity_max_abs_ha": hermiticity_max,
        "delta_vh_max_abs_ha_by_iteration": iteration_delta_max,
        "closure_passed": closure_passed,
        "charge_passed": charge_passed,
        "hermiticity_passed": hermiticity_passed,
        "response_passed": response_passed,
        "contract": {
            "symmetry": contract["symmetry"],
            "hartree_coulomb": contract["hartree_coulomb"],
            "hartree_normalization": contract["hartree_normalization"],
        },
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("matrix_trace", type=Path)
    parser.add_argument("iteration_trace", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--expected-iterations", type=int, default=2)
    parser.add_argument("--expected-symmetry", default="exx_on_gw_on_rpa_on")
    parser.add_argument("--closure-tolerance-ha", type=float, default=1.0e-10)
    parser.add_argument("--closure-relative-tolerance", type=float, default=1.0e-8)
    parser.add_argument("--hermiticity-tolerance-ha", type=float, default=1.0e-10)
    parser.add_argument("--electron-count-tolerance", type=float, default=1.0e-10)
    parser.add_argument("--initial-zero-tolerance-ha", type=float, default=1.0e-10)
    parser.add_argument("--response-nonzero-tolerance-ha", type=float, default=1.0e-12)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        report = validate_hartree_trace(
            args.matrix_trace.read_text(encoding="utf-8"),
            args.iteration_trace.read_text(encoding="utf-8"),
            expected_iterations=args.expected_iterations,
            expected_symmetry=args.expected_symmetry,
            closure_tolerance_ha=args.closure_tolerance_ha,
            closure_relative_tolerance=args.closure_relative_tolerance,
            hermiticity_tolerance_ha=args.hermiticity_tolerance_ha,
            electron_count_tolerance=args.electron_count_tolerance,
            initial_zero_tolerance_ha=args.initial_zero_tolerance_ha,
            response_nonzero_tolerance_ha=args.response_nonzero_tolerance_ha,
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
