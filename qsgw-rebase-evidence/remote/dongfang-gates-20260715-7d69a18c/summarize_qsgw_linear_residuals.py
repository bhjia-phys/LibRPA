#!/usr/bin/env python3
"""Compare legacy, current-derived, and current-reported QSGW residuals."""

from __future__ import annotations

import argparse
import csv
import json
import math
import pathlib
from collections import defaultdict
from typing import Sequence

import numpy as np


MatrixKey = tuple[int, str, int, int]


def parse_iterations(specification: str) -> list[int]:
    if ":" not in specification:
        values = [int(value) for value in specification.split(",")]
    else:
        first, last = (int(value) for value in specification.split(":"))
        values = list(range(first, last + 1))
    if not values or values[0] != 0 or values != list(range(values[-1] + 1)):
        raise ValueError("iterations must be a continuous range starting at zero")
    return values


def parse_matrix_trace(text: str, iterations: set[int]) -> dict[MatrixKey, np.ndarray]:
    entries: dict[MatrixKey, list[tuple[int, int, complex]]] = defaultdict(list)
    wanted = {"h0", "raw_h", "mixed_h"}
    for line_number, line in enumerate(text.splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) != 11:
            raise ValueError(f"matrix trace line {line_number} must have 11 columns")
        iteration, channel = int(fields[0]), int(fields[1])
        component = fields[2]
        frequency_index = int(fields[5])
        if (
            iteration not in iterations
            or channel != 0
            or component not in wanted
            or frequency_index != -1
        ):
            continue
        key = (iteration, component, int(fields[3]), int(fields[4]))
        value = complex(float(fields[9]), float(fields[10]))
        if not math.isfinite(value.real) or not math.isfinite(value.imag):
            raise ValueError(f"matrix trace line {line_number} is non-finite")
        entries[key].append((int(fields[7]), int(fields[8]), value))
    matrices: dict[MatrixKey, np.ndarray] = {}
    for key, values in entries.items():
        nrows = max(value[0] for value in values) + 1
        ncols = max(value[1] for value in values) + 1
        if nrows != ncols or len(values) != nrows * ncols:
            raise ValueError(f"matrix block {key} is incomplete or non-square")
        matrix = np.empty((nrows, ncols), dtype=np.complex128)
        occupied = set()
        for row, column, value in values:
            if (row, column) in occupied:
                raise ValueError(f"matrix block {key} has duplicate elements")
            occupied.add((row, column))
            matrix[row, column] = value
        matrices[key] = matrix
    if not matrices:
        raise ValueError("matrix trace has no selected Hamiltonian blocks")
    return matrices


def materialize_legacy_upper(matrix: np.ndarray) -> np.ndarray:
    result = np.array(matrix, dtype=np.complex128, copy=True)
    diagonal = np.diag_indices_from(result)
    result[diagonal] = result[diagonal].real
    lower = np.tril_indices_from(result, k=-1)
    result[lower] = result.T.conj()[lower]
    return result


def matrix_residual_norms(previous, raw) -> tuple[float, float]:
    difference = np.asarray(raw, dtype=np.complex128) - np.asarray(
        previous, dtype=np.complex128
    )
    return float(np.linalg.norm(difference)), float(np.max(np.abs(difference)))


def residual_trajectory(
    matrices: dict[MatrixKey, np.ndarray],
    iterations: list[int],
    *,
    legacy_upper: bool,
) -> dict[int, tuple[float, float]]:
    blocks = {
        (spin, kpoint)
        for iteration, component, spin, kpoint in matrices
        if iteration == 0 and component == "h0"
    }
    if not blocks:
        raise ValueError("iteration zero has no h0 blocks")
    previous = {
        block: matrices[(0, "h0", *block)]
        for block in blocks
    }
    trajectory = {0: (0.0, 0.0)}
    for iteration in iterations[1:]:
        raw_blocks = {
            (spin, kpoint)
            for row_iteration, component, spin, kpoint in matrices
            if row_iteration == iteration and component == "raw_h"
        }
        mixed_blocks = {
            (spin, kpoint)
            for row_iteration, component, spin, kpoint in matrices
            if row_iteration == iteration and component == "mixed_h"
        }
        if raw_blocks != blocks or mixed_blocks != blocks:
            raise ValueError(f"iteration {iteration} Hamiltonian block set differs")
        sum_square = 0.0
        maximum = 0.0
        next_previous = {}
        for block in sorted(blocks):
            raw = matrices[(iteration, "raw_h", *block)]
            mixed = matrices[(iteration, "mixed_h", *block)]
            if legacy_upper:
                raw = materialize_legacy_upper(raw)
                mixed = materialize_legacy_upper(mixed)
            difference = raw - previous[block]
            sum_square += float(np.vdot(difference, difference).real)
            maximum = max(maximum, float(np.max(np.abs(difference))))
            next_previous[block] = mixed
        trajectory[iteration] = (math.sqrt(sum_square), maximum)
        previous = next_previous
    return trajectory


def parse_iteration_trace(text: str, iterations: set[int]) -> dict[int, dict[str, float]]:
    rows = {}
    for line_number, line in enumerate(text.splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) < 16:
            raise ValueError(f"iteration trace line {line_number} has too few columns")
        iteration = int(fields[0])
        if iteration not in iterations:
            continue
        values = {
            "max_delta_ev": float(fields[1]),
            "residual_l2_ha": float(fields[2]),
            "residual_max_ha": float(fields[3]),
            "efermi_ev": float(fields[4]),
            "gap_ev": float(fields[5]),
            "electron_count": float(fields[6]),
            "requested_mode": int(fields[7]),
            "applied_mode": int(fields[8]),
            "beta": float(fields[9]),
            "fallback": int(fields[10]),
        }
        if iteration in rows or not all(
            math.isfinite(value)
            for key, value in values.items()
            if key not in {"requested_mode", "applied_mode", "fallback"}
        ):
            raise ValueError(f"iteration trace line {line_number} is invalid")
        rows[iteration] = values
    if set(rows) != iterations:
        raise ValueError("iteration trace does not cover the selected range")
    return rows


def relative_difference(first: float, second: float) -> float:
    return abs(first - second) / max(abs(first), abs(second), 1.0e-30)


def compare_residuals(
    old_text: str,
    current_text: str,
    iteration_text: str,
    iterations: list[int],
    *,
    absolute_tolerance: float,
    relative_tolerance: float,
) -> dict[str, object]:
    if absolute_tolerance < 0.0 or relative_tolerance < 0.0:
        raise ValueError("tolerances must be nonnegative")
    selected = set(iterations)
    old = residual_trajectory(
        parse_matrix_trace(old_text, selected), iterations, legacy_upper=True
    )
    current = residual_trajectory(
        parse_matrix_trace(current_text, selected), iterations, legacy_upper=False
    )
    reported = parse_iteration_trace(iteration_text, selected)
    metrics = {
        "max_old_current_l2_abs_diff_ha": 0.0,
        "max_old_current_l2_relative_diff": 0.0,
        "max_old_current_max_abs_diff_ha": 0.0,
        "max_old_current_max_relative_diff": 0.0,
        "max_current_reported_l2_abs_diff_ha": 0.0,
        "max_current_reported_l2_relative_diff": 0.0,
        "max_current_reported_max_abs_diff_ha": 0.0,
        "max_current_reported_max_relative_diff": 0.0,
    }
    trajectory = []
    comparisons = []
    for iteration in iterations:
        old_l2, old_max = old[iteration]
        current_l2, current_max = current[iteration]
        reported_l2 = reported[iteration]["residual_l2_ha"]
        reported_max = reported[iteration]["residual_max_ha"]
        pairs = (
            ("old_current_l2", old_l2, current_l2),
            ("old_current_max", old_max, current_max),
            ("current_reported_l2", current_l2, reported_l2),
            ("current_reported_max", current_max, reported_max),
        )
        for label, first, second in pairs:
            absolute = abs(first - second)
            relative = relative_difference(first, second)
            metrics[f"max_{label}_abs_diff_ha"] = max(
                metrics[f"max_{label}_abs_diff_ha"], absolute
            )
            metrics[f"max_{label}_relative_diff"] = max(
                metrics[f"max_{label}_relative_diff"], relative
            )
            comparisons.append(
                absolute <= absolute_tolerance
                or relative <= relative_tolerance
            )
        trajectory.append(
            {
                **reported[iteration],
                "iteration": iteration,
                "residual_l2_ha": current_l2,
                "residual_max_ha": current_max,
                "old_residual_l2_ha": old_l2,
                "old_residual_max_ha": old_max,
                "reported_residual_l2_ha": reported_l2,
                "reported_residual_max_ha": reported_max,
            }
        )
    return {
        "passed": all(comparisons),
        "iterations": iterations,
        "absolute_tolerance_ha": absolute_tolerance,
        "relative_tolerance": relative_tolerance,
        **metrics,
        "trajectory": trajectory,
    }


def write_csv(path: pathlib.Path, trajectory: list[dict[str, object]]) -> None:
    fields = [
        "iteration",
        "max_delta_ev",
        "gap_ev",
        "residual_l2_ha",
        "residual_max_ha",
        "old_residual_l2_ha",
        "old_residual_max_ha",
        "reported_residual_l2_ha",
        "reported_residual_max_ha",
        "requested_mode",
        "applied_mode",
        "beta",
        "fallback",
    ]
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(trajectory)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("old_trace", type=pathlib.Path)
    parser.add_argument("current_trace", type=pathlib.Path)
    parser.add_argument("current_iteration_trace", type=pathlib.Path)
    parser.add_argument("output_json", type=pathlib.Path)
    parser.add_argument("output_csv", type=pathlib.Path)
    parser.add_argument("--iterations", required=True)
    parser.add_argument("--absolute-tolerance-ha", type=float, default=1.0e-8)
    parser.add_argument("--relative-tolerance", type=float, default=1.0e-8)
    args = parser.parse_args(argv)
    report = compare_residuals(
        args.old_trace.read_text(),
        args.current_trace.read_text(),
        args.current_iteration_trace.read_text(),
        parse_iterations(args.iterations),
        absolute_tolerance=args.absolute_tolerance_ha,
        relative_tolerance=args.relative_tolerance,
    )
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    write_csv(args.output_csv, report["trajectory"])
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
