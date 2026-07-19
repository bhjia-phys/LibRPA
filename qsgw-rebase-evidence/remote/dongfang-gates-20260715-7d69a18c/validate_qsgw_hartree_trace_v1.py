#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


RowKey = tuple[int, str, int, int, int, int]


def parse_iterations(specification: str) -> list[int]:
    if ":" in specification:
        start_text, stop_text = specification.split(":", 1)
        start, stop = int(start_text), int(stop_text)
        if start < 0 or stop < start:
            raise ValueError("invalid iteration range")
        return list(range(start, stop + 1))
    result = [int(value) for value in specification.split(",")]
    if not result or any(value < 0 for value in result):
        raise ValueError("invalid iteration list")
    return result


def parse_kpoint_weights(text: str) -> dict[int, float]:
    lines = [line.split() for line in text.splitlines() if line.strip()]
    if len(lines) < 3 or len(lines[1]) < 1:
        raise ValueError("invalid bz_sampling_out")
    count = int(lines[1][0])
    if len(lines) < count + 2:
        raise ValueError("truncated bz_sampling_out")
    result: dict[int, float] = {}
    for fields in lines[2:count + 2]:
        if len(fields) < 2:
            raise ValueError("invalid k-point weight row")
        index = int(fields[0]) - 1
        weight = float(fields[1])
        if index in result or not math.isfinite(weight) or weight <= 0.0:
            raise ValueError("invalid or duplicate k-point weight")
        result[index] = weight
    if set(result) != set(range(count)):
        raise ValueError("non-contiguous k-point indices")
    if not math.isclose(sum(result.values()), 1.0, abs_tol=1.0e-12):
        raise ValueError("k-point weights do not sum to one")
    return result


def parse_trace_rows(
    text: str,
    label: str,
    iterations: set[int],
    channel: int,
) -> dict[RowKey, complex]:
    result: dict[RowKey, complex] = {}
    for line_number, line in enumerate(text.splitlines(), 1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        fields = line.split()
        if len(fields) != 11:
            raise ValueError(f"{label}:{line_number}: expected 11 fields")
        iteration = int(fields[0])
        observed_channel = int(fields[1])
        if iteration not in iterations or observed_channel != channel:
            continue
        component = fields[2]
        if component not in {"occupation", "delta_vh"}:
            continue
        spin, kpoint = int(fields[3]), int(fields[4])
        frequency_index = int(fields[5])
        frequency = float(fields[6])
        row, column = int(fields[7]), int(fields[8])
        value = complex(float(fields[9]), float(fields[10]))
        if frequency_index != -1 or frequency != 0.0:
            raise ValueError(f"{label}:{line_number}: static row has frequency")
        if row < 0 or column < 0 or not (
            math.isfinite(value.real) and math.isfinite(value.imag)
        ):
            raise ValueError(f"{label}:{line_number}: invalid matrix row")
        key = (iteration, component, spin, kpoint, row, column)
        if key in result:
            raise ValueError(f"{label}:{line_number}: duplicate row {key}")
        result[key] = value
    return result


def parse_summary_electron_counts(
    text: str, iterations: set[int]
) -> dict[int, float]:
    result: dict[int, float] = {}
    for line_number, line in enumerate(text.splitlines(), 1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        fields = line.split()
        if len(fields) < 7:
            raise ValueError(
                f"current iteration trace:{line_number}: too few fields"
            )
        iteration = int(fields[0])
        if iteration not in iterations:
            continue
        count = float(fields[6])
        if iteration in result or not math.isfinite(count):
            raise ValueError("invalid current iteration electron count")
        result[iteration] = count
    if set(result) != iterations:
        raise ValueError("current iteration electron counts are incomplete")
    return result


def occupation_counts(
    rows: dict[RowKey, complex],
    weights: dict[int, float],
    iterations: set[int],
    label: str,
) -> dict[int, float]:
    result = {iteration: 0.0 for iteration in iterations}
    observed = {iteration: set() for iteration in iterations}
    for key, value in rows.items():
        iteration, component, _spin, kpoint, row, _column = key
        if component != "occupation":
            continue
        if row != 0 or kpoint not in weights or abs(value.imag) > 1.0e-14:
            raise ValueError(f"{label}: invalid occupation row {key}")
        result[iteration] += weights[kpoint] * value.real
        observed[iteration].add(kpoint)
    expected_kpoints = set(weights)
    for iteration in iterations:
        if observed[iteration] != expected_kpoints:
            raise ValueError(
                f"{label}: iteration {iteration} occupation k-grid incomplete"
            )
    return result


def delta_rows(
    rows: dict[RowKey, complex], iterations: set[int], label: str
) -> dict[tuple[int, int, int, int, int], complex]:
    result = {
        (iteration, spin, kpoint, row, column): value
        for (iteration, component, spin, kpoint, row, column), value
        in rows.items()
        if component == "delta_vh" and iteration > 0
    }
    for iteration in sorted(iterations - {0}):
        if not any(key[0] == iteration for key in result):
            raise ValueError(f"{label}: iteration {iteration} lacks delta_vh")
    return result


def hermiticity_residual(
    rows: dict[tuple[int, int, int, int, int], complex]
) -> float:
    maximum = 0.0
    for (iteration, spin, kpoint, row, column), value in rows.items():
        transpose = (iteration, spin, kpoint, column, row)
        if transpose not in rows:
            raise ValueError(f"delta_vh matrix is incomplete at {transpose}")
        maximum = max(maximum, abs(value - rows[transpose].conjugate()))
    return maximum


def compare_hartree_traces(
    *,
    old_matrix_text: str,
    current_matrix_text: str,
    current_iteration_text: str,
    bz_sampling_text: str,
    iterations: list[int],
    channel: int = 0,
    initial_delta_tolerance_ha: float = 1.0e-10,
    charge_tolerance: float = 1.0e-10,
    matrix_max_abs_tolerance_ha: float = 1.0e-8,
    matrix_relative_tolerance: float = 1.0e-8,
    hermiticity_tolerance_ha: float = 1.0e-10,
) -> dict[str, object]:
    if iterations != list(range(iterations[-1] + 1)) or len(iterations) < 3:
        raise ValueError("Hartree validation requires continuous iterations 0..N, N>=2")
    tolerances = (
        initial_delta_tolerance_ha,
        charge_tolerance,
        matrix_max_abs_tolerance_ha,
        matrix_relative_tolerance,
        hermiticity_tolerance_ha,
    )
    if any(not math.isfinite(value) or value < 0.0 for value in tolerances):
        raise ValueError("invalid Hartree validation tolerance")
    selected = set(iterations)
    weights = parse_kpoint_weights(bz_sampling_text)
    old_rows = parse_trace_rows(
        old_matrix_text, "legacy matrix trace", selected, channel
    )
    current_rows = parse_trace_rows(
        current_matrix_text, "current matrix trace", selected, channel
    )
    old_counts = occupation_counts(old_rows, weights, selected, "legacy")
    current_counts = occupation_counts(
        current_rows, weights, selected, "current"
    )
    summary_counts = parse_summary_electron_counts(
        current_iteration_text, selected
    )
    reference_count = current_counts[0]
    max_charge_drift = max(
        abs(value - reference_count)
        for values in (old_counts, current_counts, summary_counts)
        for value in values.values()
    )
    max_old_current_count = max(
        abs(old_counts[iteration] - current_counts[iteration])
        for iteration in iterations
    )

    old_delta = delta_rows(old_rows, selected, "legacy")
    current_delta = delta_rows(current_rows, selected, "current")
    if set(old_delta) != set(current_delta):
        raise ValueError("legacy/current delta_vh row layout differs")
    first_iteration = 1
    old_initial = max(
        abs(value) for key, value in old_delta.items()
        if key[0] == first_iteration
    )
    current_initial = max(
        abs(value) for key, value in current_delta.items()
        if key[0] == first_iteration
    )
    differences = [
        old_delta[key] - current_delta[key] for key in old_delta
    ]
    maximum_difference = max(abs(value) for value in differences)
    difference_norm = math.sqrt(sum(abs(value) ** 2 for value in differences))
    old_norm = math.sqrt(sum(abs(value) ** 2 for value in old_delta.values()))
    current_norm = math.sqrt(
        sum(abs(value) ** 2 for value in current_delta.values())
    )
    scale = max(old_norm, current_norm, 1.0e-300)
    relative_difference = difference_norm / scale
    old_hermiticity = hermiticity_residual(old_delta)
    current_hermiticity = hermiticity_residual(current_delta)
    passed = (
        old_initial <= initial_delta_tolerance_ha
        and current_initial <= initial_delta_tolerance_ha
        and max_charge_drift <= charge_tolerance
        and max_old_current_count <= charge_tolerance
        and maximum_difference <= matrix_max_abs_tolerance_ha
        and relative_difference <= matrix_relative_tolerance
        and old_hermiticity <= hermiticity_tolerance_ha
        and current_hermiticity <= hermiticity_tolerance_ha
    )
    return {
        "passed": passed,
        "iterations": iterations,
        "channel": channel,
        "reference_electron_count": reference_count,
        "legacy_occupation_electron_counts": old_counts,
        "current_occupation_electron_counts": current_counts,
        "current_summary_electron_counts": summary_counts,
        "max_charge_drift": max_charge_drift,
        "max_old_current_electron_count_difference": max_old_current_count,
        "charge_tolerance": charge_tolerance,
        "legacy_initial_delta_vh_max_abs_ha": old_initial,
        "current_initial_delta_vh_max_abs_ha": current_initial,
        "initial_delta_tolerance_ha": initial_delta_tolerance_ha,
        "delta_vh_max_abs_difference_ha": maximum_difference,
        "delta_vh_relative_frobenius": relative_difference,
        "matrix_max_abs_tolerance_ha": matrix_max_abs_tolerance_ha,
        "matrix_relative_tolerance": matrix_relative_tolerance,
        "legacy_delta_vh_hermiticity_max_abs_ha": old_hermiticity,
        "current_delta_vh_hermiticity_max_abs_ha": current_hermiticity,
        "hermiticity_tolerance_ha": hermiticity_tolerance_ha,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("old_matrix_trace", type=Path)
    parser.add_argument("current_matrix_trace", type=Path)
    parser.add_argument("current_iteration_trace", type=Path)
    parser.add_argument("bz_sampling", type=Path)
    parser.add_argument("output_json", type=Path)
    parser.add_argument("--iterations", default="0:2")
    parser.add_argument("--channel", type=int, default=0)
    parser.add_argument("--initial-delta-tolerance-ha", type=float, default=1.0e-10)
    parser.add_argument("--charge-tolerance", type=float, default=1.0e-10)
    parser.add_argument("--matrix-max-abs-tolerance-ha", type=float, default=1.0e-8)
    parser.add_argument("--matrix-relative-tolerance", type=float, default=1.0e-8)
    parser.add_argument("--hermiticity-tolerance-ha", type=float, default=1.0e-10)
    args = parser.parse_args()
    try:
        report = compare_hartree_traces(
            old_matrix_text=args.old_matrix_trace.read_text(encoding="utf-8"),
            current_matrix_text=args.current_matrix_trace.read_text(encoding="utf-8"),
            current_iteration_text=args.current_iteration_trace.read_text(encoding="utf-8"),
            bz_sampling_text=args.bz_sampling.read_text(encoding="utf-8"),
            iterations=parse_iterations(args.iterations),
            channel=args.channel,
            initial_delta_tolerance_ha=args.initial_delta_tolerance_ha,
            charge_tolerance=args.charge_tolerance,
            matrix_max_abs_tolerance_ha=args.matrix_max_abs_tolerance_ha,
            matrix_relative_tolerance=args.matrix_relative_tolerance,
            hermiticity_tolerance_ha=args.hermiticity_tolerance_ha,
        )
    except Exception as error:
        report = {"passed": False, "error": str(error)}
    args.output_json.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    raise SystemExit(0 if report["passed"] else 2)


if __name__ == "__main__":
    main()
