#!/usr/bin/env python3
"""Verify that QSGW iteration zero preserves the producer mean field."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


HA2EV = 27.211386245988


class InitialStateError(ValueError):
    """Raised when iteration zero differs from the producer mean field."""


def _parse_band_out(path: Path) -> tuple[float, dict[tuple[int, int, int], float]]:
    rows = [line.split() for line in path.read_text(encoding="utf-8").splitlines()]
    if len(rows) < 5 or any(not row for row in rows[:5]):
        raise InitialStateError("band_out header is truncated")
    n_kpoints = int(rows[0][0])
    n_spins = int(rows[1][0])
    n_bands = int(rows[2][0])
    efermi_ha = float(rows[4][0])
    if n_kpoints <= 0 or n_spins <= 0 or n_bands <= 0:
        raise InitialStateError("band_out dimensions are invalid")
    if not math.isfinite(efermi_ha):
        raise InitialStateError("band_out Fermi level is non-finite")

    position = 5
    occupations: dict[tuple[int, int, int], float] = {}
    for kpoint in range(n_kpoints):
        for spin in range(n_spins):
            if position >= len(rows) or len(rows[position]) != 2:
                raise InitialStateError("band_out k-point/spin header is missing")
            observed = tuple(map(int, rows[position]))
            position += 1
            if observed != (kpoint + 1, spin + 1):
                raise InitialStateError("band_out k-point/spin ordering changed")
            for band in range(n_bands):
                if position >= len(rows) or len(rows[position]) < 4:
                    raise InitialStateError("band_out band record is truncated")
                fields = rows[position]
                position += 1
                if int(fields[0]) != band + 1:
                    raise InitialStateError("band_out band ordering changed")
                value = float(fields[1]) / n_kpoints
                if not math.isfinite(value):
                    raise InitialStateError("band_out occupation is non-finite")
                occupations[(spin, kpoint, band)] = value
    return efermi_ha, occupations


def _parse_iteration_zero_efermi(path: Path) -> float:
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        fields = stripped.split()
        if len(fields) != 17:
            raise InitialStateError("QSGW iteration summary layout changed")
        if int(fields[0]) == 0:
            value = float(fields[4]) / HA2EV
            if not math.isfinite(value):
                raise InitialStateError("iteration-zero Fermi level is non-finite")
            return value
    raise InitialStateError("QSGW iteration summary has no iteration zero")


def _parse_trace_occupations(
    path: Path,
) -> dict[tuple[int, int, int], float]:
    occupations: dict[tuple[int, int, int], float] = {}
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        fields = stripped.split()
        if len(fields) != 11:
            raise InitialStateError(
                f"matrix trace:{line_number}: expected 11 columns"
            )
        if (
            int(fields[0]) != 0
            or int(fields[1]) != 0
            or fields[2] != "occupation"
            or int(fields[5]) != -1
        ):
            continue
        if int(fields[7]) != 0:
            raise InitialStateError("occupation trace row index is not zero")
        if abs(float(fields[10])) > 1.0e-15:
            raise InitialStateError("occupation trace has an imaginary value")
        key = (int(fields[3]), int(fields[4]), int(fields[8]))
        if key in occupations:
            raise InitialStateError("occupation trace contains a duplicate state")
        value = float(fields[9])
        if not math.isfinite(value):
            raise InitialStateError("occupation trace has a non-finite value")
        occupations[key] = value
    if not occupations:
        raise InitialStateError("matrix trace has no iteration-zero occupations")
    return occupations


def validate(
    matrix_trace: Path,
    iteration_summary: Path,
    band_out: Path,
    tolerance_ha: float,
    occupation_tolerance: float,
) -> dict[str, object]:
    if not math.isfinite(tolerance_ha) or tolerance_ha < 0.0:
        raise InitialStateError("invalid Fermi-level tolerance")
    if not math.isfinite(occupation_tolerance) or occupation_tolerance < 0.0:
        raise InitialStateError("invalid occupation tolerance")
    input_efermi, input_occupations = _parse_band_out(band_out)
    trace_efermi = _parse_iteration_zero_efermi(iteration_summary)
    trace_occupations = _parse_trace_occupations(matrix_trace)
    if trace_occupations.keys() != input_occupations.keys():
        raise InitialStateError("iteration-zero occupation state set changed")
    occupation_maximum = max(
        abs(trace_occupations[key] - input_occupations[key])
        for key in input_occupations
    )
    efermi_difference = abs(trace_efermi - input_efermi)
    passed = (
        efermi_difference <= tolerance_ha
        and occupation_maximum <= occupation_tolerance
    )
    return {
        "passed": passed,
        "input_efermi_ha": input_efermi,
        "trace_iteration_zero_efermi_ha": trace_efermi,
        "efermi_max_abs_difference_ha": efermi_difference,
        "occupation_state_count": len(input_occupations),
        "occupation_max_abs_difference": occupation_maximum,
        "occupation_normalization": "band_out_occupation_divided_by_n_kpoints",
        "efermi_tolerance_ha": tolerance_ha,
        "occupation_tolerance": occupation_tolerance,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("matrix_trace", type=Path)
    parser.add_argument("iteration_summary", type=Path)
    parser.add_argument("band_out", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--efermi-tolerance-ha", type=float, default=1.0e-12)
    parser.add_argument("--occupation-tolerance", type=float, default=1.0e-12)
    args = parser.parse_args()
    try:
        report = validate(
            args.matrix_trace,
            args.iteration_summary,
            args.band_out,
            args.efermi_tolerance_ha,
            args.occupation_tolerance,
        )
        status = 0 if report["passed"] else 1
    except (OSError, ValueError) as error:
        report = {"passed": False, "error": str(error)}
        status = 1
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return status


if __name__ == "__main__":
    raise SystemExit(main())
