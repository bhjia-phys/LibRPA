#!/usr/bin/env python3
"""Compare legacy and current QSGW band tables over several iterations."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path


SCHEMA = "librpa-qsgw-band-iteration-comparison-v1"


class ComparisonError(ValueError):
    pass


@dataclass(frozen=True)
class BandRow:
    coordinate: tuple[float, float, float]
    occupations: tuple[float, ...]
    energies_ev: tuple[float, ...]


def _finite_float(text: str, label: str) -> float:
    try:
        value = float(text)
    except ValueError as error:
        raise ComparisonError(f"{label} is not numeric: {text!r}") from error
    if not math.isfinite(value):
        raise ComparisonError(f"{label} is not finite")
    return value


def parse_band_table(path: Path) -> tuple[BandRow, ...]:
    rows: list[BandRow] = []
    n_bands: int | None = None
    for line_number, raw in enumerate(
        path.read_text(encoding="ascii").splitlines(), start=1
    ):
        fields = raw.split()
        if not fields:
            continue
        if len(fields) < 6 or (len(fields) - 4) % 2:
            raise ComparisonError(
                f"{path}:{line_number} has an invalid column count"
            )
        try:
            index = int(fields[0])
        except ValueError as error:
            raise ComparisonError(
                f"{path}:{line_number} has an invalid k-point index"
            ) from error
        if index != len(rows) + 1:
            raise ComparisonError(
                f"{path}:{line_number} has a non-contiguous k-point index"
            )
        row_n_bands = (len(fields) - 4) // 2
        if n_bands is None:
            n_bands = row_n_bands
        elif row_n_bands != n_bands:
            raise ComparisonError(f"{path} has inconsistent band counts")
        coordinate = tuple(
            _finite_float(fields[index], f"{path}:{line_number} coordinate")
            for index in range(1, 4)
        )
        occupations = tuple(
            _finite_float(
                fields[4 + 2 * band],
                f"{path}:{line_number} occupation {band}",
            )
            for band in range(row_n_bands)
        )
        energies = tuple(
            _finite_float(
                fields[5 + 2 * band],
                f"{path}:{line_number} energy {band}",
            )
            for band in range(row_n_bands)
        )
        rows.append(BandRow(coordinate, occupations, energies))
    if not rows:
        raise ComparisonError(f"empty band table: {path}")
    return tuple(rows)


def indirect_gap_ev(rows: tuple[BandRow, ...], occupied_bands: int) -> float:
    n_bands = len(rows[0].energies_ev)
    if occupied_bands < 1 or occupied_bands >= n_bands:
        raise ComparisonError(
            f"occupied band count {occupied_bands} is outside 1..{n_bands - 1}"
        )
    valence_maximum = max(
        row.energies_ev[occupied_bands - 1] for row in rows
    )
    conduction_minimum = min(
        row.energies_ev[occupied_bands] for row in rows
    )
    return conduction_minimum - valence_maximum


def compare_tables(
    legacy: tuple[BandRow, ...],
    candidate: tuple[BandRow, ...],
    occupied_bands: int,
    coordinate_tolerance: float,
    energy_tolerance_ev: float,
    gap_tolerance_ev: float,
) -> dict[str, object]:
    if len(legacy) != len(candidate):
        raise ComparisonError("legacy and candidate k-point counts differ")
    legacy_bands = len(legacy[0].energies_ev)
    candidate_bands = len(candidate[0].energies_ev)
    if legacy_bands != candidate_bands:
        raise ComparisonError("legacy and candidate band counts differ")

    maximum_coordinate = 0.0
    maximum_energy = 0.0
    maximum_location = [0, 0]
    for kpoint, (legacy_row, candidate_row) in enumerate(
        zip(legacy, candidate)
    ):
        if (
            len(legacy_row.energies_ev) != legacy_bands
            or len(candidate_row.energies_ev) != candidate_bands
        ):
            raise ComparisonError("band count changes between k points")
        maximum_coordinate = max(
            maximum_coordinate,
            max(
                abs(left - right)
                for left, right in zip(
                    legacy_row.coordinate, candidate_row.coordinate
                )
            ),
        )
        for band, (left, right) in enumerate(
            zip(legacy_row.energies_ev, candidate_row.energies_ev)
        ):
            difference = abs(left - right)
            if difference > maximum_energy:
                maximum_energy = difference
                maximum_location = [kpoint, band]

    legacy_gap = indirect_gap_ev(legacy, occupied_bands)
    candidate_gap = indirect_gap_ev(candidate, occupied_bands)
    gap_difference = abs(legacy_gap - candidate_gap)
    passed = (
        maximum_coordinate <= coordinate_tolerance
        and maximum_energy <= energy_tolerance_ev
        and gap_difference <= gap_tolerance_ev
    )
    return {
        "passed": passed,
        "n_kpoints": len(legacy),
        "n_bands": legacy_bands,
        "occupied_bands": occupied_bands,
        "coordinate_max_abs": maximum_coordinate,
        "coordinate_tolerance": coordinate_tolerance,
        "energy_max_abs_ev": maximum_energy,
        "energy_max_abs_location_zero_based": maximum_location,
        "energy_tolerance_ev": energy_tolerance_ev,
        "legacy_gap_ev": legacy_gap,
        "candidate_gap_ev": candidate_gap,
        "gap_abs_difference_ev": gap_difference,
        "gap_tolerance_ev": gap_tolerance_ev,
    }


def _positive_int(text: str) -> int:
    value = int(text)
    if value <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return value


def _nonnegative_float(text: str) -> float:
    value = float(text)
    if not math.isfinite(value) or value < 0.0:
        raise argparse.ArgumentTypeError("value must be finite and nonnegative")
    return value


def parse_iteration_list(text: str) -> tuple[int, ...]:
    try:
        values = tuple(int(item) for item in text.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "iterations must be comma-separated integers"
        ) from error
    if not values or any(value <= 0 for value in values):
        raise argparse.ArgumentTypeError("iterations must be positive")
    if tuple(sorted(set(values))) != values:
        raise argparse.ArgumentTypeError(
            "iterations must be unique and strictly increasing"
        )
    return values


def run(args: argparse.Namespace) -> dict[str, object]:
    reports: dict[str, object] = {}
    passed = True
    for iteration in args.iterations:
        iteration_reports: dict[str, object] = {}
        for spin in range(1, args.spins + 1):
            filename = f"QSGW_band_spin_{spin}_{iteration}.dat"
            legacy_path = args.legacy_dir / filename
            candidate_path = args.candidate_dir / filename
            if not legacy_path.is_file() or not candidate_path.is_file():
                raise ComparisonError(
                    f"missing legacy or candidate band table: {filename}"
                )
            report = compare_tables(
                parse_band_table(legacy_path),
                parse_band_table(candidate_path),
                args.occupied_bands,
                args.coordinate_tolerance,
                args.energy_tolerance_ev,
                args.gap_tolerance_ev,
            )
            iteration_reports[str(spin)] = report
            passed = passed and bool(report["passed"])
        reports[str(iteration)] = iteration_reports
    return {
        "schema": SCHEMA,
        "passed": passed,
        "legacy_dir": str(args.legacy_dir.resolve()),
        "candidate_dir": str(args.candidate_dir.resolve()),
        "iterations": list(args.iterations),
        "spins": args.spins,
        "occupied_bands": args.occupied_bands,
        "comparisons": reports,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--legacy-dir", type=Path, required=True)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--iterations", type=parse_iteration_list, required=True)
    parser.add_argument("--spins", type=_positive_int, default=1)
    parser.add_argument("--occupied-bands", type=_positive_int, required=True)
    parser.add_argument(
        "--coordinate-tolerance", type=_nonnegative_float, default=1.0e-7
    )
    parser.add_argument(
        "--energy-tolerance-ev", type=_nonnegative_float, default=1.0e-4
    )
    parser.add_argument(
        "--gap-tolerance-ev", type=_nonnegative_float, default=2.0e-4
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = run(args)
    except (ComparisonError, OSError) as error:
        parser.error(str(error))
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="ascii",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
