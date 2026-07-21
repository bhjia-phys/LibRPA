#!/usr/bin/env python3
"""Locate per-band differences between two legacy band-path output files."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path


SCHEMA = "librpa-legacy-band0-text-difference-analysis-v1"


class AnalysisError(ValueError):
    pass


@dataclass(frozen=True)
class BandRow:
    index: int
    coordinate: tuple[float, float, float]
    occupations: tuple[float, ...]
    energies_ev: tuple[float, ...]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_band_text(text: str, label: str) -> tuple[BandRow, ...]:
    rows: list[BandRow] = []
    n_bands: int | None = None
    for line_number, raw_line in enumerate(text.splitlines(), 1):
        line = raw_line.strip()
        if not line:
            continue
        fields = line.split()
        if len(fields) < 6 or (len(fields) - 4) % 2 != 0:
            raise AnalysisError(
                f"{label}:{line_number}: invalid band-path row width"
            )
        try:
            index = int(fields[0])
            coordinate = tuple(float(value) for value in fields[1:4])
            occupations = tuple(float(value) for value in fields[4::2])
            energies = tuple(float(value) for value in fields[5::2])
        except ValueError as error:
            raise AnalysisError(
                f"{label}:{line_number}: invalid numeric value"
            ) from error
        if index != len(rows) + 1:
            raise AnalysisError(
                f"{label}:{line_number}: unexpected row index {index}"
            )
        if n_bands is None:
            n_bands = len(energies)
        if len(energies) != n_bands or len(occupations) != n_bands:
            raise AnalysisError(
                f"{label}:{line_number}: inconsistent band count"
            )
        if not all(
            math.isfinite(value)
            for value in (*coordinate, *occupations, *energies)
        ):
            raise AnalysisError(f"{label}:{line_number}: non-finite value")
        rows.append(
            BandRow(index, coordinate, occupations, energies)
        )
    if not rows:
        raise AnalysisError(f"{label}: empty band-path output")
    return tuple(rows)


def read_band_file(path: Path) -> tuple[BandRow, ...]:
    return parse_band_text(path.read_text(), str(path))


def band_edges(rows: tuple[BandRow, ...]) -> dict[str, object]:
    occupied: list[tuple[float, int, int]] = []
    unoccupied: list[tuple[float, int, int]] = []
    for row in rows:
        for band, (occupation, energy) in enumerate(
            zip(row.occupations, row.energies_ev), 1
        ):
            target = occupied if occupation > 1.0e-8 else unoccupied
            target.append((energy, row.index, band))
    if not occupied or not unoccupied:
        raise AnalysisError("band output has no occupied/unoccupied partition")
    valence = max(occupied)
    conduction = min(unoccupied)
    return {
        "vbm_ev": valence[0],
        "vbm_location": {"kpoint": valence[1], "band": valence[2]},
        "cbm_ev": conduction[0],
        "cbm_location": {"kpoint": conduction[1], "band": conduction[2]},
        "gap_ev": conduction[0] - valence[0],
    }


def compare_rows(
    reference: tuple[BandRow, ...],
    observed: tuple[BandRow, ...],
) -> dict[str, object]:
    if len(reference) != len(observed):
        raise AnalysisError("band outputs have different k-point counts")
    n_bands = len(reference[0].energies_ev)
    if any(len(row.energies_ev) != n_bands for row in observed):
        raise AnalysisError("band outputs have different band counts")

    coordinate_maximum = 0.0
    occupation_maximum = 0.0
    energy_maximum = -1.0
    energy_location: dict[str, int] = {}
    sum_difference_sq = 0.0
    sum_reference_sq = 0.0
    per_band: list[dict[str, object]] = []
    for band in range(n_bands):
        band_maximum = -1.0
        band_kpoint = -1
        for expected_row, actual_row in zip(reference, observed):
            if expected_row.index != actual_row.index:
                raise AnalysisError("band row indices differ")
            coordinate_maximum = max(
                coordinate_maximum,
                *(abs(a - b) for a, b in zip(
                    expected_row.coordinate, actual_row.coordinate
                )),
            )
            occupation_maximum = max(
                occupation_maximum,
                abs(
                    expected_row.occupations[band]
                    - actual_row.occupations[band]
                ),
            )
            difference = abs(
                expected_row.energies_ev[band]
                - actual_row.energies_ev[band]
            )
            sum_difference_sq += difference * difference
            sum_reference_sq += expected_row.energies_ev[band] ** 2
            if difference > band_maximum:
                band_maximum = difference
                band_kpoint = expected_row.index
            if difference > energy_maximum:
                energy_maximum = difference
                energy_location = {
                    "kpoint": expected_row.index,
                    "band": band + 1,
                }
        per_band.append(
            {
                "band": band + 1,
                "max_abs_diff_ev": band_maximum,
                "kpoint": band_kpoint,
            }
        )
    reference_edges = band_edges(reference)
    observed_edges = band_edges(observed)
    return {
        "n_kpoints": len(reference),
        "n_bands": n_bands,
        "coordinate_max_abs": coordinate_maximum,
        "occupation_max_abs": occupation_maximum,
        "energy_max_abs_diff_ev": energy_maximum,
        "energy_max_location": energy_location,
        "energy_relative_frobenius": (
            math.sqrt(sum_difference_sq) / math.sqrt(sum_reference_sq)
            if sum_reference_sq
            else math.sqrt(sum_difference_sq)
        ),
        "reference_edges": reference_edges,
        "observed_edges": observed_edges,
        "gap_abs_diff_ev": abs(
            float(reference_edges["gap_ev"])
            - float(observed_edges["gap_ev"])
        ),
        "per_band": per_band,
        "near_gap_bands": per_band[: min(12, n_bands)],
    }


def compare_files(reference_path: Path, observed_path: Path) -> dict[str, object]:
    reference_path = reference_path.resolve(strict=True)
    observed_path = observed_path.resolve(strict=True)
    return {
        "reference": str(reference_path),
        "observed": str(observed_path),
        "reference_sha256": sha256_file(reference_path),
        "observed_sha256": sha256_file(observed_path),
        "byte_equal": sha256_file(reference_path) == sha256_file(observed_path),
        **compare_rows(
            read_band_file(reference_path), read_band_file(observed_path)
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("reference_exx", type=Path)
    parser.add_argument("observed_exx", type=Path)
    parser.add_argument("reference_qsgw", type=Path)
    parser.add_argument("observed_qsgw", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    try:
        report = {
            "schema": SCHEMA,
            "exx": compare_files(args.reference_exx, args.observed_exx),
            "qsgw": compare_files(args.reference_qsgw, args.observed_qsgw),
        }
        report["analyzed"] = True
    except (AnalysisError, OSError, ValueError) as error:
        report = {"schema": SCHEMA, "analyzed": False, "error": str(error)}
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))
    return 0 if report.get("analyzed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
