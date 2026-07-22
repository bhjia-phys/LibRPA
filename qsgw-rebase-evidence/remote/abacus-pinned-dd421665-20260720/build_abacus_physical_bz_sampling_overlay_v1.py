#!/usr/bin/env python3
"""Rebuild BZ Cartesian k-vectors from physical reciprocal lattice rows."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Sequence


Matrix3 = tuple[tuple[float, float, float], ...]
Vector3 = tuple[float, float, float]


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _float(token: str, context: str) -> float:
    try:
        value = float(token)
    except ValueError as error:
        raise ValueError(f"{context}: expected number, found {token!r}") from error
    if not math.isfinite(value):
        raise ValueError(f"{context}: non-finite number {token!r}")
    return value


def _int(token: str, context: str) -> int:
    try:
        return int(token)
    except ValueError as error:
        raise ValueError(f"{context}: expected integer, found {token!r}") from error


def _transpose(matrix: Matrix3) -> Matrix3:
    return tuple(tuple(matrix[j][i] for j in range(3)) for i in range(3))


def _multiply(left: Matrix3, right: Matrix3) -> Matrix3:
    return tuple(
        tuple(sum(left[i][k] * right[k][j] for k in range(3)) for j in range(3))
        for i in range(3)
    )


def _row_times_matrix(vector: Sequence[float], matrix: Matrix3) -> Vector3:
    return tuple(
        sum(vector[k] * matrix[k][j] for k in range(3)) for j in range(3)
    )


def _matrix_times_column(matrix: Matrix3, vector: Sequence[float]) -> Vector3:
    return tuple(sum(matrix[i][j] * vector[j] for j in range(3)) for i in range(3))


def _parse_structure(text: str, closure_tolerance: float) -> tuple[Matrix3, Matrix3, float]:
    lines = text.splitlines()
    if len(lines) < 6:
        raise ValueError("physical stru_out: truncated lattice section")
    matrices: list[Matrix3] = []
    for block, context in ((0, "direct lattice"), (3, "reciprocal lattice")):
        rows: list[Vector3] = []
        for offset in range(3):
            tokens = lines[block + offset].split()
            if len(tokens) != 3:
                raise ValueError(
                    f"physical stru_out: {context} row {offset + 1} must have three fields"
                )
            rows.append(
                tuple(_float(token, f"physical stru_out {context}") for token in tokens)
            )
        matrices.append(tuple(rows))
    lattice, reciprocal = matrices
    closure = _multiply(lattice, _transpose(reciprocal))
    closure_max_abs = max(
        abs(closure[i][j] / (2.0 * math.pi) - (1.0 if i == j else 0.0))
        for i in range(3)
        for j in range(3)
    )
    if closure_max_abs > closure_tolerance:
        raise ValueError(
            "physical stru_out reciprocal lattice closure failed: "
            f"max_abs={closure_max_abs:.17g}, tolerance={closure_tolerance:.17g}"
        )
    return lattice, reciprocal, closure_max_abs


def _parse_bz(text: str) -> dict[str, object]:
    lines = text.splitlines()
    if len(lines) < 3 or any(not line.strip() for line in lines):
        raise ValueError("source bz_sampling_out: expected contiguous non-empty rows")
    grid_tokens = lines[0].split()
    if len(grid_tokens) != 3:
        raise ValueError("source bz_sampling_out: grid header must have three fields")
    grid = tuple(_int(token, "BZ grid") for token in grid_tokens)
    if min(grid) <= 0:
        raise ValueError("source bz_sampling_out: grid values must be positive")
    count_tokens = lines[1].split()
    if len(count_tokens) != 2:
        raise ValueError("source bz_sampling_out: count header must have two fields")
    n_scf, n_ibz = (_int(token, "BZ k-point count") for token in count_tokens)
    full_count = math.prod(grid)
    if n_scf <= 0 or n_scf > full_count or n_ibz <= 0 or n_ibz > n_scf:
        raise ValueError("source bz_sampling_out: invalid SCF/IBZ counts")
    if len(lines[2:]) != n_scf:
        raise ValueError("source bz_sampling_out: row count does not match SCF count")

    rows: list[list[str]] = []
    weight_sum = 0.0
    for row_index, line in enumerate(lines[2:], start=1):
        fields = line.split()
        if len(fields) != 10:
            raise ValueError(f"source bz_sampling_out row {row_index}: expected 10 fields")
        if _int(fields[0], f"BZ row {row_index} index") != row_index:
            raise ValueError(f"source bz_sampling_out row {row_index}: index mismatch")
        weight = _float(fields[1], f"BZ row {row_index} weight")
        if weight < 0.0:
            raise ValueError(f"source bz_sampling_out row {row_index}: negative weight")
        weight_sum += weight
        fractional = tuple(
            _float(fields[2 + axis], f"BZ row {row_index} fractional k-point")
            for axis in range(3)
        )
        for axis, value in enumerate(fractional):
            grid_coordinate = value * grid[axis]
            if abs(grid_coordinate - round(grid_coordinate)) > 1.0e-8:
                raise ValueError(
                    f"BZ row {row_index} fractional k-point is not on the BvK grid"
                )
        for axis in range(3):
            _float(fields[5 + axis], f"BZ row {row_index} Cartesian k-vector")
        ibz_index = _int(fields[8], f"BZ row {row_index} IBZ index")
        representative = _int(fields[9], f"BZ row {row_index} representative")
        if ibz_index <= 0 or ibz_index > n_ibz:
            raise ValueError(f"source bz_sampling_out row {row_index}: IBZ index out of range")
        if representative <= 0 or representative > n_scf:
            raise ValueError(
                f"source bz_sampling_out row {row_index}: representative out of range"
            )
        rows.append(fields)
    if abs(weight_sum - 1.0) > 1.0e-10:
        raise ValueError(
            f"source bz_sampling_out: weights do not sum to one ({weight_sum:.17g})"
        )
    return {
        "lines": lines,
        "grid": grid,
        "n_scf": n_scf,
        "n_ibz": n_ibz,
        "rows": rows,
        "weight_sum": weight_sum,
    }


def build_text(
    physical_stru_text: str,
    source_bz_text: str,
    *,
    closure_tolerance: float = 1.0e-12,
    fractional_tolerance: float = 1.0e-12,
) -> tuple[str, dict[str, object]]:
    lattice, reciprocal, closure_max_abs = _parse_structure(
        physical_stru_text, closure_tolerance
    )
    source = _parse_bz(source_bz_text)
    rows = source["rows"]
    assert isinstance(rows, list)
    output_rows: list[str] = []
    cartesian_kvector_max_abs_change = 0.0
    fractional_recovery_max_abs = 0.0
    non_cartesian_tokens_unchanged = True
    for row_index, fields in enumerate(rows, start=1):
        fractional = tuple(float(fields[2 + axis]) for axis in range(3))
        source_cartesian = tuple(float(fields[5 + axis]) for axis in range(3))
        physical_cartesian = _row_times_matrix(fractional, reciprocal)
        recovered = _matrix_times_column(
            lattice, tuple(value / (2.0 * math.pi) for value in physical_cartesian)
        )
        recovery_residual = max(
            abs(left - right) for left, right in zip(recovered, fractional)
        )
        fractional_recovery_max_abs = max(
            fractional_recovery_max_abs, recovery_residual
        )
        if recovery_residual > fractional_tolerance:
            raise ValueError(
                f"BZ row {row_index}: physical Cartesian vector does not recover "
                f"fractional k-point (max_abs={recovery_residual:.17g})"
            )
        cartesian_kvector_max_abs_change = max(
            cartesian_kvector_max_abs_change,
            max(
                abs(left - right)
                for left, right in zip(physical_cartesian, source_cartesian)
            ),
        )
        output_fields = list(fields)
        output_fields[5:8] = [f"{value:.17g}" for value in physical_cartesian]
        non_cartesian_tokens_unchanged = non_cartesian_tokens_unchanged and (
            output_fields[:5] == fields[:5] and output_fields[8:] == fields[8:]
        )
        output_rows.append(" ".join(output_fields))
    if not non_cartesian_tokens_unchanged:
        raise ValueError("BZ overlay changed non-Cartesian tokens")

    source_lines = source["lines"]
    assert isinstance(source_lines, list)
    output_text = "\n".join(source_lines[:2] + output_rows) + "\n"
    report: dict[str, object] = {
        "schema": "abacus-physical-bz-sampling-overlay-v1",
        "passed": True,
        "grid": list(source["grid"]),
        "scf_kpoint_count": source["n_scf"],
        "ibz_kpoint_count": source["n_ibz"],
        "weight_sum": source["weight_sum"],
        "structure_reciprocal_closure_max_abs": closure_max_abs,
        "fractional_recovery_tolerance": fractional_tolerance,
        "fractional_recovery_max_abs": fractional_recovery_max_abs,
        "cartesian_kvector_max_abs_change": cartesian_kvector_max_abs_change,
        "non_cartesian_tokens_unchanged": non_cartesian_tokens_unchanged,
    }
    return output_text, report


def build_overlay(
    physical_stru: Path,
    source_bz: Path,
    output_bz: Path,
    report_path: Path,
) -> dict[str, object]:
    structure_bytes = physical_stru.read_bytes()
    source_bytes = source_bz.read_bytes()
    output_text, report = build_text(
        structure_bytes.decode("utf-8"), source_bytes.decode("utf-8")
    )
    output_bz.write_text(output_text, encoding="utf-8", newline="\n")
    report.update(
        {
            "physical_stru_out_sha256": _sha256_bytes(structure_bytes),
            "source_bz_sampling_out_sha256": _sha256_bytes(source_bytes),
            "output_bz_sampling_out_sha256": _sha256_bytes(output_bz.read_bytes()),
        }
    )
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("physical_stru_out", type=Path)
    parser.add_argument("source_bz_sampling_out", type=Path)
    parser.add_argument("output_bz_sampling_out", type=Path)
    parser.add_argument("report", type=Path)
    args = parser.parse_args()
    report = build_overlay(
        args.physical_stru_out,
        args.source_bz_sampling_out,
        args.output_bz_sampling_out,
        args.report,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
