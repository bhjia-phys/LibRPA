#!/usr/bin/env python3
"""Rebuild LibRPA stru_out lattice rows from the matching ABACUS STRU."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Sequence


Matrix3 = tuple[tuple[float, float, float], ...]
Vector3 = tuple[float, float, float]

SECTION_NAMES = {
    "ATOMIC_SPECIES",
    "NUMERICAL_ORBITAL",
    "NUMERICAL_DESCRIPTOR",
    "LATTICE_CONSTANT",
    "LATTICE_VECTORS",
    "ATOMIC_POSITIONS",
}


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _clean(line: str) -> str:
    return line.split("#", 1)[0].strip()


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


def _determinant(matrix: Matrix3) -> float:
    a, b, c = matrix
    return (
        a[0] * (b[1] * c[2] - b[2] * c[1])
        - a[1] * (b[0] * c[2] - b[2] * c[0])
        + a[2] * (b[0] * c[1] - b[1] * c[0])
    )


def _inverse(matrix: Matrix3) -> Matrix3:
    a, b, c = matrix
    determinant = _determinant(matrix)
    if abs(determinant) <= 1.0e-15:
        raise ValueError("physical lattice is singular")
    return (
        (
            (b[1] * c[2] - b[2] * c[1]) / determinant,
            (a[2] * c[1] - a[1] * c[2]) / determinant,
            (a[1] * b[2] - a[2] * b[1]) / determinant,
        ),
        (
            (b[2] * c[0] - b[0] * c[2]) / determinant,
            (a[0] * c[2] - a[2] * c[0]) / determinant,
            (a[2] * b[0] - a[0] * b[2]) / determinant,
        ),
        (
            (b[0] * c[1] - b[1] * c[0]) / determinant,
            (a[1] * c[0] - a[0] * c[1]) / determinant,
            (a[0] * b[1] - a[1] * b[0]) / determinant,
        ),
    )


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


def _next_data_line(lines: Sequence[str], start: int, context: str) -> tuple[int, str]:
    for index in range(start, len(lines)):
        value = _clean(lines[index])
        if value:
            return index, value
    raise ValueError(f"input STRU: missing {context}")


def _find_section(lines: Sequence[str], section: str) -> int:
    matches = [index for index, line in enumerate(lines) if _clean(line).upper() == section]
    if len(matches) != 1:
        raise ValueError(
            f"input STRU: expected one {section} section, found {len(matches)}"
        )
    return matches[0]


def _parse_matrix_rows(lines: Sequence[str], start: int, context: str) -> Matrix3:
    rows: list[Vector3] = []
    position = start
    for row in range(3):
        index, value = _next_data_line(lines, position, f"{context} row {row + 1}")
        tokens = value.split()
        if len(tokens) < 3:
            raise ValueError(f"input STRU: truncated {context} row {row + 1}")
        rows.append(tuple(_float(tokens[i], context) for i in range(3)))
        position = index + 1
    return tuple(rows)


def _parse_input_stru(text: str) -> dict[str, object]:
    lines = text.splitlines()
    constant_section = _find_section(lines, "LATTICE_CONSTANT")
    _, constant_line = _next_data_line(
        lines, constant_section + 1, "LATTICE_CONSTANT value"
    )
    lattice_constant = _float(constant_line.split()[0], "LATTICE_CONSTANT")
    if lattice_constant <= 0.0:
        raise ValueError("input STRU: LATTICE_CONSTANT must be positive")

    lattice_section = _find_section(lines, "LATTICE_VECTORS")
    dimensionless_lattice = _parse_matrix_rows(
        lines, lattice_section + 1, "LATTICE_VECTORS"
    )
    lattice: Matrix3 = tuple(
        tuple(lattice_constant * value for value in row)
        for row in dimensionless_lattice
    )
    _inverse(lattice)

    species_section = _find_section(lines, "ATOMIC_SPECIES")
    species: list[str] = []
    for line in lines[species_section + 1 :]:
        value = _clean(line)
        if not value:
            continue
        if value.upper() in SECTION_NAMES:
            break
        species.append(value.split()[0])
    if not species:
        raise ValueError("input STRU: ATOMIC_SPECIES has no entries")

    positions_section = _find_section(lines, "ATOMIC_POSITIONS")
    position, coordinate_type = _next_data_line(
        lines, positions_section + 1, "ATOMIC_POSITIONS coordinate type"
    )
    if coordinate_type.lower() != "direct":
        raise ValueError(
            "input STRU: physical overlay currently requires Direct coordinates"
        )
    position += 1
    atoms: list[tuple[Vector3, int]] = []
    for species_index, expected_species in enumerate(species, start=1):
        label_index, label = _next_data_line(
            lines, position, f"ATOMIC_POSITIONS species {species_index}"
        )
        if label.split()[0] != expected_species:
            raise ValueError(
                "input STRU: ATOMIC_POSITIONS species order does not match "
                "ATOMIC_SPECIES"
            )
        moment_index, _ = _next_data_line(
            lines, label_index + 1, f"{expected_species} magnetization"
        )
        count_index, count_line = _next_data_line(
            lines, moment_index + 1, f"{expected_species} atom count"
        )
        atom_count = _int(count_line.split()[0], f"{expected_species} atom count")
        if atom_count < 0:
            raise ValueError(f"input STRU: {expected_species} atom count is negative")
        position = count_index + 1
        for atom_index in range(atom_count):
            row_index, row = _next_data_line(
                lines, position, f"{expected_species} atom {atom_index + 1}"
            )
            tokens = row.split()
            if len(tokens) < 3:
                raise ValueError(
                    f"input STRU: truncated {expected_species} atom {atom_index + 1}"
                )
            direct = tuple(
                _float(tokens[i], f"{expected_species} atom {atom_index + 1}")
                for i in range(3)
            )
            atoms.append((direct, species_index))
            position = row_index + 1
    if not atoms:
        raise ValueError("input STRU: no atoms")
    return {
        "lattice_constant_bohr": lattice_constant,
        "dimensionless_lattice": dimensionless_lattice,
        "lattice": lattice,
        "atoms": atoms,
    }


def _parse_source(text: str) -> dict[str, object]:
    lines = text.splitlines(keepends=True)
    if len(lines) < 8:
        raise ValueError("source stru_out: truncated")
    for index in range(6):
        tokens = _clean(lines[index]).split()
        if len(tokens) != 3:
            raise ValueError(
                f"source stru_out: lattice row {index + 1} must contain three numbers"
            )
        for token in tokens:
            _float(token, f"source stru_out lattice row {index + 1}")

    position = 6
    while position < len(lines) and not _clean(lines[position]):
        position += 1
    if position >= len(lines):
        raise ValueError("source stru_out: missing atom count")
    atom_count_tokens = _clean(lines[position]).split()
    if len(atom_count_tokens) != 1:
        raise ValueError("source stru_out: malformed atom count")
    atom_count = _int(atom_count_tokens[0], "source stru_out atom count")
    if atom_count <= 0:
        raise ValueError("source stru_out: atom count must be positive")
    position += 1
    atoms: list[tuple[Vector3, int]] = []
    for atom_index in range(atom_count):
        while position < len(lines) and not _clean(lines[position]):
            position += 1
        if position >= len(lines):
            raise ValueError("source stru_out: truncated atom rows")
        tokens = _clean(lines[position]).split()
        if len(tokens) != 4:
            raise ValueError(
                f"source stru_out: atom {atom_index + 1} must contain four fields"
            )
        coordinate = tuple(
            _float(tokens[i], f"source stru_out atom {atom_index + 1}")
            for i in range(3)
        )
        atom_type = _int(tokens[3], f"source stru_out atom {atom_index + 1} type")
        atoms.append((coordinate, atom_type))
        position += 1

    while position < len(lines) and not _clean(lines[position]):
        position += 1
    if position >= len(lines):
        raise ValueError("source stru_out: missing symmetry block")
    header = _clean(lines[position]).split()
    if len(header) != 2:
        raise ValueError("source stru_out: malformed symmetry header")
    symmetry_count = _int(header[0], "source stru_out symmetry operation count")
    if symmetry_count <= 0 or header[1].lower() != "row":
        raise ValueError("source stru_out: expected a positive row-convention symmetry block")
    symmetry_tokens = " ".join(_clean(line) for line in lines[position + 1 :]).split()
    if len(symmetry_tokens) != 12 * symmetry_count:
        raise ValueError("source stru_out: symmetry block token count mismatch")
    operations: list[tuple[Matrix3, Vector3]] = []
    for operation_index in range(symmetry_count):
        start = 12 * operation_index
        rotation_values = [
            _int(
                symmetry_tokens[start + offset],
                f"source stru_out symmetry operation {operation_index + 1}",
            )
            for offset in range(9)
        ]
        rotation: Matrix3 = tuple(
            tuple(float(rotation_values[3 * row + column]) for column in range(3))
            for row in range(3)
        )
        determinant = round(_determinant(rotation))
        if determinant not in {-1, 1}:
            raise ValueError(
                f"source stru_out symmetry operation {operation_index + 1}: "
                "determinant is not +/-1"
            )
        translation = tuple(
            _float(
                symmetry_tokens[start + 9 + offset],
                f"source stru_out symmetry operation {operation_index + 1}",
            )
            for offset in range(3)
        )
        operations.append((rotation, translation))
    return {
        "lines": lines,
        "atoms": atoms,
        "operations": operations,
        "symmetry_operation_count": symmetry_count,
    }


def _modular_max_abs(left: Sequence[float], right: Sequence[float]) -> float:
    return max(
        abs(delta - round(delta)) for delta in (a - b for a, b in zip(left, right))
    )


def _format_matrix(matrix: Matrix3) -> str:
    return "".join(" ".join(f"{value:.17g}" for value in row) + "\n" for row in matrix)


def build_text(
    input_stru_text: str,
    source_text: str,
    *,
    atom_tolerance: float = 1.0e-10,
    reciprocal_tolerance: float = 1.0e-12,
    symmetry_tolerance: float = 1.0e-10,
) -> tuple[str, dict[str, object]]:
    input_data = _parse_input_stru(input_stru_text)
    source_data = _parse_source(source_text)
    lattice = input_data["lattice"]
    input_atoms = input_data["atoms"]
    source_atoms = source_data["atoms"]
    operations = source_data["operations"]
    assert isinstance(lattice, tuple)
    assert isinstance(input_atoms, list)
    assert isinstance(source_atoms, list)
    assert isinstance(operations, list)

    if len(input_atoms) != len(source_atoms):
        raise ValueError("source stru_out and input STRU atom counts differ")
    atom_cartesian_max_abs = 0.0
    for atom_index, ((direct, expected_type), (cartesian, source_type)) in enumerate(
        zip(input_atoms, source_atoms), start=1
    ):
        if expected_type != source_type:
            raise ValueError(
                f"source stru_out atom {atom_index}: atom type differs from input STRU"
            )
        expected_cartesian = _row_times_matrix(direct, lattice)
        residual = max(
            abs(left - right) for left, right in zip(expected_cartesian, cartesian)
        )
        atom_cartesian_max_abs = max(atom_cartesian_max_abs, residual)
        if residual > atom_tolerance:
            raise ValueError(
                "source stru_out Cartesian atom coordinates disagree with input STRU: "
                f"atom={atom_index}, max_abs={residual:.17g}, "
                f"tolerance={atom_tolerance:.17g}"
            )

    inverse_lattice = _inverse(lattice)
    reciprocal = tuple(
        tuple(2.0 * math.pi * value for value in row)
        for row in _transpose(inverse_lattice)
    )
    closure = _multiply(lattice, _transpose(reciprocal))
    reciprocal_closure_max_abs = max(
        abs(closure[i][j] / (2.0 * math.pi) - (1.0 if i == j else 0.0))
        for i in range(3)
        for j in range(3)
    )
    if reciprocal_closure_max_abs > reciprocal_tolerance:
        raise ValueError(
            "physical reciprocal lattice closure failed: "
            f"max_abs={reciprocal_closure_max_abs:.17g}"
        )

    metric = _multiply(lattice, _transpose(lattice))
    metric_max_abs = 0.0
    fractional_atoms = [(direct, atom_type) for direct, atom_type in input_atoms]
    atom_symmetry_max_abs = 0.0
    for operation_index, (rotation, translation) in enumerate(operations, start=1):
        transformed_metric = _multiply(_multiply(rotation, metric), _transpose(rotation))
        metric_max_abs = max(
            metric_max_abs,
            max(
                abs(transformed_metric[i][j] - metric[i][j])
                for i in range(3)
                for j in range(3)
            ),
        )
        for coordinate, atom_type in fractional_atoms:
            rotated = _row_times_matrix(coordinate, rotation)
            transformed = tuple(rotated[i] + translation[i] for i in range(3))
            candidates = [
                _modular_max_abs(transformed, target)
                for target, target_type in fractional_atoms
                if target_type == atom_type
            ]
            if not candidates:
                raise ValueError(f"atom type {atom_type} has no atom mapping candidate")
            residual = min(candidates)
            atom_symmetry_max_abs = max(atom_symmetry_max_abs, residual)
            if residual > symmetry_tolerance:
                raise ValueError(
                    f"symmetry operation {operation_index} breaks atom mapping: "
                    f"max_abs={residual:.17g}, tolerance={symmetry_tolerance:.17g}"
                )
    if metric_max_abs > symmetry_tolerance:
        raise ValueError(
            "symmetry operations break the physical lattice metric: "
            f"max_abs={metric_max_abs:.17g}, tolerance={symmetry_tolerance:.17g}"
        )

    source_lines = source_data["lines"]
    assert isinstance(source_lines, list)
    output_text = _format_matrix(lattice) + _format_matrix(reciprocal) + "".join(
        source_lines[6:]
    )
    report: dict[str, object] = {
        "schema": "abacus-physical-stru-overlay-v1",
        "passed": True,
        "lattice_constant_bohr": input_data["lattice_constant_bohr"],
        "atom_count": len(input_atoms),
        "symmetry_operation_count": source_data["symmetry_operation_count"],
        "symmetry_convention": "row",
        "atom_cartesian_tolerance_bohr": atom_tolerance,
        "atom_cartesian_max_abs_bohr": atom_cartesian_max_abs,
        "reciprocal_closure_tolerance": reciprocal_tolerance,
        "reciprocal_closure_max_abs": reciprocal_closure_max_abs,
        "symmetry_tolerance": symmetry_tolerance,
        "lattice_metric_max_abs": metric_max_abs,
        "atom_symmetry_max_abs": atom_symmetry_max_abs,
        "preserved_source_payload_sha256": _sha256_bytes(
            "".join(source_lines[6:]).encode("utf-8")
        ),
    }
    return output_text, report


def build_overlay(
    input_stru: Path,
    source: Path,
    output: Path,
    report: Path,
) -> dict[str, object]:
    input_bytes = input_stru.read_bytes()
    source_bytes = source.read_bytes()
    output_text, result = build_text(
        input_bytes.decode("utf-8"), source_bytes.decode("utf-8")
    )
    output.write_text(output_text, encoding="utf-8", newline="\n")
    result.update(
        {
            "input_stru_sha256": _sha256_bytes(input_bytes),
            "source_stru_out_sha256": _sha256_bytes(source_bytes),
            "output_stru_out_sha256": _sha256_bytes(output.read_bytes()),
        }
    )
    report.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_stru", type=Path)
    parser.add_argument("source_stru_out", type=Path)
    parser.add_argument("output_stru_out", type=Path)
    parser.add_argument("report", type=Path)
    args = parser.parse_args()
    result = build_overlay(
        args.input_stru,
        args.source_stru_out,
        args.output_stru_out,
        args.report,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
