#!/usr/bin/env python3
"""Append validated fractional symmetry operations to a legacy stru_out."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Sequence


Matrix3 = tuple[tuple[float, float, float], ...]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_int(token: str, context: str) -> int:
    try:
        value = int(token)
    except ValueError as error:
        raise ValueError(f"{context}: expected integer, found {token!r}") from error
    return value


def parse_float(token: str, context: str) -> float:
    try:
        value = float(token)
    except ValueError as error:
        raise ValueError(f"{context}: expected number, found {token!r}") from error
    if not math.isfinite(value):
        raise ValueError(f"{context}: non-finite value {token!r}")
    return value


def determinant(matrix: Matrix3) -> float:
    a, b, c = matrix
    return (
        a[0] * (b[1] * c[2] - b[2] * c[1])
        - a[1] * (b[0] * c[2] - b[2] * c[0])
        + a[2] * (b[0] * c[1] - b[1] * c[0])
    )


def inverse(matrix: Matrix3) -> Matrix3:
    a, b, c = matrix
    det = determinant(matrix)
    if abs(det) <= 1.0e-15:
        raise ValueError("source stru_out: singular lattice matrix")
    return (
        (
            (b[1] * c[2] - b[2] * c[1]) / det,
            (a[2] * c[1] - a[1] * c[2]) / det,
            (a[1] * b[2] - a[2] * b[1]) / det,
        ),
        (
            (b[2] * c[0] - b[0] * c[2]) / det,
            (a[0] * c[2] - a[2] * c[0]) / det,
            (a[2] * b[0] - a[0] * b[2]) / det,
        ),
        (
            (b[0] * c[1] - b[1] * c[0]) / det,
            (a[1] * c[0] - a[0] * c[1]) / det,
            (a[0] * b[1] - a[1] * b[0]) / det,
        ),
    )


def transpose(matrix: Matrix3) -> Matrix3:
    return tuple(tuple(matrix[j][i] for j in range(3)) for i in range(3))


def multiply(left: Matrix3, right: Matrix3) -> Matrix3:
    return tuple(
        tuple(sum(left[i][k] * right[k][j] for k in range(3)) for j in range(3))
        for i in range(3)
    )


def row_times_matrix(vector: Sequence[float], matrix: Matrix3) -> tuple[float, ...]:
    return tuple(sum(vector[k] * matrix[k][j] for k in range(3)) for j in range(3))


def parse_source(
    text: str, expected_grid: tuple[int, int, int], n_scf_kpoints: int
) -> dict[str, object]:
    tokens = text.split()
    minimum = 18 + 1
    if len(tokens) < minimum:
        raise ValueError("source stru_out: truncated lattice section")
    pos = 0
    lattice_values = [parse_float(tokens[pos + i], "source lattice") for i in range(9)]
    pos += 9
    reciprocal_values = [
        parse_float(tokens[pos + i], "source reciprocal lattice") for i in range(9)
    ]
    pos += 9
    n_atoms = parse_int(tokens[pos], "source atom count")
    pos += 1
    if n_atoms <= 0:
        raise ValueError("source stru_out: atom count must be positive")
    atoms: list[tuple[tuple[float, float, float], int]] = []
    for atom in range(n_atoms):
        if pos + 4 > len(tokens):
            raise ValueError("source stru_out: truncated atom section")
        coordinate = tuple(
            parse_float(tokens[pos + i], f"source atom {atom + 1}") for i in range(3)
        )
        atom_type = parse_int(tokens[pos + 3], f"source atom {atom + 1} type")
        atoms.append((coordinate, atom_type))
        pos += 4

    legacy = tokens[pos:]
    if any(token.lower() in {"row", "col"} for token in legacy):
        raise ValueError("source stru_out already contains a symmetry block")
    if len(legacy) < 3:
        raise ValueError("source stru_out: missing legacy k-point grid")
    grid = tuple(parse_int(legacy[i], "legacy k-point grid") for i in range(3))
    if grid != expected_grid:
        raise ValueError(f"source stru_out grid {grid}, expected {expected_grid}")
    if any(value <= 0 for value in grid):
        raise ValueError("source stru_out: legacy k-point grid must be positive")
    n_full = grid[0] * grid[1] * grid[2]
    payload = legacy[3:]
    allowed_counts = {
        3 * n_scf_kpoints,
        3 * n_scf_kpoints + n_full,
        3 * n_full,
        4 * n_full,
    }
    if len(payload) not in allowed_counts:
        raise ValueError(
            "source stru_out: legacy k-point payload has "
            f"{len(payload)} tokens, expected one of {sorted(allowed_counts)}"
        )
    coordinate_count = 3 * (n_scf_kpoints if len(payload) in {
        3 * n_scf_kpoints,
        3 * n_scf_kpoints + n_full,
    } else n_full)
    for index, token in enumerate(payload[:coordinate_count]):
        parse_float(token, f"legacy k-point coordinate {index + 1}")
    for index, token in enumerate(payload[coordinate_count:]):
        parse_int(token, f"legacy k-point mapping {index + 1}")

    lattice = tuple(
        tuple(lattice_values[3 * row + column] for column in range(3))
        for row in range(3)
    )
    reciprocal = tuple(
        tuple(reciprocal_values[3 * row + column] for column in range(3))
        for row in range(3)
    )
    return {
        "lattice": lattice,
        "reciprocal": reciprocal,
        "atoms": atoms,
        "grid": grid,
        "n_full_kpoints": n_full,
        "legacy_payload_tokens": len(payload),
    }


def parse_tail(text: str) -> list[tuple[Matrix3, tuple[float, float, float]]]:
    tokens = text.split()
    if len(tokens) < 2:
        raise ValueError("symmetry tail: missing header")
    n_symops = parse_int(tokens[0], "symmetry operation count")
    convention = tokens[1].lower()
    if n_symops <= 1:
        raise ValueError("symmetry tail: expected more than identity")
    if convention != "row":
        raise ValueError(f"symmetry tail: convention {convention!r}, expected 'row'")
    if len(tokens) != 2 + 12 * n_symops:
        raise ValueError("symmetry tail: token count does not match its header")
    operations: list[tuple[Matrix3, tuple[float, float, float]]] = []
    pos = 2
    for operation in range(n_symops):
        rotation_values = [
            parse_int(tokens[pos + i], f"symmetry operation {operation + 1} rotation")
            for i in range(9)
        ]
        rotation = tuple(
            tuple(float(rotation_values[3 * row + column]) for column in range(3))
            for row in range(3)
        )
        det = round(determinant(rotation))
        if det not in {-1, 1}:
            raise ValueError(
                f"symmetry operation {operation + 1}: determinant {det}, expected +/-1"
            )
        translation = tuple(
            parse_float(tokens[pos + 9 + i], f"symmetry operation {operation + 1} translation")
            for i in range(3)
        )
        operations.append((rotation, translation))
        pos += 12
    return operations


def modular_max_abs(left: Sequence[float], right: Sequence[float]) -> float:
    return max(abs(delta - round(delta)) for delta in (a - b for a, b in zip(left, right)))


def validate_operations(
    source: dict[str, object],
    operations: Sequence[tuple[Matrix3, tuple[float, float, float]]],
    metric_tolerance: float,
    atom_tolerance: float,
) -> dict[str, object]:
    lattice = source["lattice"]
    atoms = source["atoms"]
    assert isinstance(lattice, tuple)
    assert isinstance(atoms, list)
    metric = multiply(lattice, transpose(lattice))
    metric_max_abs = 0.0
    for rotation, _ in operations:
        transformed = multiply(multiply(rotation, metric), transpose(rotation))
        metric_max_abs = max(
            metric_max_abs,
            max(abs(transformed[i][j] - metric[i][j]) for i in range(3) for j in range(3)),
        )
    if metric_max_abs > metric_tolerance:
        raise ValueError(
            "symmetry tail breaks the source lattice metric: "
            f"max_abs={metric_max_abs:.17g}, tolerance={metric_tolerance:.17g}"
        )

    inverse_lattice = inverse(lattice)
    fractional_atoms = [
        (row_times_matrix(coordinate, inverse_lattice), atom_type)
        for coordinate, atom_type in atoms
    ]
    atom_fractional_max_abs = 0.0
    for operation_index, (rotation, translation) in enumerate(operations, start=1):
        for coordinate, atom_type in fractional_atoms:
            rotated = row_times_matrix(coordinate, rotation)
            transformed = tuple(rotated[i] + translation[i] for i in range(3))
            candidates = [
                modular_max_abs(transformed, target)
                for target, target_type in fractional_atoms
                if target_type == atom_type
            ]
            if not candidates:
                raise ValueError(f"atom type {atom_type} has no symmetry mapping candidate")
            best = min(candidates)
            atom_fractional_max_abs = max(atom_fractional_max_abs, best)
            if best > atom_tolerance:
                raise ValueError(
                    f"symmetry operation {operation_index} breaks atom mapping: "
                    f"max_abs={best:.17g}, tolerance={atom_tolerance:.17g}"
                )

    identity_count = 0
    identity = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    for rotation, translation in operations:
        if rotation == identity and modular_max_abs(translation, (0.0, 0.0, 0.0)) <= 1.0e-12:
            identity_count += 1
    if identity_count != 1:
        raise ValueError(f"symmetry tail has {identity_count} identity operations, expected one")
    return {
        "metric_max_abs": metric_max_abs,
        "atom_fractional_max_abs": atom_fractional_max_abs,
        "identity_count": identity_count,
    }


def build_overlay(
    *,
    source_path: Path,
    tail_path: Path,
    output_path: Path,
    report_path: Path,
    expected_grid: tuple[int, int, int],
    n_scf_kpoints: int,
    metric_tolerance: float,
    atom_tolerance: float,
) -> dict[str, object]:
    source_bytes = source_path.read_bytes()
    tail_bytes = tail_path.read_bytes()
    source_text = source_bytes.decode("utf-8")
    tail_text = tail_bytes.decode("utf-8")
    source = parse_source(source_text, expected_grid, n_scf_kpoints)
    operations = parse_tail(tail_text)
    validation = validate_operations(
        source, operations, metric_tolerance, atom_tolerance
    )
    separator = b"" if source_bytes.endswith(b"\n") else b"\n"
    terminator = b"" if tail_bytes.endswith(b"\n") else b"\n"
    output_path.write_bytes(source_bytes + separator + tail_bytes + terminator)
    report: dict[str, object] = {
        "status": "PASS",
        "source_path": str(source_path),
        "source_sha256": sha256(source_path),
        "tail_path": str(tail_path),
        "tail_sha256": sha256(tail_path),
        "output_path": str(output_path),
        "output_sha256": sha256(output_path),
        "source_prefix_byte_identical": output_path.read_bytes().startswith(source_bytes),
        "grid": list(expected_grid),
        "n_scf_kpoints": n_scf_kpoints,
        "n_full_kpoints": source["n_full_kpoints"],
        "legacy_payload_tokens": source["legacy_payload_tokens"],
        "n_symops": len(operations),
        "convention": "row",
        "metric_tolerance": metric_tolerance,
        "atom_tolerance": atom_tolerance,
        **validation,
    }
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source_path", type=Path)
    parser.add_argument("tail_path", type=Path)
    parser.add_argument("output_path", type=Path)
    parser.add_argument("report_path", type=Path)
    parser.add_argument("--expected-grid", type=int, nargs=3, required=True)
    parser.add_argument("--n-scf-kpoints", type=int, required=True)
    parser.add_argument("--metric-tolerance", type=float, default=1.0e-10)
    parser.add_argument("--atom-tolerance", type=float, default=1.0e-5)
    args = parser.parse_args()
    if args.n_scf_kpoints <= 0:
        parser.error("--n-scf-kpoints must be positive")
    report = build_overlay(
        source_path=args.source_path.resolve(),
        tail_path=args.tail_path.resolve(),
        output_path=args.output_path.resolve(),
        report_path=args.report_path.resolve(),
        expected_grid=tuple(args.expected_grid),
        n_scf_kpoints=args.n_scf_kpoints,
        metric_tolerance=args.metric_tolerance,
        atom_tolerance=args.atom_tolerance,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
