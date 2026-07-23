#!/usr/bin/env python3
"""Build a full-BZ legacy input view without changing physical input data."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
from pathlib import Path
from typing import Iterable


class OverlayError(RuntimeError):
    """Raised when the source dataset does not satisfy the overlay contract."""


_MANIFEST_RE = re.compile(r"^([0-9a-f]{64})  (dataset/.+)$")
_NATIVE_VXC_RE = re.compile(r"^vxck([1-9][0-9]*)_nao\.txt$")
_ROWS_RE = re.compile(r"^\s*#\s*rows\s+(\d+)\s*$", re.IGNORECASE)
_COLUMNS_RE = re.compile(r"^\s*#\s*columns\s+(\d+)\s*$", re.IGNORECASE)
_ROW_MARKER_RE = re.compile(r"^\s*Row\s+(\d+)\s*$", re.IGNORECASE)
_NUMBER = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eEdD][+-]?\d+)?"
_COMPLEX_RE = re.compile(rf"\(\s*({_NUMBER})\s*,\s*({_NUMBER})\s*\)")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _parse_int(token: str, context: str) -> int:
    try:
        value = int(token)
    except ValueError as error:
        raise OverlayError(f"invalid integer for {context}: {token}") from error
    return value


def _parse_float(token: str, context: str) -> float:
    try:
        value = float(token)
    except ValueError as error:
        raise OverlayError(f"invalid float for {context}: {token}") from error
    if not math.isfinite(value):
        raise OverlayError(f"non-finite float for {context}: {token}")
    return value


def _read_manifest(manifest_path: Path, source_dataset: Path) -> dict[str, str]:
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise OverlayError(f"source manifest is not a regular file: {manifest_path}")
    entries: dict[str, str] = {}
    for lineno, raw_line in enumerate(
        manifest_path.read_text(encoding="ascii").splitlines(), start=1
    ):
        match = _MANIFEST_RE.fullmatch(raw_line)
        if match is None:
            raise OverlayError(f"invalid manifest line {lineno}: {raw_line}")
        digest, rel = match.groups()
        if rel in entries:
            raise OverlayError(f"duplicate manifest path: {rel}")
        entries[rel] = digest

    discovered: set[str] = set()
    for path in source_dataset.rglob("*"):
        if path.is_symlink():
            raise OverlayError(f"source dataset contains symlink: {path}")
        if path.is_file():
            discovered.add(f"dataset/{path.relative_to(source_dataset).as_posix()}")
    if set(entries) != discovered:
        missing = sorted(discovered - set(entries))
        extra = sorted(set(entries) - discovered)
        raise OverlayError(f"manifest coverage mismatch: missing={missing} extra={extra}")

    for rel, expected in entries.items():
        actual = _sha256(source_dataset.parent / rel)
        if actual != expected:
            raise OverlayError(
                f"manifest hash mismatch for {rel}: expected={expected} actual={actual}"
            )
    return entries


def _validate_plain_structure(path: Path) -> str:
    text = path.read_text(encoding="ascii")
    tokens = text.split()
    if len(tokens) < 19:
        raise OverlayError("stru_out is too short")
    for index, token in enumerate(tokens[:18]):
        _parse_float(token, f"stru_out lattice token {index + 1}")
    n_atoms = _parse_int(tokens[18], "stru_out atom count")
    if n_atoms < 0:
        raise OverlayError("stru_out atom count is negative")
    expected_tokens = 19 + 4 * n_atoms
    if len(tokens) != expected_tokens:
        raise OverlayError(
            "stru_out already has trailing data or malformed atom rows: "
            f"expected_tokens={expected_tokens} actual_tokens={len(tokens)}"
        )
    for atom in range(n_atoms):
        offset = 19 + 4 * atom
        for axis in range(3):
            _parse_float(tokens[offset + axis], f"atom {atom + 1} coordinate")
        atom_type = _parse_int(tokens[offset + 3], f"atom {atom + 1} type")
        if atom_type <= 0:
            raise OverlayError(f"atom {atom + 1} type must be positive")
    return text if text.endswith("\n") else text + "\n"


def _read_band_header(path: Path) -> tuple[int, int]:
    tokens = path.read_text(encoding="ascii").split()
    if len(tokens) < 4:
        raise OverlayError("band_out header is incomplete")
    count = _parse_int(tokens[0], "band_out k-point count")
    if count <= 0:
        raise OverlayError("band_out k-point count must be positive")
    n_aos = _parse_int(tokens[3], "band_out AO count")
    if n_aos <= 0:
        raise OverlayError("band_out AO count must be positive")
    return count, n_aos


def _validate_fractional_grid(rows: Iterable[list[str]], grid: list[int]) -> None:
    seen: set[tuple[int, int, int]] = set()
    for row_number, row in enumerate(rows, start=1):
        indices = []
        for axis, nk in enumerate(grid):
            fractional = _parse_float(row[2 + axis], f"BZ row {row_number} fractional k")
            scaled = fractional * nk
            nearest = round(scaled)
            if abs(scaled - nearest) > 1.0e-10:
                raise OverlayError(
                    f"BZ row {row_number} is off the declared Monkhorst-Pack grid"
                )
            indices.append(nearest % nk)
        key = tuple(indices)
        if key in seen:
            raise OverlayError(f"duplicate full-BZ grid point at row {row_number}: {key}")
        seen.add(key)
    expected = grid[0] * grid[1] * grid[2]
    if len(seen) != expected:
        raise OverlayError(f"full-BZ grid coverage mismatch: {len(seen)} != {expected}")


def _read_fullbz_sampling(path: Path, band_kpoints: int) -> tuple[list[int], list[list[str]]]:
    lines = [line.split() for line in path.read_text(encoding="ascii").splitlines() if line.split()]
    if len(lines) < 2:
        raise OverlayError("bz_sampling_out is too short")
    if len(lines[0]) != 3:
        raise OverlayError("bz_sampling_out grid header must have three integers")
    grid = [_parse_int(token, "BZ grid") for token in lines[0]]
    if any(nk <= 0 for nk in grid):
        raise OverlayError(f"BZ grid entries must be positive: {grid}")
    full_count = grid[0] * grid[1] * grid[2]

    if len(lines[1]) != 2:
        raise OverlayError("bz_sampling_out count header must have two integers")
    irreducible_count = _parse_int(lines[1][0], "IBZ count")
    declared_full_count = _parse_int(lines[1][1], "full-BZ count")
    if irreducible_count != full_count or declared_full_count != full_count:
        raise OverlayError(
            "full-BZ identity mapping is required: "
            f"ibz={irreducible_count} full={declared_full_count} grid={full_count}"
        )
    if band_kpoints != full_count:
        raise OverlayError(
            f"band_out k-point count {band_kpoints} does not match BZ grid {full_count}"
        )

    rows = lines[2:]
    if len(rows) != full_count:
        raise OverlayError(f"BZ row count {len(rows)} does not match full grid {full_count}")
    weight_sum = 0.0
    for expected_index, row in enumerate(rows, start=1):
        if len(row) != 10:
            raise OverlayError(
                f"BZ row {expected_index} must have 10 tokens, found {len(row)}"
            )
        row_index = _parse_int(row[0], f"BZ row {expected_index} index")
        if row_index != expected_index:
            raise OverlayError(
                f"BZ row index mismatch: expected={expected_index} actual={row_index}"
            )
        weight_sum += _parse_float(row[1], f"BZ row {expected_index} weight")
        for token in row[2:8]:
            _parse_float(token, f"BZ row {expected_index} coordinate")
        first_mapping = _parse_int(row[8], f"BZ row {expected_index} first mapping")
        second_mapping = _parse_int(row[9], f"BZ row {expected_index} second mapping")
        if first_mapping != expected_index or second_mapping != expected_index:
            raise OverlayError(
                "full-BZ identity mapping is required: "
                f"row={expected_index} mappings={first_mapping},{second_mapping}"
            )
    if abs(weight_sum - 1.0) > 1.0e-10:
        raise OverlayError(f"BZ weights do not sum to one: {weight_sum:.17g}")
    _validate_fractional_grid(rows, grid)
    return grid, rows


def _write_manifest(path: Path, entries: dict[str, str]) -> None:
    text = "".join(f"{entries[rel]}  {rel}\n" for rel in sorted(entries))
    path.write_text(text, encoding="ascii")


def _normalized_number(token: str, context: str) -> str:
    normalized = token.replace("D", "E").replace("d", "e")
    _parse_float(normalized, context)
    return normalized


def _parse_native_vxc(
    path: Path,
) -> tuple[int, list[list[tuple[str, str]]], list[list[tuple[float, float]]]]:
    n_rows: int | None = None
    n_columns: int | None = None
    current_row: int | None = None
    token_rows: dict[int, list[tuple[str, str]]] = {}
    numeric_rows: dict[int, list[tuple[float, float]]] = {}
    for lineno, line in enumerate(path.read_text(encoding="ascii").splitlines(), start=1):
        rows_match = _ROWS_RE.fullmatch(line)
        if rows_match:
            n_rows = _parse_int(rows_match.group(1), f"{path.name} rows")
            continue
        columns_match = _COLUMNS_RE.fullmatch(line)
        if columns_match:
            n_columns = _parse_int(columns_match.group(1), f"{path.name} columns")
            continue
        marker_match = _ROW_MARKER_RE.fullmatch(line)
        if marker_match:
            current_row = _parse_int(marker_match.group(1), f"{path.name} row marker")
            if current_row in token_rows:
                raise OverlayError(f"duplicate Vxc row marker in {path}: {current_row}")
            token_rows[current_row] = []
            numeric_rows[current_row] = []
            continue
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if current_row is None:
            raise OverlayError(f"Vxc values before a row marker in {path}:{lineno}")
        matches = list(_COMPLEX_RE.finditer(line))
        if not matches or _COMPLEX_RE.sub("", line).strip():
            raise OverlayError(f"malformed Vxc values in {path}:{lineno}")
        for match in matches:
            real_token = _normalized_number(
                match.group(1), f"{path.name} row {current_row} real"
            )
            imag_token = _normalized_number(
                match.group(2), f"{path.name} row {current_row} imag"
            )
            token_rows[current_row].append((real_token, imag_token))
            numeric_rows[current_row].append((float(real_token), float(imag_token)))

    if n_rows is None or n_columns is None or n_rows <= 0 or n_rows != n_columns:
        raise OverlayError(f"native Vxc must declare a positive square matrix: {path}")
    expected_rows = set(range(1, n_rows + 1))
    if set(token_rows) != expected_rows:
        raise OverlayError(
            f"native Vxc row coverage mismatch in {path}: "
            f"expected={sorted(expected_rows)} actual={sorted(token_rows)}"
        )
    ordered_tokens = []
    ordered_numeric = []
    for row in range(1, n_rows + 1):
        expected_length = n_rows - row + 1
        if len(token_rows[row]) != expected_length:
            raise OverlayError(
                f"native Vxc row {row} length mismatch in {path}: "
                f"expected={expected_length} actual={len(token_rows[row])}"
            )
        diagonal = numeric_rows[row][0]
        if abs(diagonal[1]) > 1.0e-12:
            raise OverlayError(
                f"native Vxc diagonal has an imaginary part in {path}, row {row}"
            )
        ordered_tokens.append(token_rows[row])
        ordered_numeric.append(numeric_rows[row])
    return n_rows, ordered_tokens, ordered_numeric


def _render_legacy_vxc(
    dimension: int, rows: list[list[tuple[str, str]]]
) -> str:
    lines = [str(dimension)]
    for row in rows:
        lines.append(" ".join(f"({real},{imag})" for real, imag in row))
    return "\n".join(lines) + "\n"


def _parse_legacy_vxc(path: Path) -> tuple[int, list[list[tuple[float, float]]]]:
    lines = path.read_text(encoding="ascii").splitlines()
    if not lines:
        raise OverlayError(f"generated legacy Vxc is empty: {path}")
    dimension = _parse_int(lines[0].strip(), f"{path.name} dimension")
    rows: list[list[tuple[float, float]]] = []
    if len(lines) != dimension + 1:
        raise OverlayError(f"generated legacy Vxc row count mismatch: {path}")
    for row_number, line in enumerate(lines[1:], start=1):
        matches = list(_COMPLEX_RE.finditer(line))
        if _COMPLEX_RE.sub("", line).strip():
            raise OverlayError(f"generated legacy Vxc contains malformed data: {path}")
        values = [
            (
                _parse_float(match.group(1), f"{path.name} real"),
                _parse_float(match.group(2), f"{path.name} imag"),
            )
            for match in matches
        ]
        expected_length = dimension - row_number + 1
        if len(values) != expected_length:
            raise OverlayError(f"generated legacy Vxc triangle mismatch: {path}")
        rows.append(values)
    return dimension, rows


def _build_legacy_vxc_views(
    source_dataset: Path,
    output_dataset: Path,
    n_kpoints: int,
    n_aos: int,
) -> list[Path]:
    if list(source_dataset.glob("vxcs1k*_nao.txt")):
        raise OverlayError("source dataset already contains legacy Vxc aliases")
    indexed: dict[int, Path] = {}
    for path in source_dataset.glob("vxck*_nao.txt"):
        match = _NATIVE_VXC_RE.fullmatch(path.name)
        if match is None:
            raise OverlayError(f"invalid native Vxc filename: {path.name}")
        indexed[_parse_int(match.group(1), "native Vxc index")] = path
    expected_indices = set(range(1, n_kpoints + 1))
    if set(indexed) != expected_indices:
        raise OverlayError(
            "native Vxc index coverage mismatch: "
            f"expected={sorted(expected_indices)} actual={sorted(indexed)}"
        )

    outputs = []
    for index in sorted(indexed):
        dimension, token_rows, numeric_rows = _parse_native_vxc(indexed[index])
        if dimension != n_aos:
            raise OverlayError(
                f"native Vxc dimension mismatch at k={index}: {dimension} != {n_aos}"
            )
        output = output_dataset / f"vxcs1k{index}_nao.txt"
        output.write_text(_render_legacy_vxc(dimension, token_rows), encoding="ascii")
        roundtrip_dimension, roundtrip_rows = _parse_legacy_vxc(output)
        if roundtrip_dimension != dimension or roundtrip_rows != numeric_rows:
            raise OverlayError(f"legacy Vxc round-trip mismatch at k={index}")
        outputs.append(output)
    return outputs


def build_overlay(
    source_dataset: Path | str,
    source_manifest: Path | str,
    output_dataset: Path | str,
) -> dict[str, object]:
    source_dataset = Path(source_dataset).resolve()
    source_manifest = Path(source_manifest).resolve()
    output_dataset = Path(output_dataset).resolve()
    if not source_dataset.is_dir() or source_dataset.is_symlink():
        raise OverlayError(f"source dataset is not a regular directory: {source_dataset}")
    if output_dataset.exists() or output_dataset.is_symlink():
        raise OverlayError(f"output dataset already exists: {output_dataset}")

    entries = _read_manifest(source_manifest, source_dataset)
    required = {"dataset/stru_out", "dataset/band_out", "dataset/bz_sampling_out"}
    if not required.issubset(entries):
        raise OverlayError(f"source manifest lacks required files: {sorted(required - set(entries))}")

    structure_text = _validate_plain_structure(source_dataset / "stru_out")
    band_kpoints, n_aos = _read_band_header(source_dataset / "band_out")
    grid, rows = _read_fullbz_sampling(source_dataset / "bz_sampling_out", band_kpoints)
    structure_overlay = structure_text
    structure_overlay += f"{grid[0]} {grid[1]} {grid[2]}\n"
    structure_overlay += "".join(" ".join(row[5:8]) + "\n" for row in rows)
    structure_overlay += "".join(row[9] + "\n" for row in rows)

    output_dataset.parent.mkdir(parents=True, exist_ok=True)
    output_dataset.mkdir()
    hardlinked = 0
    for source_path in sorted(source_dataset.rglob("*")):
        relative = source_path.relative_to(source_dataset)
        target_path = output_dataset / relative
        if source_path.is_dir():
            target_path.mkdir(exist_ok=True)
            continue
        if relative.as_posix() == "stru_out":
            continue
        target_path.parent.mkdir(parents=True, exist_ok=True)
        os.link(source_path, target_path)
        hardlinked += 1
    overlay_stru = output_dataset / "stru_out"
    overlay_stru.write_text(structure_overlay, encoding="ascii")
    legacy_vxc_outputs = _build_legacy_vxc_views(
        source_dataset, output_dataset, band_kpoints, n_aos
    )

    output_entries = dict(entries)
    output_entries["dataset/stru_out"] = _sha256(overlay_stru)
    for output in legacy_vxc_outputs:
        output_entries[f"dataset/{output.name}"] = _sha256(output)
    output_manifest = output_dataset.parent / "DATASET_SHA256SUMS.txt"
    _write_manifest(output_manifest, output_entries)
    report: dict[str, object] = {
        "status": "PASS",
        "contract": "legacy_fullbz_input_overlay_v2",
        "source_dataset": str(source_dataset),
        "source_manifest": str(source_manifest),
        "source_manifest_sha256": _sha256(source_manifest),
        "source_stru_sha256": entries["dataset/stru_out"],
        "overlay_stru_sha256": output_entries["dataset/stru_out"],
        "grid": grid,
        "n_kpoints": band_kpoints,
        "mapping": "identity",
        "unchanged_files_hardlinked": hardlinked,
        "legacy_vxc_conversion": "native_comment_row_to_dimension_upper_triangle",
        "legacy_vxc_files_generated": len(legacy_vxc_outputs),
        "legacy_vxc_values_equal": True,
        "dataset_file_count": len(output_entries),
    }
    report_path = output_dataset.parent / "LEGACY_FULLBZ_INPUT_OVERLAY.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="ascii")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_dataset", type=Path)
    parser.add_argument("source_manifest", type=Path)
    parser.add_argument("output_dataset", type=Path)
    args = parser.parse_args()
    report = build_overlay(args.source_dataset, args.source_manifest, args.output_dataset)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
