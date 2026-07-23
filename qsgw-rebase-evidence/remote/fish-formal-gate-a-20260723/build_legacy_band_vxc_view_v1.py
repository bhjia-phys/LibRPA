#!/usr/bin/env python3
"""Build the legacy QSGW band-Vxc filenames from a frozen v2 manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path


MAGIC = "# librpa-qsgw-vxc-manifest-v2"
HEADER = "spin k_index kx ky kz rows columns sha256 file"
NUMBER = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eEdD][+-]?\d+)?"
COMPLEX = re.compile(rf"\(\s*({NUMBER})\s*,\s*({NUMBER})\s*\)")
ROWS = re.compile(r"^\s*#\s*rows\s+(\d+)\s*$", re.IGNORECASE)
COLUMNS = re.compile(r"^\s*#\s*columns\s+(\d+)\s*$", re.IGNORECASE)
ROW_MARKER = re.compile(r"^\s*Row\s+(\d+)\s*$", re.IGNORECASE)
SHA256 = re.compile(r"^[0-9a-f]{64}$")


class LegacyBandVxcError(RuntimeError):
    """Raised when the source cannot produce an exact legacy view."""


@dataclass(frozen=True)
class ManifestEntry:
    spin: int
    k_index: int
    rows: int
    columns: int
    digest: str
    file: str


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def finite_float(token: str, context: str) -> float:
    try:
        value = float(token.replace("D", "E").replace("d", "e"))
    except ValueError as error:
        raise LegacyBandVxcError(
            f"invalid number for {context}: {token}"
        ) from error
    if not math.isfinite(value):
        raise LegacyBandVxcError(f"non-finite number for {context}: {token}")
    return value


def parse_manifest(path: Path) -> list[ManifestEntry]:
    if not path.is_file() or path.is_symlink():
        raise LegacyBandVxcError(f"manifest is not a regular file: {path}")
    lines = [
        line.strip()
        for line in path.read_text(encoding="ascii").splitlines()
        if line.strip()
    ]
    if not lines or lines[0] != MAGIC:
        raise LegacyBandVxcError(f"invalid manifest magic: {path}")
    metadata: dict[str, str] = {}
    entries: list[ManifestEntry] = []
    saw_header = False
    seen: set[tuple[int, int]] = set()
    for line in lines[1:]:
        if line == HEADER:
            if saw_header:
                raise LegacyBandVxcError("duplicate manifest table header")
            saw_header = True
            continue
        fields = line.split()
        if not saw_header:
            if len(fields) != 2 or fields[0] in metadata:
                raise LegacyBandVxcError(f"malformed manifest metadata: {line}")
            metadata[fields[0]] = fields[1]
            continue
        if len(fields) != 9:
            raise LegacyBandVxcError(f"malformed manifest entry: {line}")
        try:
            spin = int(fields[0])
            k_index = int(fields[1])
            rows = int(fields[5])
            columns = int(fields[6])
        except ValueError as error:
            raise LegacyBandVxcError(
                f"invalid integer in manifest entry: {line}"
            ) from error
        for axis, token in enumerate(fields[2:5]):
            finite_float(token, f"k-point axis {axis + 1}")
        digest = fields[7]
        if (
            spin <= 0
            or k_index <= 0
            or rows <= 0
            or columns <= 0
            or not SHA256.fullmatch(digest)
        ):
            raise LegacyBandVxcError(f"invalid manifest entry: {line}")
        key = (spin, k_index)
        if key in seen:
            raise LegacyBandVxcError(f"duplicate manifest entry: {key}")
        seen.add(key)
        entries.append(
            ManifestEntry(
                spin, k_index, rows, columns, digest, fields[8]
            )
        )
    expected_metadata = {
        "kind": "band",
        "producer": "abacus",
        "units": "Ry",
        "basis": "state",
        "gauge": "mf0_state",
    }
    if metadata != expected_metadata:
        raise LegacyBandVxcError(
            "legacy parity requires ABACUS Ry state/mf0_state band Vxc; "
            f"found {metadata}"
        )
    if not saw_header or not entries:
        raise LegacyBandVxcError("manifest contains no band Vxc entries")
    for spin in sorted({entry.spin for entry in entries}):
        indices = sorted(
            entry.k_index for entry in entries if entry.spin == spin
        )
        if indices != list(range(1, len(indices) + 1)):
            raise LegacyBandVxcError(
                f"non-contiguous k-point coverage for spin {spin}: {indices}"
            )
    return entries


def parse_native_matrix(path: Path) -> list[list[complex]]:
    n_rows: int | None = None
    n_columns: int | None = None
    current_row: int | None = None
    values: dict[int, list[complex]] = {}
    for lineno, line in enumerate(
        path.read_text(encoding="ascii").splitlines(), start=1
    ):
        match = ROWS.fullmatch(line)
        if match:
            n_rows = int(match.group(1))
            continue
        match = COLUMNS.fullmatch(line)
        if match:
            n_columns = int(match.group(1))
            continue
        match = ROW_MARKER.fullmatch(line)
        if match:
            current_row = int(match.group(1))
            if current_row in values:
                raise LegacyBandVxcError(
                    f"duplicate Row {current_row} in {path}"
                )
            values[current_row] = []
            continue
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if current_row is None:
            raise LegacyBandVxcError(
                f"values before Row marker in {path}:{lineno}"
            )
        matches = list(COMPLEX.finditer(line))
        if not matches or COMPLEX.sub("", line).strip():
            raise LegacyBandVxcError(
                f"malformed complex values in {path}:{lineno}"
            )
        values[current_row].extend(
            complex(
                finite_float(match.group(1), f"{path}:{lineno} real"),
                finite_float(match.group(2), f"{path}:{lineno} imag"),
            )
            for match in matches
        )
    if (
        n_rows is None
        or n_columns is None
        or n_rows <= 0
        or n_rows != n_columns
    ):
        raise LegacyBandVxcError(
            f"native Vxc must declare a positive square matrix: {path}"
        )
    if set(values) != set(range(1, n_rows + 1)):
        raise LegacyBandVxcError(f"incomplete Row coverage in {path}")
    ordered = [values[row] for row in range(1, n_rows + 1)]
    for row, row_values in enumerate(ordered, start=1):
        if len(row_values) != n_rows - row + 1:
            raise LegacyBandVxcError(
                f"upper-triangle length mismatch in {path}, Row {row}"
            )
        if abs(row_values[0].imag) > 1.0e-12:
            raise LegacyBandVxcError(
                f"complex diagonal in {path}, Row {row}"
            )
    return ordered


def render_legacy_matrix(values: list[list[complex]]) -> str:
    lines = [str(len(values))]
    lines.extend(
        " ".join(
            f"({value.real:.16e},{value.imag:.16e})" for value in row
        )
        for row in values
    )
    return "\n".join(lines) + "\n"


def parse_legacy_matrix(text: str, source_name: str) -> list[list[complex]]:
    lines = text.splitlines()
    if not lines:
        raise LegacyBandVxcError(f"empty legacy matrix: {source_name}")
    try:
        dimension = int(lines[0])
    except ValueError as error:
        raise LegacyBandVxcError(
            f"invalid legacy dimension: {source_name}"
        ) from error
    if dimension <= 0 or len(lines) != dimension + 1:
        raise LegacyBandVxcError(
            f"legacy row count mismatch: {source_name}"
        )
    result: list[list[complex]] = []
    for row, line in enumerate(lines[1:], start=1):
        matches = list(COMPLEX.finditer(line))
        if COMPLEX.sub("", line).strip() or len(matches) != dimension - row + 1:
            raise LegacyBandVxcError(
                f"legacy triangle mismatch: {source_name}, Row {row}"
            )
        result.append(
            [
                complex(
                    finite_float(match.group(1), f"{source_name} real"),
                    finite_float(match.group(2), f"{source_name} imag"),
                )
                for match in matches
            ]
        )
    return result


def resolve_source(dataset: Path, relative: str) -> Path:
    candidate = Path(relative)
    if candidate.is_absolute() or not relative:
        raise LegacyBandVxcError(f"unsafe manifest path: {relative}")
    source = (dataset / candidate).resolve()
    try:
        source.relative_to(dataset)
    except ValueError as error:
        raise LegacyBandVxcError(
            f"manifest path escapes dataset: {relative}"
        ) from error
    if not source.is_file() or source.is_symlink():
        raise LegacyBandVxcError(
            f"Vxc source is not a regular file: {source}"
        )
    return source


def build_legacy_band_vxc_view(
    dataset: Path | str,
    manifest: Path | str,
    output_dir: Path | str,
) -> dict[str, object]:
    dataset = Path(dataset).resolve()
    manifest = Path(manifest).resolve()
    output_dir = Path(output_dir).resolve()
    if not dataset.is_dir() or dataset.is_symlink():
        raise LegacyBandVxcError(f"dataset is not a regular directory: {dataset}")
    if not output_dir.is_dir() or output_dir.is_symlink():
        raise LegacyBandVxcError(
            f"output is not a regular directory: {output_dir}"
        )
    entries = parse_manifest(manifest)
    prepared: list[tuple[Path, str, list[list[complex]]]] = []
    for entry in entries:
        source = resolve_source(dataset, entry.file)
        if sha256(source) != entry.digest:
            raise LegacyBandVxcError(
                f"Vxc source hash mismatch: {entry.file}"
            )
        values = parse_native_matrix(source)
        if (
            len(values) != entry.rows
            or entry.rows != entry.columns
        ):
            raise LegacyBandVxcError(
                f"Vxc dimensions disagree with manifest: {entry.file}"
            )
        target = output_dir / (
            f"band_vxcs{entry.spin}k{entry.k_index}_nao.txt"
        )
        if target.exists() or target.is_symlink():
            raise LegacyBandVxcError(
                f"refusing to overwrite legacy Vxc view: {target}"
            )
        rendered = render_legacy_matrix(values)
        if parse_legacy_matrix(rendered, str(target)) != values:
            raise LegacyBandVxcError(
                f"legacy numeric round-trip failed: {entry.file}"
            )
        prepared.append((target, rendered, values))
    for target, rendered, _values in prepared:
        target.write_text(rendered, encoding="ascii", newline="\n")
    return {
        "schema": "librpa-legacy-band-vxc-view-v1",
        "status": "PASS",
        "source_dataset": str(dataset),
        "source_manifest": str(manifest),
        "source_manifest_sha256": sha256(manifest),
        "source_basis": "state",
        "source_gauge": "mf0_state",
        "source_units": "Ry",
        "legacy_filename_schema": "band_vxcs<spin>k<index>_nao.txt",
        "energy_conversion": "legacy_reader_Ry_to_Ha",
        "files_generated": len(prepared),
        "numeric_roundtrip_equal": True,
        "output_sha256": {
            target.name: sha256(target) for target, _rendered, _values in prepared
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()
    if args.report.exists() or args.report.is_symlink():
        raise LegacyBandVxcError(
            f"refusing to overwrite report: {args.report}"
        )
    report = build_legacy_band_vxc_view(
        args.dataset, args.manifest, args.output_dir
    )
    args.report.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="ascii",
    )
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
