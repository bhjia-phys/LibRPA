#!/usr/bin/env python3
"""Validate a pinned ABACUS NSCF band-path output before preprocessing."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path


FLOAT = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][+-]?\d+)?"


@dataclass(frozen=True)
class WfcData:
    k_index: int
    k_cartesian: tuple[float, float, float]
    n_bands: int
    n_basis: int
    eigenvalues_ry: list[float]
    occupations: list[float]
    coefficients: list[complex]


@dataclass(frozen=True)
class DiagonalVxcData:
    n_kpoints: int
    n_spins: int
    n_bands: int
    values_ha: list[float]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_file(path: Path) -> Path:
    if not path.is_file():
        raise ValueError(f"required NSCF output is missing: {path}")
    return path


def finite_float(token: str, label: str) -> float:
    try:
        value = float(token.replace("D", "E").replace("d", "e"))
    except ValueError as error:
        raise ValueError(f"invalid numeric value in {label}: {token}") from error
    if not math.isfinite(value):
        raise ValueError(f"non-finite numeric value in {label}: {token}")
    return value


def first_integer(line: str, label: str) -> int:
    fields = line.split()
    if not fields:
        raise ValueError(f"missing integer in {label}")
    try:
        value = int(fields[0])
    except ValueError as error:
        raise ValueError(f"invalid integer in {label}: {fields[0]}") from error
    if value <= 0:
        raise ValueError(f"non-positive integer in {label}: {value}")
    return value


def parse_kpt_info_text(
    text: str, source_name: str
) -> list[tuple[float, float, float]]:
    lines = text.splitlines()
    header = next(
        (
            index
            for index, line in enumerate(lines)
            if line.strip() == "K-POINTS DIRECT COORDINATES"
        ),
        None,
    )
    if header is None:
        raise ValueError(f"missing direct-coordinate table in {source_name}")
    row_pattern = re.compile(
        rf"^\s*(\d+)\s+({FLOAT})\s+({FLOAT})\s+({FLOAT})\s+({FLOAT})\s*$"
    )
    rows: list[tuple[int, tuple[float, float, float]]] = []
    for line in lines[header + 1 :]:
        match = row_pattern.fullmatch(line)
        if match:
            index = int(match.group(1))
            coordinates = tuple(
                finite_float(match.group(offset), source_name)
                for offset in (2, 3, 4)
            )
            finite_float(match.group(5), source_name)
            rows.append((index, coordinates))
        elif rows:
            break
    if not rows:
        raise ValueError(f"empty direct-coordinate table in {source_name}")
    indices = [index for index, _ in rows]
    if indices != list(range(1, len(rows) + 1)):
        raise ValueError(f"k-point indices are not contiguous in {source_name}")
    return [coordinates for _, coordinates in rows]


def parse_wfc_text(text: str, source_name: str) -> WfcData:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) < 10:
        raise ValueError(f"incomplete ABACUS WFC text file: {source_name}")
    k_index = first_integer(lines[0], f"{source_name} k index")
    k_fields = lines[1].split()
    if len(k_fields) != 3:
        raise ValueError(f"invalid WFC k-vector in {source_name}")
    k_cartesian = tuple(finite_float(value, source_name) for value in k_fields)
    n_bands = first_integer(lines[2], f"{source_name} band count")
    n_basis = first_integer(lines[3], f"{source_name} orbital count")

    band_pattern = re.compile(r"^(\d+)\s+\(band\)$")
    markers = [
        index
        for index, line in enumerate(lines[4:], start=4)
        if band_pattern.fullmatch(line)
    ]
    if len(markers) != n_bands:
        raise ValueError(f"WFC band marker count mismatch in {source_name}")
    eigenvalues: list[float] = []
    occupations: list[float] = []
    coefficients: list[complex] = []
    for band_offset, start in enumerate(markers):
        stop = markers[band_offset + 1] if band_offset + 1 < len(markers) else len(lines)
        block = lines[start:stop]
        match = band_pattern.fullmatch(block[0])
        if match is None or int(match.group(1)) != band_offset + 1:
            raise ValueError(f"WFC band indices are not contiguous in {source_name}")
        if len(block) < 4:
            raise ValueError(f"incomplete WFC band block in {source_name}")
        eigenvalues.append(finite_float(block[1].split()[0], source_name))
        occupations.append(finite_float(block[2].split()[0], source_name))
        values = [
            finite_float(token, source_name)
            for token in " ".join(block[3:]).split()
        ]
        if len(values) != 2 * n_basis:
            raise ValueError(
                f"WFC coefficient count mismatch for band {band_offset + 1} "
                f"in {source_name}: {len(values)} != {2 * n_basis}"
            )
        coefficients.extend(
            complex(values[index], values[index + 1])
            for index in range(0, len(values), 2)
        )
    return WfcData(
        k_index=k_index,
        k_cartesian=k_cartesian,
        n_bands=n_bands,
        n_basis=n_basis,
        eigenvalues_ry=eigenvalues,
        occupations=occupations,
        coefficients=coefficients,
    )


def parse_vxc_out_text(text: str, source_name: str) -> DiagonalVxcData:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) < 4:
        raise ValueError(f"incomplete diagonal Vxc output: {source_name}")
    dimensions = [first_integer(lines[index], source_name) for index in range(3)]
    n_kpoints, n_spins, n_bands = dimensions
    expected = n_kpoints * n_spins * n_bands
    if len(lines[3:]) != expected:
        raise ValueError(
            f"diagonal Vxc row count mismatch in {source_name}: "
            f"{len(lines[3:])} != {expected}"
        )
    values: list[float] = []
    for line in lines[3:]:
        fields = line.split()
        if len(fields) < 2:
            raise ValueError(f"malformed diagonal Vxc row in {source_name}")
        values.append(finite_float(fields[0], source_name))
        finite_float(fields[1], source_name)
    return DiagonalVxcData(n_kpoints, n_spins, n_bands, values)


def parse_native_vxc_text(text: str, source_name: str) -> tuple[int, int]:
    dimensions: dict[str, int] = {}
    for line in text.splitlines():
        match = re.fullmatch(r"#\s+(rows|columns)\s+(\d+)", line.strip())
        if match:
            dimensions[match.group(1)] = int(match.group(2))
    if set(dimensions) != {"rows", "columns"}:
        raise ValueError(f"incomplete native Vxc dimensions in {source_name}")
    dimension = dimensions["rows"]
    if dimension <= 0 or dimensions["columns"] != dimension:
        raise ValueError(f"non-square native Vxc matrix in {source_name}")
    complex_pattern = re.compile(rf"\(\s*({FLOAT})\s*,\s*({FLOAT})\s*\)")
    entries = complex_pattern.findall(text)
    expected = dimension * (dimension + 1) // 2
    if len(entries) != expected:
        raise ValueError(
            f"native Vxc upper-triangle count mismatch in {source_name}: "
            f"{len(entries)} != {expected}"
        )
    for real, imaginary in entries:
        finite_float(real, source_name)
        finite_float(imaginary, source_name)
    return dimension, len(entries)


def indexed_names(
    names: list[str], pattern: str, label: str
) -> list[tuple[int, str]]:
    regex = re.compile(pattern)
    indexed: dict[int, str] = {}
    for name in names:
        match = regex.fullmatch(name)
        if not match:
            continue
        index = int(match.group(1))
        if index in indexed:
            raise ValueError(f"duplicate {label} index {index}")
        indexed[index] = name
    if not indexed:
        raise ValueError(f"no indexed {label} files found")
    expected = list(range(1, max(indexed) + 1))
    if sorted(indexed) != expected:
        raise ValueError(f"{label} indices are not contiguous")
    return [(index, indexed[index]) for index in expected]


def parse_abacus_input(path: Path) -> dict[str, str]:
    lines = require_file(path).read_text(encoding="utf-8").splitlines()
    if not lines or lines[0].strip() != "INPUT_PARAMETERS":
        raise ValueError(f"missing INPUT_PARAMETERS header: {path}")
    values: dict[str, str] = {}
    for raw in lines[1:]:
        fields = raw.split("#", 1)[0].split()
        if not fields:
            continue
        if len(fields) < 2 or fields[0].lower() in values:
            raise ValueError(f"malformed or duplicate ABACUS input setting: {raw}")
        values[fields[0].lower()] = " ".join(fields[1:])
    return values


def validate_output(
    run_dir: Path,
    expected_charge_sha256: str,
    expected_n_bands: int,
    expected_n_basis: int,
    expected_n_spins: int,
) -> dict[str, object]:
    run_dir = run_dir.resolve(strict=True)
    output = run_dir / "OUT.ABACUS"
    if not output.is_dir():
        raise ValueError(f"missing ABACUS output directory: {output}")
    if not re.fullmatch(r"[0-9a-f]{64}", expected_charge_sha256):
        raise ValueError("expected charge SHA256 is malformed")

    settings = parse_abacus_input(run_dir / "INPUT")
    required_settings = {
        "calculation": "nscf",
        "nbands": str(expected_n_bands),
        "basis_type": "lcao",
        "symmetry": "-1",
        "init_chg": "file",
        "out_app_flag": "0",
        "out_mat_xc": "1",
        "out_mat_xc2": "1",
        "out_wfc_lcao": "1",
    }
    for key, expected in required_settings.items():
        if settings.get(key) != expected:
            raise ValueError(
                f"NSCF INPUT setting mismatch: {key}={settings.get(key)!r}, "
                f"expected {expected!r}"
            )

    running = require_file(output / "running_nscf.log").read_text(
        encoding="utf-8", errors="replace"
    )
    if "NONSELF-CONSISTENT" not in running or "Finish Time" not in running:
        raise ValueError("ABACUS NSCF completion markers are missing")
    require_file(output / "INPUT.info")
    require_file(run_dir / "KPT")
    require_file(run_dir / "STRU")

    charge = require_file(output / "ABACUS-CHARGE-DENSITY.restart")
    charge_sha = sha256_file(charge)
    if charge_sha != expected_charge_sha256:
        raise ValueError(
            f"SCF charge restart changed during NSCF: {charge_sha}"
        )

    kpt_info = require_file(output / "KPT.info")
    kpoints = parse_kpt_info_text(
        kpt_info.read_text(encoding="utf-8"), str(kpt_info)
    )
    wfc_dir = output / "WFC"
    if not wfc_dir.is_dir():
        raise ValueError(f"missing ABACUS WFC directory: {wfc_dir}")
    wfc_files = indexed_names(
        [path.name for path in wfc_dir.iterdir() if path.is_file()],
        r"wfk(\d+)_nao\.txt",
        "WFC",
    )
    vxc_files = indexed_names(
        [path.name for path in output.iterdir() if path.is_file()],
        r"vxck(\d+)_nao\.txt",
        "Vxc",
    )
    if len(wfc_files) != len(kpoints) or len(vxc_files) != len(kpoints):
        raise ValueError(
            "KPT/WFC/Vxc cardinalities disagree: "
            f"{len(kpoints)}/{len(wfc_files)}/{len(vxc_files)}"
        )

    wfc_hashes: dict[str, str] = {}
    vxc_hashes: dict[str, str] = {}
    native_entries = expected_n_bands * (expected_n_bands + 1) // 2
    for (index, wfc_name), (_, vxc_name) in zip(wfc_files, vxc_files):
        wfc_path = wfc_dir / wfc_name
        wfc = parse_wfc_text(
            wfc_path.read_text(encoding="utf-8"), str(wfc_path)
        )
        if (
            wfc.k_index != index
            or wfc.n_bands != expected_n_bands
            or wfc.n_basis != expected_n_basis
        ):
            raise ValueError(f"WFC dimensions or index mismatch: {wfc_path}")
        vxc_path = output / vxc_name
        dimension, entries = parse_native_vxc_text(
            vxc_path.read_text(encoding="utf-8"), str(vxc_path)
        )
        if dimension != expected_n_bands or entries != native_entries:
            raise ValueError(f"native Vxc state dimension mismatch: {vxc_path}")
        wfc_hashes[wfc_name] = sha256_file(wfc_path)
        vxc_hashes[vxc_name] = sha256_file(vxc_path)

    diagonal_path = require_file(output / "vxc_out.dat")
    diagonal = parse_vxc_out_text(
        diagonal_path.read_text(encoding="utf-8"), str(diagonal_path)
    )
    if (
        diagonal.n_kpoints != len(kpoints)
        or diagonal.n_spins != expected_n_spins
        or diagonal.n_bands != expected_n_bands
    ):
        raise ValueError("diagonal Vxc dimensions disagree with NSCF outputs")

    return {
        "schema": "abacus-si-band-nscf-output-validation-v1",
        "status": "PASS",
        "n_kpoints": len(kpoints),
        "n_spins": expected_n_spins,
        "n_bands": expected_n_bands,
        "n_basis": expected_n_basis,
        "kpoint_first": list(kpoints[0]),
        "kpoint_last": list(kpoints[-1]),
        "wfc_directory": "OUT.ABACUS/WFC",
        "wfc_filename_schema": "wfk<one-based-k-index>_nao.txt",
        "wfc_layout": "band_major_complex_pairs",
        "wfc_hashes": wfc_hashes,
        "vxc_filename_schema": "vxck<one-based-k-index>_nao.txt",
        "vxc_matrix_basis": "ks_state",
        "vxc_matrix_dimension": expected_n_bands,
        "vxc_upper_triangle_entries": native_entries,
        "vxc_hashes": vxc_hashes,
        "diagonal_vxc_units": "Ha",
        "diagonal_vxc_sha256": sha256_file(diagonal_path),
        "charge_restart_sha256": charge_sha,
        "kpt_info_sha256": sha256_file(kpt_info),
        "input_sha256": sha256_file(run_dir / "INPUT"),
        "input_info_sha256": sha256_file(output / "INPUT.info"),
        "running_log_sha256": sha256_file(output / "running_nscf.log"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--expected-charge-sha256", required=True)
    parser.add_argument("--expected-n-bands", type=int, default=44)
    parser.add_argument("--expected-n-basis", type=int, default=44)
    parser.add_argument("--expected-n-spins", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    for value, label in (
        (args.expected_n_bands, "bands"),
        (args.expected_n_basis, "basis"),
        (args.expected_n_spins, "spins"),
    ):
        if value <= 0:
            raise ValueError(f"expected {label} must be positive")
    if args.output.exists():
        raise ValueError(f"refusing to overwrite validation report: {args.output}")
    report = validate_output(
        args.run_dir,
        args.expected_charge_sha256,
        args.expected_n_bands,
        args.expected_n_basis,
        args.expected_n_spins,
    )
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    args.output.write_text(rendered, encoding="ascii")
    print(rendered, end="")


if __name__ == "__main__":
    main()
