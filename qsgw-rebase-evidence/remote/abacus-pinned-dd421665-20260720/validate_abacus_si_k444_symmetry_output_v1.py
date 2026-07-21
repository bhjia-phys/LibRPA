#!/usr/bin/env python3
"""Validate a fresh reader-v0 Si k444 symmetry ABACUS producer run."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path


ABACUS_HARTREE_TO_EV = 27.211396
COMPLEX_ENTRY_RE = re.compile(r"\(\s*([^,()\s]+)\s*,\s*([^,()\s]+)\s*\)")


def fail(message: str) -> None:
    raise SystemExit(f"FAIL: {message}")


def require_file(path: Path) -> Path:
    if not path.is_file() or path.stat().st_size == 0:
        fail(f"missing or empty file: {path}")
    return path


def parse_int(token: str, context: str) -> int:
    if not re.fullmatch(r"[+-]?\d+", token):
        fail(f"{context}: expected integer, got {token!r}")
    return int(token)


def parse_float(token: str, context: str) -> float:
    try:
        value = float(token)
    except ValueError:
        fail(f"{context}: expected floating-point value, got {token!r}")
    if not math.isfinite(value):
        fail(f"{context}: non-finite value {token!r}")
    return value


def parse_bz_sampling(path: Path) -> dict[str, object]:
    tokens = require_file(path).read_text(encoding="utf-8").split()
    if len(tokens) < 5:
        fail("bz_sampling_out: truncated header")
    grid = tuple(parse_int(tokens[i], "bz_sampling_out grid") for i in range(3))
    n_scf = parse_int(tokens[3], "bz_sampling_out SCF count")
    n_ibz = parse_int(tokens[4], "bz_sampling_out IBZ count")
    if grid != (4, 4, 4):
        fail(f"bz_sampling_out: grid {grid}, expected (4, 4, 4)")
    if (n_scf, n_ibz) != (8, 8):
        fail(f"bz_sampling_out: counts {(n_scf, n_ibz)}, expected (8, 8)")
    if len(tokens) != 5 + 10 * n_scf:
        fail("bz_sampling_out: row count or width does not match header")

    weights: list[float] = []
    labels: set[int] = set()
    representatives: set[int] = set()
    offset = 5
    for row in range(n_scf):
        fields = tokens[offset : offset + 10]
        offset += 10
        index = parse_int(fields[0], f"bz_sampling_out row {row + 1}")
        if index != row + 1:
            fail(f"bz_sampling_out row {row + 1}: index is {index}")
        weights.append(parse_float(fields[1], f"bz_sampling_out row {row + 1} weight"))
        for field in fields[2:8]:
            parse_float(field, f"bz_sampling_out row {row + 1} vector")
        label = parse_int(fields[8], f"bz_sampling_out row {row + 1} IBZ label")
        representative = parse_int(
            fields[9], f"bz_sampling_out row {row + 1} representative"
        )
        if not 1 <= label <= n_ibz or not 1 <= representative <= n_scf:
            fail(f"bz_sampling_out row {row + 1}: mapping index out of range")
        labels.add(label)
        representatives.add(representative)

    if any(weight < 0.0 for weight in weights):
        fail("bz_sampling_out: negative k-point weight")
    if not math.isclose(sum(weights), 1.0, rel_tol=0.0, abs_tol=1.0e-10):
        fail(f"bz_sampling_out: weights sum to {sum(weights):.17g}")
    if labels != set(range(1, n_ibz + 1)):
        fail("bz_sampling_out: not every IBZ label is represented")
    if len(representatives) != n_ibz:
        fail("bz_sampling_out: representative cardinality does not match IBZ count")
    return {
        "grid": list(grid),
        "n_scf": n_scf,
        "n_ibz": n_ibz,
        "weight_sum": sum(weights),
    }


def determinant3(rotation: list[int]) -> int:
    a, b, c, d, e, f, g, h, i = rotation
    return a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)


def parse_stru(path: Path) -> dict[str, object]:
    tokens = require_file(path).read_text(encoding="utf-8").split()
    pos = 0
    for index in range(18):
        if pos >= len(tokens):
            fail("stru_out: truncated lattice section")
        parse_float(tokens[pos], f"stru_out lattice token {index + 1}")
        pos += 1
    if pos >= len(tokens):
        fail("stru_out: missing atom count")
    n_atoms = parse_int(tokens[pos], "stru_out atom count")
    pos += 1
    if n_atoms != 2:
        fail(f"stru_out: atom count {n_atoms}, expected 2")
    for atom in range(n_atoms):
        if pos + 4 > len(tokens):
            fail("stru_out: truncated atom section")
        for field in tokens[pos : pos + 3]:
            parse_float(field, f"stru_out atom {atom + 1} coordinate")
        atom_type = parse_int(tokens[pos + 3], f"stru_out atom {atom + 1} type")
        if atom_type != 1:
            fail(f"stru_out atom {atom + 1}: type {atom_type}, expected 1")
        pos += 4

    if pos + 2 > len(tokens):
        fail("stru_out: missing symmetry-operation header")
    n_symops = parse_int(tokens[pos], "stru_out symmetry-operation count")
    convention = tokens[pos + 1].lower()
    pos += 2
    if n_symops <= 1:
        fail(f"stru_out: expected more than identity, found {n_symops} operation(s)")
    if convention != "row":
        fail(f"stru_out: convention {convention!r}, expected 'row'")
    if len(tokens) - pos != 12 * n_symops:
        fail("stru_out: symmetry tail size does not match its header")

    identity = [1, 0, 0, 0, 1, 0, 0, 0, 1]
    identity_present = False
    determinants: list[int] = []
    for operation in range(n_symops):
        rotation = [
            parse_int(tokens[pos + i], f"stru_out operation {operation + 1} rotation")
            for i in range(9)
        ]
        translation = [
            parse_float(
                tokens[pos + 9 + i], f"stru_out operation {operation + 1} translation"
            )
            for i in range(3)
        ]
        pos += 12
        determinant = determinant3(rotation)
        if abs(determinant) != 1:
            fail(f"stru_out operation {operation + 1}: determinant {determinant}")
        determinants.append(determinant)
        if rotation == identity and all(
            math.isclose(value, round(value), rel_tol=0.0, abs_tol=1.0e-12)
            for value in translation
        ):
            identity_present = True
    if not identity_present:
        fail("stru_out: identity operation is absent")
    return {
        "n_atoms": n_atoms,
        "n_symops": n_symops,
        "convention": convention,
        "determinants": sorted(set(determinants)),
        "identity_present": identity_present,
    }


def parse_band(path: Path, expected_n_scf: int) -> dict[str, object]:
    lines = require_file(path).read_text(encoding="utf-8").splitlines()
    if len(lines) < 5:
        fail("band_out: truncated header")
    n_kpoints = parse_int(lines[0].strip(), "band_out k-point count")
    n_spins = parse_int(lines[1].strip(), "band_out spin count")
    n_bands = parse_int(lines[2].strip(), "band_out band count")
    n_basis = parse_int(lines[3].strip(), "band_out basis count")
    fermi_ha = parse_float(lines[4].strip(), "band_out Fermi energy")
    if (n_kpoints, n_spins, n_bands, n_basis) != (expected_n_scf, 1, 44, 44):
        fail(
            "band_out: dimensions "
            f"{(n_kpoints, n_spins, n_bands, n_basis)}, expected "
            f"{(expected_n_scf, 1, 44, 44)}"
        )

    pos = 5
    eigenvalues_ev: list[list[float]] = []
    for ik in range(n_kpoints):
        if pos >= len(lines):
            fail("band_out: missing k-point header")
        header = lines[pos].split()
        pos += 1
        if len(header) != 2 or [parse_int(x, "band_out k-point header") for x in header] != [ik + 1, 1]:
            fail(f"band_out: invalid header for k-point {ik + 1}")
        bands: list[float] = []
        for band in range(n_bands):
            if pos >= len(lines):
                fail("band_out: truncated band rows")
            fields = lines[pos].split()
            pos += 1
            if len(fields) != 4:
                fail(f"band_out k-point {ik + 1} band {band + 1}: expected four fields")
            if parse_int(fields[0], "band_out band index") != band + 1:
                fail(f"band_out k-point {ik + 1}: band index mismatch")
            parse_float(fields[1], "band_out occupation")
            energy_ha = parse_float(fields[2], "band_out energy Ha")
            energy_ev = parse_float(fields[3], "band_out energy eV")
            if not math.isclose(
                energy_ev,
                energy_ha * ABACUS_HARTREE_TO_EV,
                rel_tol=1.0e-12,
                abs_tol=1.0e-12,
            ):
                fail(f"band_out k-point {ik + 1} band {band + 1}: Ha/eV mismatch")
            bands.append(energy_ev)
        eigenvalues_ev.append(bands)
    if any(line.strip() for line in lines[pos:]):
        fail("band_out: unexpected trailing rows")
    vbm = max(bands[3] for bands in eigenvalues_ev)
    cbm = min(bands[4] for bands in eigenvalues_ev)
    return {
        "n_kpoints": n_kpoints,
        "n_spins": n_spins,
        "n_bands": n_bands,
        "n_basis": n_basis,
        "fermi_ha": fermi_ha,
        "vbm_ev": vbm,
        "cbm_ev": cbm,
        "gap_ev": cbm - vbm,
    }


def numbered_files(run_dir: Path, prefix: str) -> list[Path]:
    pattern = re.compile(rf"{re.escape(prefix)}_(\d+)\.txt")
    matches = []
    for path in run_dir.glob(f"{prefix}_*.txt"):
        match = pattern.fullmatch(path.name)
        if match:
            matches.append((int(match.group(1)), path))
    matches.sort()
    if not matches or [index for index, _ in matches] != list(range(len(matches))):
        fail(f"{prefix}: expected contiguous zero-based rank suffixes")
    for _, path in matches:
        require_file(path)
    return [path for _, path in matches]


def require_any(directory: Path, patterns: list[str], label: str) -> list[str]:
    matches = sorted(
        {path.name for pattern in patterns for path in directory.glob(pattern) if path.is_file() and path.stat().st_size > 0}
    )
    if not matches:
        fail(f"missing {label}; tried {patterns}")
    return matches


def parse_native_vxc_lines(
    lines: list[str], source: str, expected_dimension: int
) -> dict[str, int]:
    """Validate ABACUS's modern comment-and-Row upper-triangle schema."""

    dimensions: dict[str, int] = {}
    rows: dict[int, list[complex]] = {}
    current_row: int | None = None

    for line_number, line in enumerate(lines, start=1):
        stripped = line.strip()
        metadata = re.fullmatch(r"#\s+(rows|columns)\s+(\d+)", stripped)
        if metadata:
            key = metadata.group(1)
            if key in dimensions:
                raise ValueError(f"{source}: duplicate '# {key}' header")
            dimensions[key] = int(metadata.group(2))
            continue

        row_header = re.fullmatch(r"Row\s+(\d+)", stripped)
        if row_header:
            current_row = int(row_header.group(1))
            if current_row in rows:
                raise ValueError(f"{source}: duplicate Row {current_row}")
            rows[current_row] = []
            continue

        if not stripped or stripped.startswith("#"):
            continue
        if current_row is None:
            raise ValueError(
                f"{source}:{line_number}: matrix data appears before a Row header"
            )

        entries = list(COMPLEX_ENTRY_RE.finditer(stripped))
        remainder = COMPLEX_ENTRY_RE.sub("", stripped)
        if not entries or remainder.strip():
            raise ValueError(
                f"{source}:{line_number}: malformed upper-triangle complex entries"
            )
        for entry in entries:
            real = float(entry.group(1))
            imag = float(entry.group(2))
            if not (math.isfinite(real) and math.isfinite(imag)):
                raise ValueError(
                    f"{source}:{line_number}: non-finite complex entry"
                )
            rows[current_row].append(complex(real, imag))

    expected_dimensions = {"rows": expected_dimension, "columns": expected_dimension}
    if dimensions != expected_dimensions:
        raise ValueError(
            f"{source}: dimensions {dimensions}, expected {expected_dimensions}"
        )
    expected_rows = list(range(1, expected_dimension + 1))
    if sorted(rows) != expected_rows:
        raise ValueError(
            f"{source}: Row labels {sorted(rows)}, expected {expected_rows}"
        )
    for row in expected_rows:
        expected_entries = expected_dimension - row + 1
        if len(rows[row]) != expected_entries:
            raise ValueError(
                f"{source}: Row {row} has {len(rows[row])} complex entries; "
                f"expected {expected_entries} complex entries"
            )

    return {
        "rows": expected_dimension,
        "columns": expected_dimension,
        "row_count": len(rows),
        "upper_triangle_entries": sum(len(entries) for entries in rows.values()),
    }


def indexed_vxc_files(
    directory: Path, expected_count: int, expected_dimension: int
) -> tuple[list[Path], list[dict[str, int]]]:
    pattern = re.compile(r"vxck(\d+)_nao\.txt")
    indexed: dict[int, Path] = {}
    for path in directory.glob("vxck*_nao.txt"):
        match = pattern.fullmatch(path.name)
        if not match:
            continue
        index = int(match.group(1))
        if index in indexed:
            fail(f"duplicate Vxc k-point index {index}")
        indexed[index] = require_file(path)
    expected = list(range(1, expected_count + 1))
    if sorted(indexed) != expected:
        fail(f"Vxc KS-matrix indices {sorted(indexed)}, expected {expected}")
    summaries: list[dict[str, int]] = []
    for index in expected:
        path = indexed[index]
        try:
            summaries.append(
                parse_native_vxc_lines(
                    path.read_text(encoding="utf-8").splitlines(),
                    path.name,
                    expected_dimension,
                )
            )
        except (ValueError, OverflowError) as error:
            fail(str(error))
    return [indexed[index] for index in expected], summaries


def validate_run(run_dir: Path, mpi_ranks: int) -> dict[str, object]:
    out_dir = run_dir / "OUT.ABACUS"
    if not out_dir.is_dir():
        fail(f"missing directory: {out_dir}")
    running_log = require_file(out_dir / "running_scf.log").read_text(
        encoding="utf-8", errors="replace"
    )
    if "SCF IS CONVERGED" not in running_log:
        fail("OUT.ABACUS/running_scf.log: SCF did not converge")
    require_file(out_dir / "INPUT.info")

    bz = parse_bz_sampling(run_dir / "bz_sampling_out")
    structure = parse_stru(run_dir / "stru_out")
    band = parse_band(run_dir / "band_out", int(bz["n_scf"]))

    eigenvectors = sorted(run_dir.glob("KS_eigenvector_*.dat"))
    expected_eigenvectors = [run_dir / f"KS_eigenvector_{ik}.dat" for ik in range(int(bz["n_scf"]))]
    if eigenvectors != expected_eigenvectors:
        fail("KS eigenvector file set does not match the SCF k-point count")
    for path in eigenvectors:
        require_file(path)

    cs = numbered_files(run_dir, "Cs_data")
    coulomb_cut = numbered_files(run_dir, "coulomb_cut")
    coulomb_full = numbered_files(run_dir, "coulomb_mat")
    if not (len(cs) == len(coulomb_cut) == len(coulomb_full) == mpi_ranks):
        fail(
            "rank-file counts do not match MPI ranks: "
            f"Cs={len(cs)}, cut={len(coulomb_cut)}, full={len(coulomb_full)}, mpi={mpi_ranks}"
        )

    vxc_summary = require_file(out_dir / "vxc_out.dat").read_text(
        encoding="utf-8"
    ).splitlines()
    if len(vxc_summary) < 3 or [line.strip() for line in vxc_summary[:3]] != [
        "8",
        "1",
        "44",
    ]:
        fail("vxc_out.dat: expected header 8 k-points, 1 spin, 44 bands")
    require_file(out_dir / "Vxc_R_spin0.csr")
    vxc_matrices, vxc_matrix_summaries = indexed_vxc_files(
        out_dir, int(bz["n_scf"]), int(band["n_basis"])
    )

    sparse = {
        "hamiltonian": require_any(
            out_dir,
            ["data-HR-sparse_SPIN0.csr", "hrs1_nao.csr", "*HR*sparse*SPIN0*.csr"],
            "real-space Hamiltonian matrix",
        ),
        "overlap": require_any(
            out_dir,
            ["data-SR-sparse_SPIN0.csr", "srs1_nao.csr", "*SR*sparse*SPIN0*.csr"],
            "real-space overlap matrix",
        ),
        "position": require_any(
            out_dir,
            ["rr.csr", "data-rR-sparse_SPIN0.csr", "*rR*sparse*SPIN0*.csr"],
            "real-space position matrix",
        ),
    }

    basis_sidecars = [
        run_dir / "basis_wfc_out",
        run_dir / "basis_aux_out",
        run_dir / "basis_out",
    ]
    stale_basis = [path.name for path in basis_sidecars if path.exists()]
    if stale_basis:
        fail(f"reader-v0 run unexpectedly contains basis sidecars: {stale_basis}")

    return {
        "status": "PASS",
        "reader_version": 0,
        "vxc_filename_schema": "vxck<one-based-k-index>_nao.txt",
        "vxc_matrix_schema": {
            "format": "abacus_modern_comment_row_upper_triangle",
            "rows": int(band["n_basis"]),
            "columns": int(band["n_basis"]),
            "upper_triangle_entries_per_matrix": int(band["n_basis"])
            * (int(band["n_basis"]) + 1)
            // 2,
            "validated_matrix_count": len(vxc_matrix_summaries),
        },
        "basis_metadata_status": "not_emitted_by_reader_v0; pending frozen dataset assembly",
        "bz_sampling": bz,
        "structure": structure,
        "band": band,
        "file_counts": {
            "KS_eigenvector": len(eigenvectors),
            "Cs_data": len(cs),
            "coulomb_cut": len(coulomb_cut),
            "coulomb_mat": len(coulomb_full),
            "vxc_ks_matrix": len(vxc_matrices),
        },
        "sparse_outputs": sparse,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--mpi-ranks", type=int, required=True)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if args.mpi_ranks <= 0:
        fail("--mpi-ranks must be positive")
    report = validate_run(args.run_dir.resolve(), args.mpi_ranks)
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.report:
        args.report.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
