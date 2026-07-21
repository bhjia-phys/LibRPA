#!/usr/bin/env python3
"""Validate a fresh reader-v0 Si k444 full-BZ ABACUS producer run."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import validate_abacus_si_k444_symmetry_output_v1 as shared


def fail(message: str) -> None:
    shared.fail(message)


def periodic_key(values: tuple[float, float, float]) -> tuple[float, float, float]:
    wrapped = []
    for value in values:
        reduced = value - math.floor(value)
        if math.isclose(reduced, 1.0, rel_tol=0.0, abs_tol=1.0e-12):
            reduced = 0.0
        wrapped.append(round(reduced, 12))
    return tuple(wrapped)


def parse_bz_sampling(path: Path) -> dict[str, object]:
    tokens = shared.require_file(path).read_text(encoding="utf-8").split()
    if len(tokens) < 5:
        fail("bz_sampling_out: truncated header")
    grid = tuple(shared.parse_int(tokens[i], "bz_sampling_out grid") for i in range(3))
    n_scf = shared.parse_int(tokens[3], "bz_sampling_out SCF count")
    n_ibz = shared.parse_int(tokens[4], "bz_sampling_out IBZ count")
    if grid != (4, 4, 4):
        fail(f"bz_sampling_out: grid {grid}, expected (4, 4, 4)")
    if (n_scf, n_ibz) != (64, 64):
        fail(
            "bz_sampling_out: full-BZ counts "
            f"{(n_scf, n_ibz)}, expected (64, 64)"
        )
    if len(tokens) != 5 + 10 * n_scf:
        fail("bz_sampling_out: row count or width does not match header")

    expected_weight = 1.0 / 64.0
    weights: list[float] = []
    labels: set[int] = set()
    representatives: set[int] = set()
    kpoint_keys: set[tuple[float, float, float]] = set()
    offset = 5
    for row in range(n_scf):
        fields = tokens[offset : offset + 10]
        offset += 10
        index = shared.parse_int(fields[0], f"bz_sampling_out row {row + 1}")
        if index != row + 1:
            fail(f"bz_sampling_out row {row + 1}: index is {index}")
        weight = shared.parse_float(
            fields[1], f"bz_sampling_out row {row + 1} weight"
        )
        if not math.isclose(
            weight, expected_weight, rel_tol=0.0, abs_tol=1.0e-12
        ):
            fail(
                f"bz_sampling_out row {row + 1}: expected uniform 1/64 "
                f"weight, got {weight:.17g}"
            )
        vectors = tuple(
            shared.parse_float(field, f"bz_sampling_out row {row + 1} vector")
            for field in fields[2:8]
        )
        key = periodic_key(vectors[:3])
        if key in kpoint_keys:
            fail("bz_sampling_out: periodic k-points are not unique")
        kpoint_keys.add(key)
        label = shared.parse_int(
            fields[8], f"bz_sampling_out row {row + 1} IBZ label"
        )
        representative = shared.parse_int(
            fields[9], f"bz_sampling_out row {row + 1} representative"
        )
        if not 1 <= label <= n_ibz or not 1 <= representative <= n_scf:
            fail(f"bz_sampling_out row {row + 1}: mapping index out of range")
        labels.add(label)
        representatives.add(representative)
        weights.append(weight)

    if labels != set(range(1, 65)) or representatives != set(range(1, 65)):
        fail("bz_sampling_out: full-BZ labels/representatives are not bijective")
    if not math.isclose(sum(weights), 1.0, rel_tol=0.0, abs_tol=1.0e-10):
        fail(f"bz_sampling_out: weights sum to {sum(weights):.17g}")
    return {
        "grid": list(grid),
        "n_scf": n_scf,
        "n_ibz": n_ibz,
        "weight_sum": sum(weights),
        "uniform_weight": expected_weight,
        "periodic_unique_kpoints": len(kpoint_keys),
        "full_bz": True,
    }


def parse_stru(path: Path) -> dict[str, object]:
    tokens = shared.require_file(path).read_text(encoding="utf-8").split()
    pos = 0
    for index in range(18):
        if pos >= len(tokens):
            fail("stru_out: truncated lattice section")
        shared.parse_float(tokens[pos], f"stru_out lattice token {index + 1}")
        pos += 1
    if pos >= len(tokens):
        fail("stru_out: missing atom count")
    n_atoms = shared.parse_int(tokens[pos], "stru_out atom count")
    pos += 1
    if n_atoms != 2:
        fail(f"stru_out: atom count {n_atoms}, expected 2")
    for atom in range(n_atoms):
        if pos + 4 > len(tokens):
            fail("stru_out: truncated atom section")
        for field in tokens[pos : pos + 3]:
            shared.parse_float(field, f"stru_out atom {atom + 1} coordinate")
        atom_type = shared.parse_int(
            tokens[pos + 3], f"stru_out atom {atom + 1} type"
        )
        if atom_type != 1:
            fail(f"stru_out atom {atom + 1}: type {atom_type}, expected 1")
        pos += 4

    if pos != len(tokens):
        fail("stru_out: unexpected symmetry-operation metadata for symmetry=-1")
    return {
        "n_atoms": n_atoms,
        "n_symops": 0,
        "convention": "absent",
        "determinants": [],
        "identity_present": False,
    }


def indexed_eigenvectors(run_dir: Path, expected_count: int) -> list[Path]:
    expression = re.compile(r"KS_eigenvector_(\d+)\.dat")
    indexed: dict[int, Path] = {}
    for path in run_dir.glob("KS_eigenvector_*.dat"):
        match = expression.fullmatch(path.name)
        if match:
            index = int(match.group(1))
            if index in indexed:
                fail(f"duplicate KS eigenvector index {index}")
            indexed[index] = shared.require_file(path)
    expected = list(range(expected_count))
    if sorted(indexed) != expected:
        fail("KS eigenvector file set does not match the full-BZ count")
    return [indexed[index] for index in expected]


def validate_run(run_dir: Path, mpi_ranks: int) -> dict[str, object]:
    out_dir = run_dir / "OUT.ABACUS"
    if not out_dir.is_dir():
        fail(f"missing directory: {out_dir}")
    running_log = shared.require_file(out_dir / "running_scf.log").read_text(
        encoding="utf-8", errors="replace"
    )
    if "SCF IS CONVERGED" not in running_log:
        fail("OUT.ABACUS/running_scf.log: SCF did not converge")
    shared.require_file(out_dir / "INPUT.info")

    bz = parse_bz_sampling(run_dir / "bz_sampling_out")
    structure = parse_stru(run_dir / "stru_out")
    band = shared.parse_band(run_dir / "band_out", int(bz["n_scf"]))
    eigenvectors = indexed_eigenvectors(run_dir, int(bz["n_scf"]))

    cs = shared.numbered_files(run_dir, "Cs_data")
    coulomb_cut = shared.numbered_files(run_dir, "coulomb_cut")
    coulomb_full = shared.numbered_files(run_dir, "coulomb_mat")
    if not (len(cs) == len(coulomb_cut) == len(coulomb_full) == mpi_ranks):
        fail(
            "rank-file counts do not match MPI ranks: "
            f"Cs={len(cs)}, cut={len(coulomb_cut)}, "
            f"full={len(coulomb_full)}, mpi={mpi_ranks}"
        )

    vxc_summary = shared.require_file(out_dir / "vxc_out.dat").read_text(
        encoding="utf-8"
    ).splitlines()
    expected_vxc_header = [str(bz["n_scf"]), "1", "44"]
    if len(vxc_summary) < 3 or [line.strip() for line in vxc_summary[:3]] != expected_vxc_header:
        fail(
            "vxc_out.dat: expected header "
            f"{expected_vxc_header[0]} k-points, 1 spin, 44 bands"
        )
    shared.require_file(out_dir / "Vxc_R_spin0.csr")
    vxc_matrices, vxc_matrix_summaries = shared.indexed_vxc_files(
        out_dir, int(bz["n_scf"]), int(band["n_basis"])
    )

    sparse = {
        "hamiltonian": shared.require_any(
            out_dir,
            ["data-HR-sparse_SPIN0.csr", "hrs1_nao.csr", "*HR*sparse*SPIN0*.csr"],
            "real-space Hamiltonian matrix",
        ),
        "overlap": shared.require_any(
            out_dir,
            ["data-SR-sparse_SPIN0.csr", "srs1_nao.csr", "*SR*sparse*SPIN0*.csr"],
            "real-space overlap matrix",
        ),
        "position": shared.require_any(
            out_dir,
            ["rr.csr", "data-rR-sparse_SPIN0.csr", "*rR*sparse*SPIN0*.csr"],
            "real-space position matrix",
        ),
    }
    stale_basis = [
        path.name
        for path in (
            run_dir / "basis_wfc_out",
            run_dir / "basis_aux_out",
            run_dir / "basis_out",
        )
        if path.exists()
    ]
    if stale_basis:
        fail(f"reader-v0 run unexpectedly contains basis sidecars: {stale_basis}")

    return {
        "status": "PASS",
        "reader_version": 0,
        "sampling_mode": "full_bz",
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
        args.report.write_text(rendered, encoding="ascii")
    print(rendered, end="")


if __name__ == "__main__":
    main()
