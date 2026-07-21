#!/usr/bin/env python3
"""Generate a strict state-basis QSGW contract for a k444 full-BZ dataset."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import prepare_abacus_qsgw_ibz_contract_v3 as shared


def periodic_key(values: tuple[float, float, float]) -> tuple[float, float, float]:
    wrapped = []
    for value in values:
        reduced = value - math.floor(value)
        if math.isclose(reduced, 1.0, rel_tol=0.0, abs_tol=1.0e-12):
            reduced = 0.0
        wrapped.append(round(reduced, 12))
    return tuple(wrapped)


def read_full_bz(
    path: Path, n_kpoints: int
) -> tuple[tuple[int, int, int], list[tuple[float, float, float]], list[int]]:
    rows = [
        line.split()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(rows) < 3:
        raise ValueError(f"incomplete BZ sampling file: {path}")
    grid = tuple(map(int, rows[0]))
    if grid != (4, 4, 4):
        raise ValueError(f"expected exact k444 BZ grid, got {grid}: {path}")
    full_count = math.prod(grid)
    counts = tuple(map(int, rows[1]))
    if counts != (full_count, full_count) or n_kpoints != full_count:
        raise ValueError(
            f"full-BZ counts {counts} and band count {n_kpoints} must all equal "
            f"{full_count}: {path}"
        )
    if len(rows[2:]) != full_count:
        raise ValueError(f"BZ row count does not match full k444 grid: {path}")

    expected_weight = 1.0 / full_count
    kpoints: list[tuple[float, float, float]] = []
    periodic_keys: set[tuple[float, float, float]] = set()
    labels: set[int] = set()
    representatives: set[int] = set()
    for index, fields in enumerate(rows[2:], start=1):
        if len(fields) != 10 or int(fields[0]) != index:
            raise ValueError(f"malformed BZ row {index}: {path}")
        weight = float(fields[1])
        if not math.isfinite(weight) or not math.isclose(
            weight, expected_weight, rel_tol=0.0, abs_tol=1.0e-12
        ):
            raise ValueError(f"BZ row {index} does not have uniform 1/64 weight: {path}")
        vectors = tuple(map(float, fields[2:8]))
        if not all(math.isfinite(value) for value in vectors):
            raise ValueError(f"BZ row {index} has a non-finite vector: {path}")
        key = periodic_key(vectors[:3])
        if key in periodic_keys:
            raise ValueError(f"periodic k-points are not unique: {path}")
        periodic_keys.add(key)
        labels.add(int(fields[8]))
        representatives.add(int(fields[9]))
        kpoints.append(vectors[:3])
    expected_indices = set(range(1, full_count + 1))
    if labels != expected_indices or representatives != expected_indices:
        raise ValueError(f"full-BZ labels/representatives are not bijective: {path}")
    return grid, kpoints, [1] * full_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--use-shrink-abfs", action="store_true")
    parser.add_argument(
        "--output-vxc-manifest", default="qsgw_vxc_scf.manifest", type=Path
    )
    parser.add_argument("--output-contract", default="qsgw_input.contract", type=Path)
    parser.add_argument(
        "--output-summary", default="qsgw_input_contract.summary.json", type=Path
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = args.dataset.resolve(strict=True)
    outputs = [
        dataset / args.output_vxc_manifest,
        dataset / args.output_contract,
        dataset / args.output_summary,
    ]
    for output in outputs:
        if output.parent.resolve() != dataset:
            raise ValueError(f"output must be directly inside the dataset: {output}")
        if output.exists():
            raise ValueError(f"refusing to overwrite output: {output}")

    n_kpoints, n_spins, n_bands, n_aos = shared.read_band_dimensions(
        shared.require_file(dataset / "band_out")
    )
    shared.validate_basis_dimension(
        shared.require_file(dataset / "basis_wfc_out"), n_aos
    )
    grid, kpoints, multiplicities = read_full_bz(
        shared.require_file(dataset / "bz_sampling_out"), n_kpoints
    )
    wavefunctions = shared.indexed_files(
        dataset, r"KS_eigenvector_(\d+)\.dat", list(range(n_kpoints))
    )
    vxc_matrices = shared.write_vxc_manifest(
        dataset, outputs[0], kpoints, n_spins, n_bands
    )
    roles = shared.write_contract(
        dataset,
        outputs[1],
        outputs[0],
        wavefunctions,
        n_spins,
        n_bands,
        n_aos,
        n_kpoints,
        args.use_shrink_abfs,
    )
    summary = {
        "producer": "abacus",
        "sampling_mode": "full_bz",
        "grid": list(grid),
        "full_bz_kpoints": math.prod(grid),
        "n_scf_kpoints": n_kpoints,
        "kstar_multiplicities": multiplicities,
        "kstar_coverage": sum(multiplicities),
        "uniform_kpoint_weight": 1.0 / n_kpoints,
        "n_spins": n_spins,
        "n_bands": n_bands,
        "n_aos": n_aos,
        "vxc_matrix_count": len(vxc_matrices),
        "vxc_filename_schema": "vxck<one-based-k-index>_nao.txt",
        "vxc_source_format": "abacus_native_comment_row_upper_triangle",
        "vxc_matrix_basis": "ks_state",
        "vxc_matrix_gauge": "mf0_state",
        "vxc_basis_transform": "none",
        "reader_static_count": sum(role == "reader_static" for role, _ in roles),
        "contract_file_count": len(roles),
        "use_shrink_abfs": args.use_shrink_abfs,
        "headwing_update": "none",
        "hartree_update": "off",
        "band_update": "off",
        "vxc_manifest_sha256": shared.sha256_file(outputs[0]),
        "input_contract_sha256": shared.sha256_file(outputs[1]),
    }
    outputs[2].write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="ascii"
    )
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
