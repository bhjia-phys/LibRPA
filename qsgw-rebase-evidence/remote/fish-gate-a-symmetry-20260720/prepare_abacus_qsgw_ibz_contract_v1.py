#!/usr/bin/env python3
"""Generate a strict QSGW contract for a symmetry-reduced ABACUS dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path


VXC_MAGIC = "# librpa-qsgw-vxc-manifest-v2"
CONTRACT_MAGIC = "# librpa-qsgw-input-contract-v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_file(path: Path) -> Path:
    if not path.is_file():
        raise ValueError(f"required input file is missing: {path}")
    return path


def relative_name(dataset: Path, path: Path) -> str:
    return path.relative_to(dataset).as_posix()


def read_band_dimensions(path: Path) -> tuple[int, int, int, int]:
    fields = path.read_text(encoding="utf-8").split()
    if len(fields) < 4:
        raise ValueError(f"incomplete band_out header: {path}")
    dimensions = tuple(map(int, fields[:4]))
    if min(dimensions) <= 0:
        raise ValueError(f"invalid band_out dimensions: {path}")
    return dimensions


def validate_basis_dimension(path: Path, n_aos: int) -> None:
    fields = path.read_text(encoding="utf-8").splitlines()[0].split()
    if len(fields) < 2 or int(fields[1]) != n_aos:
        raise ValueError(f"basis_wfc_out AO dimension does not match band_out: {path}")


def read_reduced_bz(
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
    if len(grid) != 3 or min(grid) <= 0:
        raise ValueError(f"invalid BZ grid: {path}")
    full_count = math.prod(grid)
    counts = tuple(map(int, rows[1]))
    if counts != (n_kpoints, n_kpoints):
        raise ValueError(
            f"BZ representative counts {counts} do not match band_out ({n_kpoints}): {path}"
        )
    if n_kpoints >= full_count:
        raise ValueError(f"BZ sampling is not symmetry reduced: {path}")
    if len(rows[2:]) != n_kpoints:
        raise ValueError(f"BZ row count does not match band_out: {path}")

    kpoints: list[tuple[float, float, float]] = []
    multiplicities: list[int] = []
    weight_sum = 0.0
    for index, fields in enumerate(rows[2:], start=1):
        if len(fields) != 10 or int(fields[0]) != index:
            raise ValueError(f"malformed BZ row {index}: {path}")
        weight = float(fields[1])
        multiplicity = round(weight * full_count)
        if multiplicity <= 0 or abs(weight * full_count - multiplicity) > 1.0e-8:
            raise ValueError(f"nonintegral k-star weight on BZ row {index}: {path}")
        weight_sum += weight
        multiplicities.append(multiplicity)
        kpoints.append(tuple(map(float, fields[2:5])))
    if abs(weight_sum - 1.0) > 1.0e-10:
        raise ValueError(f"BZ weights do not sum to one ({weight_sum:.17g}): {path}")
    if sum(multiplicities) != full_count:
        raise ValueError(f"k-star multiplicities do not cover the full BZ: {path}")
    return grid, kpoints, multiplicities


def indexed_files(
    dataset: Path, pattern: str, expected_indices: list[int]
) -> list[Path]:
    regex = re.compile(pattern)
    indexed: dict[int, Path] = {}
    for path in dataset.iterdir():
        match = regex.fullmatch(path.name)
        if match:
            indexed[int(match.group(1))] = require_file(path)
    if sorted(indexed) != expected_indices:
        raise ValueError(
            f"expected indices {expected_indices} for {pattern}, got {sorted(indexed)}"
        )
    return [indexed[index] for index in expected_indices]


def exact_group(dataset: Path, pattern: str) -> list[Path]:
    regex = re.compile(pattern)
    result = sorted(
        (require_file(path) for path in dataset.iterdir() if regex.fullmatch(path.name)),
        key=lambda path: path.name,
    )
    if not result:
        raise ValueError(f"empty reader file group for {pattern}")
    return result


def read_abacus_vxc_dimension(path: Path) -> int:
    with path.open("r", encoding="utf-8") as stream:
        first = stream.readline().split()
    if len(first) != 1:
        raise ValueError(f"invalid legacy ABACUS Vxc header: {path}")
    dimension = int(first[0])
    if dimension <= 0:
        raise ValueError(f"invalid legacy ABACUS Vxc dimension: {path}")
    return dimension


def write_vxc_manifest(
    dataset: Path,
    output: Path,
    kpoints: list[tuple[float, float, float]],
    n_spins: int,
    n_aos: int,
) -> list[Path]:
    if n_spins != 1:
        raise ValueError("the ABACUS QSGW IBZ producer contract requires one spin channel")
    matrices = indexed_files(
        dataset, r"vxcs1k(\d+)_nao\.txt", list(range(1, len(kpoints) + 1))
    )
    lines = [
        VXC_MAGIC,
        "kind scf",
        "producer abacus",
        "units Ry",
        "basis nao",
        "gauge ao_bloch",
        "spin k_index kx ky kz rows columns sha256 file",
    ]
    for k_index, (kpoint, matrix) in enumerate(zip(kpoints, matrices), start=1):
        if read_abacus_vxc_dimension(matrix) != n_aos:
            raise ValueError(f"Vxc AO dimension does not match band_out: {matrix}")
        lines.append(
            "1 {index:d} {kx:.17g} {ky:.17g} {kz:.17g} "
            "{dim:d} {dim:d} {sha} {name}".format(
                index=k_index,
                kx=kpoint[0],
                ky=kpoint[1],
                kz=kpoint[2],
                dim=n_aos,
                sha=sha256_file(matrix),
                name=matrix.name,
            )
        )
    output.write_text("\n".join(lines) + "\n", encoding="ascii")
    return matrices


def reader_static_files(dataset: Path, use_shrink_abfs: bool) -> list[Path]:
    result = [
        require_file(dataset / "stru_out"),
        require_file(dataset / "basis_wfc_out"),
        require_file(dataset / "basis_aux_out"),
    ]
    result.extend(exact_group(dataset, r"Cs_data_\d+\.txt"))
    if use_shrink_abfs:
        result.append(require_file(dataset / "basis_aux_shrink_out"))
        result.extend(exact_group(dataset, r"Cs_shrinked_data_\d+\.txt"))
        result.extend(exact_group(dataset, r"shrink_sinvS_\d+\.txt"))
    result.extend(exact_group(dataset, r"coulomb_mat_\d+\.txt"))
    result.extend(exact_group(dataset, r"coulomb_cut_\d+\.txt"))
    return result


def write_contract(
    dataset: Path,
    output: Path,
    vxc_manifest: Path,
    wavefunctions: list[Path],
    n_spins: int,
    n_bands: int,
    n_aos: int,
    n_kpoints: int,
    use_shrink_abfs: bool,
) -> list[tuple[str, Path]]:
    roles: list[tuple[str, Path]] = [
        ("mf0_eigenvalues", require_file(dataset / "band_out")),
    ]
    roles.extend(("mf0_wavefunctions", path) for path in wavefunctions)
    roles.extend(
        [
            ("scf_kpoints", require_file(dataset / "bz_sampling_out")),
            ("vxc_scf_manifest", require_file(vxc_manifest)),
        ]
    )
    roles.extend(
        ("reader_static", path)
        for path in reader_static_files(dataset, use_shrink_abfs)
    )
    lines = [
        CONTRACT_MAGIC,
        "producer abacus",
        "internal_energy_units hartree",
        "mf0_basis state_coefficients_in_nao",
        "mf0_gauge producer_state",
        f"n_spins {n_spins}",
        f"n_bands {n_bands}",
        f"n_aos {n_aos}",
        f"n_scf_kpoints {n_kpoints}",
        "n_headwing_kpoints 0",
        "n_band_kpoints 0",
        "headwing_grid disabled",
        "headwing_update none",
        "hartree_update off",
        "band_update off",
        "role sha256 file",
    ]
    lines.extend(
        f"{role} {sha256_file(path)} {relative_name(dataset, path)}"
        for role, path in roles
    )
    output.write_text("\n".join(lines) + "\n", encoding="ascii")
    return roles


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

    n_kpoints, n_spins, n_bands, n_aos = read_band_dimensions(
        require_file(dataset / "band_out")
    )
    validate_basis_dimension(require_file(dataset / "basis_wfc_out"), n_aos)
    grid, kpoints, multiplicities = read_reduced_bz(
        require_file(dataset / "bz_sampling_out"), n_kpoints
    )
    wavefunctions = indexed_files(
        dataset, r"KS_eigenvector_(\d+)\.dat", list(range(n_kpoints))
    )
    vxc_matrices = write_vxc_manifest(
        dataset, outputs[0], kpoints, n_spins, n_aos
    )
    roles = write_contract(
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
        "grid": list(grid),
        "full_bz_kpoints": math.prod(grid),
        "n_scf_kpoints": n_kpoints,
        "kstar_multiplicities": multiplicities,
        "kstar_coverage": sum(multiplicities),
        "n_spins": n_spins,
        "n_bands": n_bands,
        "n_aos": n_aos,
        "vxc_matrix_count": len(vxc_matrices),
        "reader_static_count": sum(role == "reader_static" for role, _ in roles),
        "contract_file_count": len(roles),
        "use_shrink_abfs": args.use_shrink_abfs,
        "headwing_update": "none",
        "hartree_update": "off",
        "band_update": "off",
        "vxc_manifest_sha256": sha256_file(outputs[0]),
        "input_contract_sha256": sha256_file(outputs[1]),
    }
    outputs[2].write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="ascii"
    )
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
