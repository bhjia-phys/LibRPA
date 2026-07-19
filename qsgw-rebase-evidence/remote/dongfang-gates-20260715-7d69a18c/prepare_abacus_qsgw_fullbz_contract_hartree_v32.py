#!/usr/bin/env python3
"""Build a strict, full-BZ ABACUS QSGW input contract.

The source dataset is treated as immutable. Run this script in a fresh
dataset directory containing physical copies of the input files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path


VXC_V1_MAGIC = "# librpa-qsgw-vxc-manifest-v1"
VXC_V2_MAGIC = "# librpa-qsgw-vxc-manifest-v2"
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
    try:
        return path.relative_to(dataset).as_posix()
    except ValueError as error:
        raise ValueError(f"input must be inside the dataset directory: {path}") from error


def read_band_dimensions(path: Path) -> tuple[int, int, int, int]:
    values = path.read_text(encoding="utf-8").split()
    if len(values) < 4:
        raise ValueError(f"incomplete band_out header: {path}")
    n_kpoints, n_spins, n_bands, n_aos = map(int, values[:4])
    if min(n_kpoints, n_spins, n_bands, n_aos) <= 0:
        raise ValueError(f"invalid band_out dimensions: {path}")
    return n_kpoints, n_spins, n_bands, n_aos


def validate_basis_dimension(path: Path, n_aos: int) -> None:
    first_line = path.read_text(encoding="utf-8").splitlines()[0].split()
    if len(first_line) < 2 or int(first_line[1]) != n_aos:
        raise ValueError(f"basis_wfc_out AO dimension does not match band_out: {path}")


def validate_full_bz(
    path: Path, n_kpoints: int
) -> tuple[tuple[int, int, int], list[tuple[float, float, float]]]:
    lines = [line.split() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(lines) < 2:
        raise ValueError(f"incomplete BZ sampling file: {path}")
    grid = tuple(map(int, lines[0]))
    if len(grid) != 3 or math.prod(grid) != n_kpoints:
        raise ValueError(f"BZ grid is not a full {n_kpoints}-point mesh: {path}")
    counts = tuple(map(int, lines[1]))
    if counts != (n_kpoints, n_kpoints):
        raise ValueError(
            f"BZ sampling is reduced (expected {n_kpoints} {n_kpoints}, got {counts}): {path}"
        )
    if len(lines[2:]) != n_kpoints:
        raise ValueError(f"BZ row count does not match band_out: {path}")
    weight_sum = 0.0
    kpoints = []
    for index, fields in enumerate(lines[2:], start=1):
        if len(fields) != 10 or int(fields[0]) != index:
            raise ValueError(f"malformed BZ row {index}: {path}")
        weight_sum += float(fields[1])
        kpoints.append(tuple(map(float, fields[2:5])))
        if (int(fields[-2]), int(fields[-1])) != (index, index):
            raise ValueError(
                f"BZ row {index} is symmetry/time-reversal reduced: {path}"
            )
    if abs(weight_sum - 1.0) > 1.0e-10:
        raise ValueError(f"BZ weights do not sum to one ({weight_sum:.17g}): {path}")
    return grid, kpoints


def periodic_kpoint_distance(
    left: tuple[float, float, float], right: tuple[float, float, float]
) -> float:
    return max(abs((a - b + 0.5) % 1.0 - 0.5) for a, b in zip(left, right))


def validate_same_kpoint_set(
    scf_kpoints: list[tuple[float, float, float]],
    headwing_kpoints: list[tuple[float, float, float]],
    source: Path,
) -> None:
    unmatched = list(headwing_kpoints)
    for target in scf_kpoints:
        matches = [
            index
            for index, candidate in enumerate(unmatched)
            if periodic_kpoint_distance(target, candidate) <= 1.0e-8
        ]
        if len(matches) != 1:
            raise ValueError(
                f"head-wing k-point grid does not map uniquely to the SCF grid: {source}"
            )
        unmatched.pop(matches[0])
    if unmatched:
        raise ValueError(f"head-wing k-point grid has unmatched points: {source}")


def validate_same_grid_headwing(
    dataset: Path,
    n_spins: int,
    n_bands: int,
    n_aos: int,
    n_kpoints: int,
    scf_kpoints: list[tuple[float, float, float]],
) -> list[Path]:
    pyatb = dataset / "pyatb_librpa_df"
    velocity = require_file(pyatb / "velocity_matrix")
    kpath = require_file(pyatb / "k_path_info")
    band = require_file(pyatb / "band_out")

    lines = [
        line.split()
        for line in kpath.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not lines or len(lines[0]) != 4:
        raise ValueError(f"malformed head-wing k_path_info header: {kpath}")
    dimensions = tuple(map(int, lines[0]))
    expected = (n_aos, n_bands, n_spins, n_kpoints)
    if dimensions != expected:
        raise ValueError(
            f"head-wing dimensions {dimensions} do not match SCF {expected}: {kpath}"
        )
    if len(lines[1:]) != n_kpoints:
        raise ValueError(f"head-wing k-point count does not match SCF grid: {kpath}")
    headwing_kpoints = []
    for fields in lines[1:]:
        if len(fields) != 3:
            raise ValueError(f"malformed head-wing k-point row: {kpath}")
        headwing_kpoints.append(tuple(map(float, fields)))
    validate_same_kpoint_set(scf_kpoints, headwing_kpoints, kpath)

    if read_band_dimensions(band) != (n_kpoints, n_spins, n_bands, n_aos):
        raise ValueError(f"head-wing band_out dimensions do not match SCF: {band}")
    wavefunctions = indexed_files(
        pyatb, r"KS_eigenvector_(\d+)\.dat", n_kpoints
    )
    return [kpath, band, *wavefunctions, velocity]


def indexed_files(dataset: Path, pattern: str, expected_count: int) -> list[Path]:
    regex = re.compile(pattern)
    indexed: dict[int, Path] = {}
    for path in dataset.iterdir():
        match = regex.fullmatch(path.name)
        if match:
            indexed[int(match.group(1))] = require_file(path)
    expected = list(range(expected_count))
    if sorted(indexed) != expected:
        raise ValueError(
            f"expected indices 0..{expected_count - 1} for {pattern}, got {sorted(indexed)}"
        )
    return [indexed[index] for index in expected]


def parse_source_vxc_manifest(path: Path) -> tuple[dict[str, str], list[dict[str, object]]]:
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines or lines[0] not in {VXC_V1_MAGIC, VXC_V2_MAGIC}:
        raise ValueError(f"unsupported source Vxc manifest: {path}")
    version = 1 if lines[0] == VXC_V1_MAGIC else 2
    metadata: dict[str, str] = {}
    entries: list[dict[str, object]] = []
    saw_header = False
    for line in lines[1:]:
        fields = line.split()
        if fields[0].lower() == "spin":
            saw_header = True
            continue
        if not saw_header:
            if len(fields) != 2:
                raise ValueError(f"malformed Vxc metadata line: {line}")
            metadata[fields[0].lower()] = fields[1]
            continue
        expected_columns = 6 if version == 1 else 9
        if len(fields) != expected_columns:
            raise ValueError(f"malformed Vxc entry: {line}")
        entry: dict[str, object] = {
            "spin": int(fields[0]),
            "k_index": int(fields[1]),
            "kx": float(fields[2]),
            "ky": float(fields[3]),
            "kz": float(fields[4]),
        }
        if version == 1:
            entry["file"] = fields[5]
        else:
            entry["file"] = fields[8]
        entries.append(entry)
    if not saw_header or not entries:
        raise ValueError(f"incomplete source Vxc manifest: {path}")
    return metadata, entries


def read_vxc_dimensions(path: Path) -> tuple[int, int]:
    rows = columns = None
    with path.open("r", encoding="utf-8") as stream:
        for _ in range(8):
            line = stream.readline()
            if not line:
                break
            fields = line.split()
            if len(fields) == 3 and fields[:2] == ["#", "rows"]:
                rows = int(fields[2])
            elif len(fields) == 3 and fields[:2] == ["#", "columns"]:
                columns = int(fields[2])
            if rows is not None and columns is not None:
                return rows, columns
    raise ValueError(f"Vxc matrix lacks '# rows'/'# columns' headers: {path}")


def build_vxc_manifest(
    dataset: Path,
    source_manifest: Path,
    output_manifest: Path,
    n_spins: int,
    n_kpoints: int,
    n_aos: int,
) -> list[dict[str, object]]:
    metadata, entries = parse_source_vxc_manifest(source_manifest)
    normalized_metadata = {key: value.lower() for key, value in metadata.items()}
    required = {"kind": "scf", "producer": "abacus", "units": "ry", "basis": "nao"}
    for key, value in required.items():
        if normalized_metadata.get(key) != value:
            raise ValueError(f"source Vxc manifest has incompatible {key}: {source_manifest}")
    if len(entries) != n_spins * n_kpoints:
        raise ValueError("source Vxc manifest entry count does not match band_out")

    by_key = {(int(entry["spin"]), int(entry["k_index"])): entry for entry in entries}
    if len(by_key) != len(entries):
        raise ValueError("source Vxc manifest contains duplicate spin/k entries")

    output_entries: list[dict[str, object]] = []
    for spin in range(1, n_spins + 1):
        for k_index in range(1, n_kpoints + 1):
            try:
                source = by_key[(spin, k_index)]
            except KeyError as error:
                raise ValueError(f"source Vxc manifest misses spin={spin}, k={k_index}") from error
            matrix = require_file(dataset / str(source["file"]))
            rows, columns = read_vxc_dimensions(matrix)
            if (rows, columns) != (n_aos, n_aos):
                raise ValueError(
                    f"Vxc dimension {(rows, columns)} != {(n_aos, n_aos)}: {matrix}"
                )
            output_entries.append(
                {
                    **source,
                    "rows": rows,
                    "columns": columns,
                    "sha256": sha256_file(matrix),
                    "file": relative_name(dataset, matrix),
                }
            )

    lines = [
        VXC_V2_MAGIC,
        "kind scf",
        "producer abacus",
        "units Ry",
        "basis nao",
        "gauge ao_bloch",
        "spin k_index kx ky kz rows columns sha256 file",
    ]
    for entry in output_entries:
        lines.append(
            "{spin:d} {k_index:d} {kx:.17g} {ky:.17g} {kz:.17g} "
            "{rows:d} {columns:d} {sha256} {file}".format(**entry)
        )
    output_manifest.write_text("\n".join(lines) + "\n", encoding="ascii")
    return output_entries


def required_static_files(dataset: Path, use_shrink_abfs: bool) -> list[Path]:
    fixed = [
        require_file(dataset / "stru_out"),
        require_file(dataset / "basis_wfc_out"),
        require_file(dataset / "basis_aux_out"),
    ]
    grouped: list[Path] = []
    for glob in ("Cs_data_*.txt", "coulomb_mat_*.txt", "coulomb_cut_*.txt"):
        matches = sorted((require_file(path) for path in dataset.glob(glob)), key=lambda path: path.name)
        if not matches:
            raise ValueError(f"no files match required reader input {glob}")
        grouped.extend(matches)
    if use_shrink_abfs:
        for glob in ("Cs_shrinked_data_*.txt", "shrink_sinvS_*.txt"):
            matches = sorted(
                (require_file(path) for path in dataset.glob(glob)),
                key=lambda path: path.name,
            )
            if not matches:
                raise ValueError(f"no files match required shrink input {glob}")
            grouped.extend(matches)
    return fixed + grouped


def hartree_role_files(
    dataset: Path,
    use_shrink_abfs: bool,
    hartree_coulomb: str,
) -> list[tuple[str, Path]]:
    ri_prefix = "Cs_shrinked_data_" if use_shrink_abfs else "Cs_data_"
    ri_files = sorted(
        (require_file(path) for path in dataset.iterdir()
         if path.name.startswith(ri_prefix)),
        key=lambda path: path.name,
    )
    coulomb_prefix = (
        "coulomb_mat_" if hartree_coulomb == "full" else "coulomb_cut_"
    )
    coulomb_files = sorted(
        (require_file(path) for path in dataset.iterdir()
         if path.name.startswith(coulomb_prefix)),
        key=lambda path: path.name,
    )
    if not ri_files or not coulomb_files:
        raise ValueError("Hartree reader RI or Coulomb file set is empty")

    auxiliary_sources: list[Path]
    if use_shrink_abfs:
        explicit_auxiliary = next(
            (
                require_file(dataset / filename)
                for filename in (
                    "basis_aux_shrink_out",
                    "basis_out_shrink",
                    "basis_out.shrink_backup",
                )
                if (dataset / filename).is_file()
            ),
            None,
        )
        auxiliary_sources = (
            [explicit_auxiliary] if explicit_auxiliary is not None else ri_files
        )
    else:
        auxiliary_sources = [require_file(dataset / "basis_aux_out")]

    role_files: list[tuple[str, Path]] = []
    role_files.extend(("hartree_ri_coefficients", path) for path in ri_files)
    role_files.extend(("hartree_coulomb", path) for path in coulomb_files)
    role_files.extend(("hartree_aux_basis", path) for path in auxiliary_sources)
    return role_files


def build_contract(
    dataset: Path,
    output_contract: Path,
    output_manifest: Path,
    n_spins: int,
    n_bands: int,
    n_aos: int,
    n_kpoints: int,
    wavefunctions: list[Path],
    headwing_files: list[Path],
    headwing_mode: str,
    use_shrink_abfs: bool,
    update_hartree: bool,
    hartree_coulomb: str,
) -> list[dict[str, str]]:
    role_files: list[tuple[str, Path]] = [
        ("mf0_eigenvalues", require_file(dataset / "band_out")),
    ]
    role_files.extend(("mf0_wavefunctions", path) for path in wavefunctions)
    role_files.extend(
        [
            ("scf_kpoints", require_file(dataset / "bz_sampling_out")),
            ("vxc_scf_manifest", require_file(output_manifest)),
        ]
    )
    role_files.extend(
        ("reader_static", path)
        for path in required_static_files(dataset, use_shrink_abfs)
    )
    if update_hartree:
        role_files.extend(
            hartree_role_files(
                dataset, use_shrink_abfs, hartree_coulomb
            )
        )
    if headwing_mode == "scf":
        role_files.extend(("velocity_mf0", path) for path in headwing_files)
    elif headwing_mode == "independent_full":
        if len(headwing_files) < 4:
            raise ValueError("independent head-wing input set is incomplete")
        role_files.append(("headwing_kpoints", headwing_files[0]))
        role_files.append(("headwing_mf0_eigenvalues", headwing_files[1]))
        role_files.extend(
            ("headwing_mf0_wavefunctions", path)
            for path in headwing_files[2:-1]
        )
        role_files.append(("headwing_velocity_mf0", headwing_files[-1]))
    elif headwing_mode != "disabled":
        raise ValueError(f"unsupported head-wing mode: {headwing_mode}")

    records = [
        {
            "role": role,
            "sha256": sha256_file(path),
            "file": relative_name(dataset, path),
        }
        for role, path in role_files
    ]
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
        f"n_headwing_kpoints {n_kpoints if headwing_files else 0}",
        "n_band_kpoints 0",
        f"headwing_grid {headwing_mode}",
        "headwing_update "
        + (
            "fixed_basis_rotation"
            if headwing_mode == "scf"
            else "live_ao_fourier"
            if headwing_mode == "independent_full"
            else "none"
        ),
        "hartree_update delta_density" if update_hartree else "hartree_update off",
        "band_update off",
        "role sha256 file",
    ]
    lines.extend(f"{item['role']} {item['sha256']} {item['file']}" for item in records)
    output_contract.write_text("\n".join(lines) + "\n", encoding="ascii")
    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument(
        "--source-vxc-manifest",
        type=Path,
        default=Path("qsgw_vxc_scf.v1.manifest"),
    )
    parser.add_argument(
        "--output-vxc-manifest",
        type=Path,
        default=Path("qsgw_vxc_scf.manifest"),
    )
    parser.add_argument(
        "--output-contract",
        type=Path,
        default=Path("qsgw_input.contract"),
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("qsgw_input_contract.summary.json"),
    )
    headwing_group = parser.add_mutually_exclusive_group()
    headwing_group.add_argument(
        "--same-grid-headwing",
        action="store_true",
        help="bind the full PyATB SCF-grid head/wing package",
    )
    headwing_group.add_argument(
        "--independent-full-headwing",
        action="store_true",
        help="bind PyATB as an independent full-grid live AO-Fourier package",
    )
    parser.add_argument(
        "--use-shrink-abfs",
        action="store_true",
        help="require and bind the shrink-transformation reader inputs",
    )
    parser.add_argument(
        "--update-hartree",
        action="store_true",
        help="bind full-basis RI, selected Coulomb, and auxiliary-basis inputs for a density-difference Hartree update",
    )
    parser.add_argument(
        "--hartree-coulomb",
        choices=("full", "truncated"),
        default="full",
    )
    parser.add_argument(
        "--hartree-normalization",
        choices=("weighted_occupations", "legacy_extra_inverse_nk"),
        default="weighted_occupations",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = args.dataset.resolve(strict=True)
    source_manifest = require_file(
        args.source_vxc_manifest
        if args.source_vxc_manifest.is_absolute()
        else dataset / args.source_vxc_manifest
    )
    output_manifest = (
        args.output_vxc_manifest
        if args.output_vxc_manifest.is_absolute()
        else dataset / args.output_vxc_manifest
    )
    output_contract = (
        args.output_contract
        if args.output_contract.is_absolute()
        else dataset / args.output_contract
    )
    summary_path = args.summary if args.summary.is_absolute() else dataset / args.summary
    for output in (output_manifest, output_contract, summary_path):
        if output.exists():
            raise ValueError(f"refusing to overwrite existing output: {output}")
        if output.parent.resolve() != dataset:
            raise ValueError(f"output must be directly inside the dataset directory: {output}")

    n_kpoints, n_spins, n_bands, n_aos = read_band_dimensions(
        require_file(dataset / "band_out")
    )
    validate_basis_dimension(require_file(dataset / "basis_wfc_out"), n_aos)
    grid, scf_kpoints = validate_full_bz(
        require_file(dataset / "bz_sampling_out"), n_kpoints
    )
    wavefunctions = indexed_files(dataset, r"KS_eigenvector_(\d+)\.dat", n_kpoints)
    headwing_mode = (
        "independent_full"
        if args.independent_full_headwing
        else "scf"
        if args.same_grid_headwing
        else "disabled"
    )
    headwing_files = (
        validate_same_grid_headwing(
            dataset,
            n_spins,
            n_bands,
            n_aos,
            n_kpoints,
            scf_kpoints,
        )
        if headwing_mode != "disabled"
        else []
    )
    vxc_entries = build_vxc_manifest(
        dataset,
        source_manifest,
        output_manifest,
        n_spins,
        n_kpoints,
        n_aos,
    )
    contract_records = build_contract(
        dataset,
        output_contract,
        output_manifest,
        n_spins,
        n_bands,
        n_aos,
        n_kpoints,
        wavefunctions,
        headwing_files,
        headwing_mode,
        args.use_shrink_abfs,
        args.update_hartree,
        args.hartree_coulomb,
    )
    summary = {
        "producer": "abacus",
        "full_bz_grid": list(grid),
        "n_scf_kpoints": n_kpoints,
        "n_spins": n_spins,
        "n_bands": n_bands,
        "n_aos": n_aos,
        "headwing_grid": headwing_mode,
        "headwing_update": (
            "fixed_basis_rotation"
            if headwing_mode == "scf"
            else "live_ao_fourier"
            if headwing_mode == "independent_full"
            else "none"
        ),
        "headwing_file_count": len(headwing_files),
        "use_shrink_abfs": args.use_shrink_abfs,
        "hartree_update": "delta_density" if args.update_hartree else "off",
        "hartree_coulomb": args.hartree_coulomb,
        "hartree_normalization": args.hartree_normalization,
        "band_update": "off",
        "vxc_manifest": relative_name(dataset, output_manifest),
        "vxc_manifest_sha256": sha256_file(output_manifest),
        "vxc_matrix_count": len(vxc_entries),
        "input_contract": relative_name(dataset, output_contract),
        "input_contract_sha256": sha256_file(output_contract),
        "contract_file_count": len(contract_records),
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="ascii")
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
