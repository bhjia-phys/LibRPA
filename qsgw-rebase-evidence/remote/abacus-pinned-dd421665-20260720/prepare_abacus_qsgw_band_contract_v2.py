#!/usr/bin/env python3
"""Extend a frozen QSGW SCF contract with exact LibRPA band-reader files."""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

from validate_abacus_si_band_nscf_output_v1 import require_file, sha256_file


CONTRACT_MAGIC = "# librpa-qsgw-input-contract-v1"
BAND_ROLES = {
    "band_mf0_eigenvalues",
    "band_mf0_wavefunctions",
    "band_kpoints",
    "vxc_band_manifest",
}
METADATA_ORDER = [
    "producer",
    "internal_energy_units",
    "mf0_basis",
    "mf0_gauge",
    "n_spins",
    "n_bands",
    "n_aos",
    "n_scf_kpoints",
    "n_headwing_kpoints",
    "n_band_kpoints",
    "headwing_grid",
    "headwing_update",
    "hartree_update",
    "band_update",
]


@dataclass(frozen=True)
class ContractRecord:
    role: str
    sha256: str
    file: str


@dataclass(frozen=True)
class ParsedContract:
    metadata: dict[str, str]
    records: list[ContractRecord]


@dataclass(frozen=True)
class BandHeader:
    n_basis: int
    n_states: int
    n_spins: int
    kpoints: list[tuple[float, float, float]]


def valid_sha256(value: str) -> bool:
    return bool(re.fullmatch(r"[0-9a-f]{64}", value))


def finite_float(token: str, source_name: str) -> float:
    try:
        value = float(token)
    except ValueError as error:
        raise ValueError(f"invalid numeric value in {source_name}: {token}") from error
    if not math.isfinite(value):
        raise ValueError(f"non-finite numeric value in {source_name}: {token}")
    return value


def parse_contract_text(text: str, source_name: str) -> ParsedContract:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines or lines[0] != CONTRACT_MAGIC:
        raise ValueError(f"invalid QSGW contract magic in {source_name}")
    metadata: dict[str, str] = {}
    records: list[ContractRecord] = []
    saw_header = False
    for line in lines[1:]:
        fields = line.split()
        if fields == ["role", "sha256", "file"]:
            if saw_header:
                raise ValueError(f"duplicate role table in {source_name}")
            saw_header = True
            continue
        if not saw_header:
            if len(fields) != 2 or fields[0] in metadata:
                raise ValueError(f"malformed contract metadata in {source_name}: {line}")
            metadata[fields[0]] = fields[1]
        else:
            if len(fields) != 3 or not valid_sha256(fields[1]):
                raise ValueError(f"malformed contract record in {source_name}: {line}")
            records.append(ContractRecord(*fields))
    if not saw_header or set(metadata) != set(METADATA_ORDER) or not records:
        raise ValueError(f"incomplete QSGW contract in {source_name}")
    try:
        n_band_kpoints = int(metadata["n_band_kpoints"])
    except ValueError as error:
        raise ValueError(f"invalid band count in {source_name}") from error
    band_records = [record for record in records if record.role in BAND_ROLES]
    if (
        metadata["band_update"] != "off"
        or n_band_kpoints != 0
        or band_records
    ):
        raise ValueError(
            f"input is not a clean disabled SCF contract in {source_name}"
        )
    return ParsedContract(metadata, records)


def parse_band_kpath_info_text(text: str, source_name: str) -> BandHeader:
    lines = [line.split() for line in text.splitlines() if line.strip()]
    if not lines or len(lines[0]) != 4:
        raise ValueError(f"invalid band k-path header in {source_name}")
    try:
        n_basis, n_states, n_spins, n_kpoints = map(int, lines[0])
    except ValueError as error:
        raise ValueError(f"invalid band k-path dimensions in {source_name}") from error
    if min(n_basis, n_states, n_spins, n_kpoints) <= 0:
        raise ValueError(f"non-positive band k-path dimensions in {source_name}")
    if len(lines[1:]) != n_kpoints:
        raise ValueError(f"band k-point count mismatch in {source_name}")
    kpoints: list[tuple[float, float, float]] = []
    for fields in lines[1:]:
        if len(fields) != 3:
            raise ValueError(f"malformed band k-point in {source_name}")
        kpoints.append(tuple(finite_float(value, source_name) for value in fields))
    return BandHeader(n_basis, n_states, n_spins, kpoints)


def validate_record(record: ContractRecord, expected_role: str) -> None:
    if record.role != expected_role or not valid_sha256(record.sha256):
        raise ValueError(f"invalid {expected_role} contract record")
    path = Path(record.file)
    if not record.file or path.is_absolute() or path.name != record.file:
        raise ValueError(f"contract record must name a direct dataset file: {record.file}")


def render_band_contract(
    scf: ParsedContract,
    header: BandHeader,
    band_kpoints: ContractRecord,
    eigenvalues: list[ContractRecord],
    wavefunctions: list[ContractRecord],
    vxc_manifest: ContractRecord,
) -> str:
    metadata = dict(scf.metadata)
    expected_dimensions = (
        int(metadata["n_aos"]),
        int(metadata["n_bands"]),
        int(metadata["n_spins"]),
    )
    actual_dimensions = (header.n_basis, header.n_states, header.n_spins)
    if actual_dimensions != expected_dimensions:
        raise ValueError(
            "band reference dimensions do not match the frozen SCF contract: "
            f"{actual_dimensions} != {expected_dimensions}"
        )
    n_kpoints = len(header.kpoints)
    if len(eigenvalues) != n_kpoints or len(wavefunctions) != n_kpoints:
        raise ValueError("band contract file counts do not match band k-points")
    validate_record(band_kpoints, "band_kpoints")
    validate_record(vxc_manifest, "vxc_band_manifest")
    for index, record in enumerate(eigenvalues, start=1):
        validate_record(record, "band_mf0_eigenvalues")
        if record.file != f"band_KS_eigenvalue_k_{index:05d}.txt":
            raise ValueError("band eigenvalue contract filenames are not exact")
    for index, record in enumerate(wavefunctions, start=1):
        validate_record(record, "band_mf0_wavefunctions")
        if record.file != f"band_KS_eigenvector_k_{index:05d}.txt":
            raise ValueError("band wavefunction contract filenames are not exact")
    if band_kpoints.file != "band_kpath_info":
        raise ValueError("band k-point contract filename is not exact")

    metadata["n_band_kpoints"] = str(n_kpoints)
    metadata["band_update"] = "operator_fourier"
    records = [
        *scf.records,
        band_kpoints,
        *eigenvalues,
        *wavefunctions,
        vxc_manifest,
    ]
    files = [record.file for record in records]
    if len(files) != len(set(files)):
        raise ValueError("QSGW band contract contains duplicate file bindings")
    lines = [CONTRACT_MAGIC]
    lines.extend(f"{key} {metadata[key]}" for key in METADATA_ORDER)
    lines.append("role sha256 file")
    lines.extend(
        f"{record.role} {record.sha256} {record.file}" for record in records
    )
    return "\n".join(lines) + "\n"


def dataset_file(dataset: Path, name: str) -> Path:
    if not name or Path(name).name != name:
        raise ValueError(f"dataset filename must be a basename: {name}")
    return require_file(dataset / name)


def verify_scf_record_hashes(dataset: Path, scf: ParsedContract) -> None:
    for record in scf.records:
        path = dataset_file(dataset, record.file)
        if sha256_file(path) != record.sha256:
            raise ValueError(f"frozen SCF contract file changed: {record.file}")


def validate_band_manifest(
    path: Path,
    header: BandHeader,
    producer: str,
) -> None:
    lines = [line.strip() for line in path.read_text(encoding="ascii").splitlines() if line.strip()]
    if not lines or lines[0] != "# librpa-qsgw-vxc-manifest-v2":
        raise ValueError("invalid band Vxc manifest magic")
    metadata: dict[str, str] = {}
    rows: list[list[str]] = []
    saw_header = False
    for line in lines[1:]:
        fields = line.split()
        if fields == [
            "spin",
            "k_index",
            "kx",
            "ky",
            "kz",
            "rows",
            "columns",
            "sha256",
            "file",
        ]:
            saw_header = True
        elif not saw_header:
            if len(fields) != 2 or fields[0] in metadata:
                raise ValueError("malformed band Vxc manifest metadata")
            metadata[fields[0]] = fields[1]
        else:
            rows.append(fields)
    expected_metadata = {
        "kind": "band",
        "producer": producer,
        "units": "Ry",
        "basis": "state",
        "gauge": "mf0_state",
    }
    if metadata != expected_metadata or len(rows) != len(header.kpoints):
        raise ValueError("band Vxc manifest contract mismatch")
    for index, (fields, expected_kpoint) in enumerate(
        zip(rows, header.kpoints), start=1
    ):
        if len(fields) != 9:
            raise ValueError("malformed band Vxc manifest row")
        spin, k_index, kx, ky, kz, n_rows, n_columns, digest, name = fields
        if (
            int(spin) != 1
            or int(k_index) != index
            or int(n_rows) != header.n_states
            or int(n_columns) != header.n_states
            or name != f"band_vxck{index}_nao.txt"
            or not valid_sha256(digest)
        ):
            raise ValueError("band Vxc manifest row dimensions or names mismatch")
        actual_kpoint = tuple(finite_float(value, str(path)) for value in (kx, ky, kz))
        if max(abs(actual - expected) for actual, expected in zip(actual_kpoint, expected_kpoint)) > 1.0e-10:
            raise ValueError("band Vxc manifest k-point mismatch")
        matrix = dataset_file(path.parent, name)
        if sha256_file(matrix) != digest:
            raise ValueError(f"band Vxc manifest hash mismatch: {name}")


def prepare(
    dataset: Path,
    scf_contract_name: str,
    band_kpath_name: str,
    band_manifest_name: str,
    output_contract_name: str,
    output_summary_name: str,
) -> dict[str, object]:
    dataset = dataset.resolve(strict=True)
    if not dataset.is_dir():
        raise ValueError(f"dataset is not a directory: {dataset}")
    scf_path = dataset_file(dataset, scf_contract_name)
    scf = parse_contract_text(scf_path.read_text(encoding="ascii"), str(scf_path))
    verify_scf_record_hashes(dataset, scf)
    kpath_path = dataset_file(dataset, band_kpath_name)
    header = parse_band_kpath_info_text(
        kpath_path.read_text(encoding="ascii"), str(kpath_path)
    )
    manifest_path = dataset_file(dataset, band_manifest_name)
    validate_band_manifest(manifest_path, header, scf.metadata["producer"])

    eigenvalue_paths = [
        dataset_file(dataset, f"band_KS_eigenvalue_k_{index:05d}.txt")
        for index in range(1, len(header.kpoints) + 1)
    ]
    wavefunction_paths = [
        dataset_file(dataset, f"band_KS_eigenvector_k_{index:05d}.txt")
        for index in range(1, len(header.kpoints) + 1)
    ]
    expected_eigenvalue_names = {path.name for path in eigenvalue_paths}
    expected_wavefunction_names = {path.name for path in wavefunction_paths}
    if {path.name for path in dataset.glob("band_KS_eigenvalue_k_*.txt")} != expected_eigenvalue_names:
        raise ValueError("band eigenvalue file set is not exact")
    if {path.name for path in dataset.glob("band_KS_eigenvector_k_*.txt")} != expected_wavefunction_names:
        raise ValueError("band wavefunction file set is not exact")

    rendered = render_band_contract(
        scf,
        header,
        ContractRecord("band_kpoints", sha256_file(kpath_path), kpath_path.name),
        [
            ContractRecord("band_mf0_eigenvalues", sha256_file(path), path.name)
            for path in eigenvalue_paths
        ],
        [
            ContractRecord("band_mf0_wavefunctions", sha256_file(path), path.name)
            for path in wavefunction_paths
        ],
        ContractRecord(
            "vxc_band_manifest", sha256_file(manifest_path), manifest_path.name
        ),
    )
    output_contract = dataset / output_contract_name
    output_summary = dataset / output_summary_name
    if output_contract.exists() or output_summary.exists():
        raise ValueError("refusing to overwrite QSGW band contract outputs")
    output_contract.write_text(rendered, encoding="ascii")
    summary = {
        "schema": "abacus-qsgw-band-contract-v2",
        "status": "PASS",
        "producer": scf.metadata["producer"],
        "n_basis": header.n_basis,
        "n_states": header.n_states,
        "n_spins": header.n_spins,
        "n_band_kpoints": len(header.kpoints),
        "band_update": "operator_fourier",
        "band_vxc_basis": "state",
        "band_vxc_gauge": "mf0_state",
        "scf_contract_file": scf_path.name,
        "scf_contract_sha256": sha256_file(scf_path),
        "band_contract_file": output_contract.name,
        "band_contract_sha256": sha256_file(output_contract),
        "band_manifest_sha256": sha256_file(manifest_path),
    }
    output_summary.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="ascii"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--scf-contract", default="qsgw_input.contract")
    parser.add_argument("--band-kpath", default="band_kpath_info")
    parser.add_argument("--band-manifest", default="qsgw_vxc_band.manifest")
    parser.add_argument("--output-contract", default="qsgw_band_input.contract")
    parser.add_argument(
        "--output-summary", default="qsgw_band_input_contract.summary.json"
    )
    args = parser.parse_args()
    for name in (
        args.scf_contract,
        args.band_kpath,
        args.band_manifest,
        args.output_contract,
        args.output_summary,
    ):
        if not name or Path(name).name != name:
            raise ValueError(f"contract filename must be a basename: {name}")
    summary = prepare(
        args.dataset,
        args.scf_contract,
        args.band_kpath,
        args.band_manifest,
        args.output_contract,
        args.output_summary,
    )
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
