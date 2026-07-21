#!/usr/bin/env python3
"""Extend a verified QSGW input contract with exact Hartree reader roles."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath


MAGIC = "# librpa-qsgw-input-contract-v1"
ROLE_HEADER = "role sha256 file"
SHA256 = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True)
class ContractRecord:
    role: str
    sha256: str
    file: str


@dataclass(frozen=True)
class ContractDocument:
    metadata: tuple[tuple[str, str], ...]
    records: tuple[ContractRecord, ...]

    def value(self, key: str) -> str:
        values = [value for name, value in self.metadata if name == key]
        if len(values) != 1:
            raise ValueError(f"contract must contain exactly one {key}")
        return values[0]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_relative_file(value: str) -> bool:
    if not value or "\\" in value:
        return False
    path = PurePosixPath(value)
    return (
        bool(path.parts)
        and not path.is_absolute()
        and all(part not in ("", ".", "..") for part in path.parts)
    )


def parse_contract(text: str) -> ContractDocument:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines or lines[0] != MAGIC:
        raise ValueError("unsupported QSGW input-contract magic")
    try:
        role_header = lines.index(ROLE_HEADER)
    except ValueError as error:
        raise ValueError("QSGW input contract lacks the role header") from error
    if role_header < 2 or lines.count(ROLE_HEADER) != 1:
        raise ValueError("QSGW input contract has an invalid role-header layout")

    metadata = []
    metadata_names = set()
    for line in lines[1:role_header]:
        fields = line.split()
        if len(fields) != 2 or fields[0] in metadata_names:
            raise ValueError(f"invalid or duplicate contract metadata: {line}")
        metadata.append((fields[0], fields[1]))
        metadata_names.add(fields[0])

    records = []
    record_keys = set()
    for line in lines[role_header + 1:]:
        fields = line.split()
        if (
            len(fields) != 3
            or not SHA256.fullmatch(fields[1])
            or not safe_relative_file(fields[2])
        ):
            raise ValueError(f"invalid contract file record: {line}")
        key = (fields[0], fields[2])
        if key in record_keys:
            raise ValueError(f"duplicate contract role/file record: {line}")
        record_keys.add(key)
        records.append(ContractRecord(*fields))

    document = ContractDocument(tuple(metadata), tuple(records))
    for key in (
        "producer",
        "n_spins",
        "n_bands",
        "n_aos",
        "n_scf_kpoints",
        "n_band_kpoints",
        "hartree_update",
        "band_update",
    ):
        document.value(key)
    return document


def reader_static_hashes(document: ContractDocument) -> dict[str, str]:
    result = {}
    for record in document.records:
        if record.role != "reader_static":
            continue
        if record.file in result and result[record.file] != record.sha256:
            raise ValueError("reader_static contains conflicting file hashes")
        result[record.file] = record.sha256
    if not result:
        raise ValueError("base QSGW input contract has no reader_static files")
    return result


def _matching_files(files: dict[str, str], pattern: str) -> list[str]:
    expression = re.compile(pattern)
    return sorted(
        name
        for name in files
        if PurePosixPath(name).name == name and expression.fullmatch(name)
    )


def select_hartree_records(
    document: ContractDocument,
    *,
    use_shrink_abfs: bool,
    hartree_coulomb: str,
) -> tuple[ContractRecord, ...]:
    if document.value("hartree_update") != "off":
        raise ValueError("base QSGW input contract must have hartree_update off")
    if any(record.role.startswith("hartree_") for record in document.records):
        raise ValueError("base QSGW input contract already contains Hartree roles")
    if hartree_coulomb not in ("full", "truncated"):
        raise ValueError("hartree_coulomb must be full or truncated")
    static = reader_static_hashes(document)
    ri_pattern = (
        r"Cs_shrinked_data_\d+\.txt"
        if use_shrink_abfs
        else r"Cs_data_\d+\.txt"
    )
    coulomb_pattern = (
        r"coulomb_mat_\d+\.txt"
        if hartree_coulomb == "full"
        else r"coulomb_cut_\d+\.txt"
    )
    ri_files = _matching_files(static, ri_pattern)
    coulomb_files = _matching_files(static, coulomb_pattern)
    if not ri_files or not coulomb_files:
        raise ValueError("base contract lacks the requested Hartree RI or Coulomb files")

    if use_shrink_abfs:
        explicit_auxiliary = next(
            (
                name
                for name in (
                    "basis_aux_shrink_out",
                    "basis_out_shrink",
                    "basis_out.shrink_backup",
                )
                if name in static
            ),
            None,
        )
        auxiliary_files = [explicit_auxiliary] if explicit_auxiliary else ri_files
    elif "basis_wfc_out" in static and "basis_aux_out" in static:
        auxiliary_files = ["basis_aux_out"]
    elif "basis_out" in static:
        auxiliary_files = ["basis_out"]
    else:
        auxiliary_files = ri_files

    additions = []
    additions.extend(
        ContractRecord("hartree_ri_coefficients", static[name], name)
        for name in ri_files
    )
    additions.extend(
        ContractRecord("hartree_coulomb", static[name], name)
        for name in coulomb_files
    )
    additions.extend(
        ContractRecord("hartree_aux_basis", static[name], name)
        for name in auxiliary_files
    )
    return tuple(additions)


def extend_contract(
    document: ContractDocument,
    additions: tuple[ContractRecord, ...],
) -> str:
    if document.value("hartree_update") != "off":
        raise ValueError("base QSGW input contract must have hartree_update off")
    if any(record.role.startswith("hartree_") for record in document.records):
        raise ValueError("base QSGW input contract already contains Hartree roles")
    required_roles = {
        "hartree_ri_coefficients",
        "hartree_coulomb",
        "hartree_aux_basis",
    }
    if {record.role for record in additions} != required_roles:
        raise ValueError("Hartree additions do not contain the exact required roles")
    static = reader_static_hashes(document)
    for record in additions:
        if static.get(record.file) != record.sha256:
            raise ValueError(
                f"Hartree role is not SHA-identical to reader_static: {record.file}"
            )

    metadata = [
        (key, "delta_density" if key == "hartree_update" else value)
        for key, value in document.metadata
    ]
    lines = [MAGIC]
    lines.extend(f"{key} {value}" for key, value in metadata)
    lines.append(ROLE_HEADER)
    lines.extend(
        f"{record.role} {record.sha256} {record.file}"
        for record in (*document.records, *additions)
    )
    return "\n".join(lines) + "\n"


def validate_base_files(dataset: Path, document: ContractDocument) -> None:
    dataset_resolved = dataset.resolve(strict=True)
    observed = {}
    for record in document.records:
        if record.file in observed:
            if observed[record.file] != record.sha256:
                raise ValueError(f"contract has conflicting hashes for {record.file}")
            continue
        path = dataset / PurePosixPath(record.file)
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"contract input is missing or is a symlink: {record.file}")
        resolved = path.resolve(strict=True)
        try:
            resolved.relative_to(dataset_resolved)
        except ValueError as error:
            raise ValueError(f"contract input escapes the dataset: {record.file}") from error
        digest = sha256_file(resolved)
        if digest != record.sha256:
            raise ValueError(f"contract input SHA256 differs: {record.file}")
        observed[record.file] = digest


def direct_output(dataset: Path, value: Path) -> Path:
    output = value if value.is_absolute() else dataset / value
    if output.parent.resolve() != dataset.resolve():
        raise ValueError(f"output must be directly inside the dataset: {output}")
    if output.exists():
        raise ValueError(f"refusing to overwrite output: {output}")
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("base_contract", type=Path)
    parser.add_argument("--output-contract", required=True, type=Path)
    parser.add_argument("--output-summary", required=True, type=Path)
    parser.add_argument(
        "--hartree-coulomb", required=True, choices=("full", "truncated")
    )
    parser.add_argument("--use-shrink-abfs", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = args.dataset.resolve(strict=True)
    base_contract = (
        args.base_contract
        if args.base_contract.is_absolute()
        else dataset / args.base_contract
    )
    if base_contract.parent.resolve() != dataset or not base_contract.is_file():
        raise ValueError("base contract must be a file directly inside the dataset")
    output_contract = direct_output(dataset, args.output_contract)
    output_summary = direct_output(dataset, args.output_summary)

    document = parse_contract(base_contract.read_text(encoding="ascii"))
    validate_base_files(dataset, document)
    additions = select_hartree_records(
        document,
        use_shrink_abfs=args.use_shrink_abfs,
        hartree_coulomb=args.hartree_coulomb,
    )
    output_contract.write_text(
        extend_contract(document, additions), encoding="ascii"
    )
    summary = {
        "schema": "librpa-qsgw-hartree-contract-extension-v1",
        "base_contract": base_contract.name,
        "base_contract_sha256": sha256_file(base_contract),
        "output_contract": output_contract.name,
        "output_contract_sha256": sha256_file(output_contract),
        "producer": document.value("producer"),
        "n_scf_kpoints": int(document.value("n_scf_kpoints")),
        "n_band_kpoints": int(document.value("n_band_kpoints")),
        "band_update": document.value("band_update"),
        "hartree_update": "delta_density",
        "hartree_coulomb": args.hartree_coulomb,
        "use_shrink_abfs": args.use_shrink_abfs,
        "hartree_ri_file_count": sum(
            record.role == "hartree_ri_coefficients" for record in additions
        ),
        "hartree_coulomb_file_count": sum(
            record.role == "hartree_coulomb" for record in additions
        ),
        "hartree_aux_basis_file_count": sum(
            record.role == "hartree_aux_basis" for record in additions
        ),
    }
    output_summary.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="ascii"
    )
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
