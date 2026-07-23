#!/usr/bin/env python3
"""Merge frozen same-grid head-only and band QSGW input contracts."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path


HEADER = "# librpa-qsgw-input-contract-v1"
IDENTITY_KEYS = (
    "producer",
    "internal_energy_units",
    "mf0_basis",
    "mf0_gauge",
    "n_spins",
    "n_bands",
    "n_aos",
    "n_scf_kpoints",
)


class ContractError(ValueError):
    pass


@dataclass(frozen=True)
class Contract:
    metadata_order: tuple[str, ...]
    metadata: dict[str, str]
    records: tuple[tuple[str, str, str], ...]


def parse_contract(path: Path) -> Contract:
    lines = [
        line.strip()
        for line in path.read_text(encoding="ascii").splitlines()
        if line.strip()
    ]
    if not lines or lines[0] != HEADER:
        raise ContractError(f"invalid QSGW contract header: {path}")
    metadata_order: list[str] = []
    metadata: dict[str, str] = {}
    records: list[tuple[str, str, str]] = []
    in_records = False
    for line_number, line in enumerate(lines[1:], start=2):
        fields = line.split()
        if fields == ["role", "sha256", "file"]:
            if in_records:
                raise ContractError(f"duplicate record header in {path}")
            in_records = True
            continue
        if not in_records:
            if len(fields) != 2 or fields[0] in metadata:
                raise ContractError(
                    f"invalid metadata at {path}:{line_number}"
                )
            metadata_order.append(fields[0])
            metadata[fields[0]] = fields[1]
        else:
            if len(fields) != 3 or len(fields[1]) != 64:
                raise ContractError(
                    f"invalid file record at {path}:{line_number}"
                )
            records.append((fields[0], fields[1], fields[2]))
    if not in_records or not records:
        raise ContractError(f"contract has no file records: {path}")
    return Contract(tuple(metadata_order), metadata, tuple(records))


def _positive_metadata_int(contract: Contract, key: str) -> int:
    try:
        value = int(contract.metadata[key])
    except (KeyError, ValueError) as error:
        raise ContractError(f"invalid {key} metadata") from error
    if value <= 0:
        raise ContractError(f"{key} must be positive")
    return value


def merge_contracts(head: Contract, band: Contract) -> str:
    for key in IDENTITY_KEYS:
        if head.metadata.get(key) != band.metadata.get(key):
            raise ContractError(f"head and band contracts differ in {key}")
    if (
        head.metadata.get("headwing_grid") != "scf"
        or head.metadata.get("headwing_update") != "fixed_basis_rotation"
        or head.metadata.get("band_update") != "off"
        or head.metadata.get("hartree_update") != "off"
        or head.metadata.get("n_band_kpoints") != "0"
    ):
        raise ContractError("head contract is not same-grid head-only")
    if (
        band.metadata.get("headwing_grid") != "disabled"
        or band.metadata.get("headwing_update") != "none"
        or band.metadata.get("band_update") != "fixed_basis_rotation"
        or band.metadata.get("hartree_update") != "off"
        or band.metadata.get("n_headwing_kpoints") != "0"
    ):
        raise ContractError(
            "band contract is not head-disabled fixed-basis rotation"
        )
    _positive_metadata_int(head, "n_headwing_kpoints")
    _positive_metadata_int(band, "n_band_kpoints")

    metadata = dict(band.metadata)
    metadata["n_headwing_kpoints"] = head.metadata["n_headwing_kpoints"]
    metadata["headwing_grid"] = head.metadata["headwing_grid"]
    metadata["headwing_update"] = head.metadata["headwing_update"]

    records: list[tuple[str, str, str]] = []
    by_role_path: dict[tuple[str, str], str] = {}
    for role, digest, relative in band.records + head.records:
        key = (role, relative)
        previous = by_role_path.get(key)
        if previous is not None:
            if previous != digest:
                raise ContractError(
                    f"conflicting hashes for {role} {relative}"
                )
            continue
        by_role_path[key] = digest
        records.append((role, digest, relative))
    if not any(role == "velocity_mf0" for role, _digest, _path in records):
        raise ContractError("merged contract has no velocity_mf0 records")
    if not any(
        role == "band_mf0_eigenvalues"
        for role, _digest, _path in records
    ):
        raise ContractError("merged contract has no band eigenvalue records")

    output = [HEADER]
    for key in band.metadata_order:
        output.append(f"{key} {metadata[key]}")
    output.append("role sha256 file")
    output.extend(
        f"{role} {digest} {relative}"
        for role, digest, relative in records
    )
    return "\n".join(output) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--head-contract", type=Path, required=True)
    parser.add_argument("--band-contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        merged = merge_contracts(
            parse_contract(args.head_contract),
            parse_contract(args.band_contract),
        )
    except (ContractError, OSError) as error:
        parser.error(str(error))
    args.output.write_text(merged, encoding="ascii")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
