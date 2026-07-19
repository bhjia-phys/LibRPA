#!/usr/bin/env python3
"""Activate same-grid live head/wing input in a frozen QSGW contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


SHA256 = re.compile(r"[0-9a-f]{64}")
HEADWING_ROLES = {
    "velocity_mf0",
    "headwing_mf0_eigenvalues",
    "headwing_mf0_wavefunctions",
    "headwing_kpoints",
    "headwing_velocity_mf0",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _ending(line: str) -> str:
    if line.endswith("\r\n"):
        return "\r\n"
    if line.endswith("\n"):
        return "\n"
    return ""


def _resolve_payload(root: Path, relative: str) -> Path:
    if not relative or "\\" in relative:
        raise ValueError(f"unsafe contract payload path {relative!r}")
    path = (root / relative).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as error:
        raise ValueError(
            f"contract payload escapes dataset: {relative}"
        ) from error
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"contract payload is not a regular file: {relative}")
    return path


def activate_same_grid_headwing(
    dataset: Path,
    *,
    expected_source_contract_sha256: str,
    velocity_filename: str = "velocity_matrix",
) -> dict[str, object]:
    dataset = dataset.resolve()
    contract = dataset / "qsgw_input.contract"
    if not dataset.is_dir() or not contract.is_file() or contract.is_symlink():
        raise ValueError("missing regular QSGW input contract")
    if not SHA256.fullmatch(expected_source_contract_sha256):
        raise ValueError("invalid source contract SHA256")
    source_sha = _sha256(contract)
    if source_sha != expected_source_contract_sha256:
        raise ValueError(
            f"source contract SHA256 mismatch: expected "
            f"{expected_source_contract_sha256}, got {source_sha}"
        )

    original = contract.read_text(encoding="ascii")
    lines = original.splitlines(keepends=True)
    if not lines or lines[0].strip() != "# librpa-qsgw-input-contract-v1":
        raise ValueError("unsupported QSGW input contract header")

    metadata: dict[str, tuple[int, str]] = {}
    records: list[tuple[str, str, str]] = []
    for index, line in enumerate(lines[1:], 1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        fields = stripped.split()
        if fields == ["role", "sha256", "file"]:
            continue
        if len(fields) == 3 and SHA256.fullmatch(fields[1]):
            records.append((fields[0], fields[1], fields[2]))
            continue
        if len(fields) != 2 or fields[0] in metadata:
            raise ValueError(
                f"invalid or duplicate contract metadata at line {index + 1}"
            )
        metadata[fields[0]] = (index, fields[1])

    required = {
        "producer",
        "n_scf_kpoints",
        "n_headwing_kpoints",
        "headwing_grid",
        "headwing_update",
    }
    missing = sorted(required - set(metadata))
    if missing:
        raise ValueError(f"contract is missing metadata {missing}")
    n_scf = int(metadata["n_scf_kpoints"][1])
    if n_scf <= 0:
        raise ValueError("invalid SCF k-point count")
    if (
        metadata["n_headwing_kpoints"][1] != "0"
        or metadata["headwing_grid"][1] != "disabled"
        or metadata["headwing_update"][1] != "none"
        or any(role in HEADWING_ROLES for role, _digest, _file in records)
    ):
        raise ValueError("head/wing activation requires a disabled source contract")

    seen_records: set[tuple[str, str]] = set()
    for role, expected, relative in records:
        key = (role, relative)
        if key in seen_records:
            raise ValueError(f"duplicate contract role/file record {key}")
        seen_records.add(key)
        actual = _sha256(_resolve_payload(dataset, relative))
        if actual != expected:
            raise ValueError(
                f"payload SHA256 mismatch for {relative}: "
                f"expected {expected}, got {actual}"
            )

    producer = metadata["producer"][1]
    if producer == "abacus":
        pyatb = Path("pyatb_librpa_df")
        if (dataset / pyatb / "velocity_matrix").is_file():
            velocity_files = [
                (pyatb / "k_path_info").as_posix(),
                (pyatb / "band_out").as_posix(),
                *[
                    (pyatb / f"KS_eigenvector_{kpoint}.dat").as_posix()
                    for kpoint in range(n_scf)
                ],
                (pyatb / "velocity_matrix").as_posix(),
            ]
        else:
            velocity_files = [velocity_filename]
    elif producer == "fhi-aims":
        velocity_files = [
            f"mommat_ks_kpt_{kpoint:06d}.dat"
            for kpoint in range(1, n_scf + 1)
        ]
    else:
        raise ValueError(f"unsupported QSGW producer {producer!r}")
    velocity_hashes = {
        relative: _sha256(_resolve_payload(dataset, relative))
        for relative in velocity_files
    }
    updated = list(lines)
    changes = {
        "n_headwing_kpoints": ("0", str(n_scf)),
        "headwing_grid": ("disabled", "scf"),
        "headwing_update": ("none", "fixed_basis_rotation"),
    }
    changed_lines: list[int] = []
    for key, (before, after) in changes.items():
        index, observed = metadata[key]
        if observed != before:
            raise ValueError(f"unexpected source metadata {key}={observed}")
        updated[index] = f"{key} {after}{_ending(lines[index])}"
        changed_lines.append(index + 1)

    newline = "\r\n" if any(line.endswith("\r\n") for line in lines) else "\n"
    if updated and not _ending(updated[-1]):
        updated[-1] += newline
    for relative in velocity_files:
        updated.append(
            f"velocity_mf0 {velocity_hashes[relative]} {relative}{newline}"
        )
    output = "".join(updated)
    contract.write_text(output, encoding="ascii", newline="")

    return {
        "passed": True,
        "dataset": str(dataset),
        "contract": str(contract),
        "source_contract_sha256": source_sha,
        "new_contract_sha256": _sha256(contract),
        "changed_metadata": {
            key: [before, after]
            for key, (before, after) in sorted(changes.items())
        },
        "changed_line_numbers": sorted(changed_lines),
        "added_roles": ["velocity_mf0"],
        "velocity_files": velocity_files,
        "velocity_sha256": velocity_hashes,
        "n_headwing_kpoints": n_scf,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--expected-source-contract-sha256", required=True)
    parser.add_argument("--velocity-filename", default="velocity_matrix")
    args = parser.parse_args()
    try:
        report = activate_same_grid_headwing(
            args.dataset,
            expected_source_contract_sha256=(
                args.expected_source_contract_sha256
            ),
            velocity_filename=args.velocity_filename,
        )
        status = 0
    except (OSError, ValueError) as error:
        report = {"passed": False, "error": str(error)}
        status = 2
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="ascii",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return status


if __name__ == "__main__":
    raise SystemExit(main())
