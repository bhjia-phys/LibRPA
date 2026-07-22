#!/usr/bin/env python3
"""Correct exact847 ABACUS Vxc metadata without changing matrix payloads."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


VXC_MAGIC = "# librpa-qsgw-vxc-manifest-v2"
CONTRACT_MAGIC = "# librpa-qsgw-input-contract-v1"
VXC_HEADER = "spin k_index kx ky kz rows columns sha256 file"
CONTRACT_HEADER = "role sha256 file"
SHA256_RE = re.compile(r"[0-9a-f]{64}")
LEGACY_ALIAS_RE = re.compile(r"vxcs(\d+)k(\d+)_nao\.txt")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_ascii_lines(path: Path) -> list[str]:
    if not path.is_file():
        raise ValueError(f"required input file is missing: {path}")
    return path.read_text(encoding="ascii").splitlines()


def require_unique_line(lines: list[str], prefix: str, expected: str) -> int:
    matches = [index for index, line in enumerate(lines) if line.startswith(prefix)]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one {prefix.strip()} line")
    index = matches[0]
    if lines[index] != expected:
        raise ValueError(f"expected {expected!r}, got {lines[index]!r}")
    return index


def validate_relative_file(root: Path, name: str, expected_sha: str) -> Path:
    if not SHA256_RE.fullmatch(expected_sha):
        raise ValueError(f"invalid SHA256 for {name}: {expected_sha}")
    relative = Path(name)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"unsafe relative input path: {name}")
    path = root / relative
    if not path.is_file():
        raise ValueError(f"contract-bound input file is missing: {path}")
    actual_sha = sha256_file(path)
    if actual_sha != expected_sha:
        raise ValueError(
            f"SHA256 mismatch for {path}: expected {expected_sha}, got {actual_sha}"
        )
    return path


def upgrade_manifest(source: Path, output: Path) -> dict[str, object]:
    lines = read_ascii_lines(source)
    if not lines or lines[0] != VXC_MAGIC:
        raise ValueError(f"unexpected Vxc manifest magic: {source}")
    require_unique_line(lines, "kind ", "kind scf")
    require_unique_line(lines, "producer ", "producer abacus")
    require_unique_line(lines, "units ", "units Ry")
    basis_index = require_unique_line(lines, "basis ", "basis nao")
    gauge_index = require_unique_line(lines, "gauge ", "gauge ao_bloch")
    header_index = require_unique_line(lines, "spin ", VXC_HEADER)

    rows: list[dict[str, object]] = []
    seen_indices: set[tuple[int, int]] = set()
    for line_number, line in enumerate(lines[header_index + 1 :], header_index + 2):
        fields = line.split()
        if len(fields) != 9:
            raise ValueError(f"{source}:{line_number}: malformed Vxc row")
        spin = int(fields[0])
        k_index = int(fields[1])
        rows_count = int(fields[5])
        columns_count = int(fields[6])
        expected_sha = fields[7]
        name = fields[8]
        alias = LEGACY_ALIAS_RE.fullmatch(name)
        if alias is None:
            raise ValueError(
                f"{source}:{line_number}: expected exact847 vxcs<spin>k<index> alias"
            )
        if (int(alias.group(1)), int(alias.group(2))) != (spin, k_index):
            raise ValueError(f"{source}:{line_number}: alias indices do not match row")
        if spin <= 0 or k_index <= 0 or rows_count <= 0 or rows_count != columns_count:
            raise ValueError(f"{source}:{line_number}: invalid Vxc dimensions or indices")
        if (spin, k_index) in seen_indices:
            raise ValueError(f"{source}:{line_number}: duplicate Vxc matrix index")
        seen_indices.add((spin, k_index))
        matrix = validate_relative_file(source.parent, name, expected_sha)
        rows.append(
            {
                "columns": columns_count,
                "file": name,
                "k_index": k_index,
                "rows": rows_count,
                "sha256": sha256_file(matrix),
                "spin": spin,
            }
        )
    if not rows:
        raise ValueError(f"Vxc manifest has no matrices: {source}")

    if output.exists():
        raise ValueError(f"refusing to overwrite output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    lines[basis_index] = "basis state"
    lines[gauge_index] = "gauge mf0_state"
    output.write_text("\n".join(lines) + "\n", encoding="ascii")
    return {
        "matrix_count": len(rows),
        "matrices": rows,
        "new_basis": "state",
        "new_gauge": "mf0_state",
        "new_manifest_sha256": sha256_file(output),
        "old_basis": "nao",
        "old_gauge": "ao_bloch",
        "old_manifest_sha256": sha256_file(source),
    }


def upgrade_contract(
    source: Path,
    output: Path,
    source_manifest: Path,
    output_manifest: Path,
    manifest_report: dict[str, object],
) -> dict[str, object]:
    lines = read_ascii_lines(source)
    if not lines or lines[0] != CONTRACT_MAGIC:
        raise ValueError(f"unexpected QSGW input contract magic: {source}")
    header_index = require_unique_line(lines, "role ", CONTRACT_HEADER)
    old_sha = str(manifest_report["old_manifest_sha256"])
    new_sha = str(manifest_report["new_manifest_sha256"])
    role_count = 0
    manifest_role_index: int | None = None
    verified_roles: list[dict[str, str]] = []
    for line_index, line in enumerate(lines[header_index + 1 :], header_index + 1):
        fields = line.split()
        if len(fields) != 3:
            raise ValueError(f"{source}:{line_index + 1}: malformed contract role")
        role, expected_sha, name = fields
        validate_relative_file(source.parent, name, expected_sha)
        role_count += 1
        verified_roles.append({"file": name, "role": role, "sha256": expected_sha})
        if role == "vxc_scf_manifest":
            if manifest_role_index is not None:
                raise ValueError("contract has multiple vxc_scf_manifest roles")
            if expected_sha != old_sha or name != source_manifest.name:
                raise ValueError("contract Vxc role does not bind the source manifest")
            manifest_role_index = line_index
    if manifest_role_index is None:
        raise ValueError("contract has no vxc_scf_manifest role")
    if output_manifest.name != source_manifest.name:
        raise ValueError("corrected manifest must preserve the contract-bound filename")

    if output.exists():
        raise ValueError(f"refusing to overwrite output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    lines[manifest_role_index] = (
        f"vxc_scf_manifest {new_sha} {output_manifest.name}"
    )
    output.write_text("\n".join(lines) + "\n", encoding="ascii")
    return {
        "new_contract_sha256": sha256_file(output),
        "old_contract_sha256": sha256_file(source),
        "verified_role_count": role_count,
        "verified_roles": verified_roles,
    }


def run_upgrade(
    source_contract: Path,
    source_manifest: Path,
    output_contract: Path,
    output_manifest: Path,
    report_path: Path,
) -> dict[str, object]:
    resolved_inputs = {source_contract.resolve(), source_manifest.resolve()}
    resolved_outputs = {output_contract.resolve(), output_manifest.resolve()}
    if resolved_inputs & resolved_outputs:
        raise ValueError("source and output files must be distinct")
    if report_path.exists():
        raise ValueError(f"refusing to overwrite report: {report_path}")

    manifest = upgrade_manifest(source_manifest, output_manifest)
    contract = upgrade_contract(
        source_contract,
        output_contract,
        source_manifest,
        output_manifest,
        manifest,
    )
    report = {
        "contract": contract,
        "manifest": manifest,
        "passed": True,
        "scope": "exact847_legacy_alias_metadata_only",
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="ascii"
    )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("source_contract", type=Path)
    parser.add_argument("source_manifest", type=Path)
    parser.add_argument("output_contract", type=Path)
    parser.add_argument("output_manifest", type=Path)
    parser.add_argument("report", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = run_upgrade(
        args.source_contract,
        args.source_manifest,
        args.output_contract,
        args.output_manifest,
        args.report,
    )
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
