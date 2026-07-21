#!/usr/bin/env python3
"""Validate legacy/current reader views over one frozen physical dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


SCHEMA = "librpa-qsgw-gate-a-input-views-v1"
ALLOWED_DIFFERENCES = {"qsgw_input.contract", "stru_out"}


class InputViewError(ValueError):
    pass


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def input_files(root: Path) -> dict[str, Path]:
    if not root.is_dir():
        raise InputViewError(f"input view is not a directory: {root}")
    files: dict[str, Path] = {}
    for path in root.iterdir():
        if not path.is_file():
            raise InputViewError(f"input view contains a non-file: {path.name}")
        files[path.name] = path
    return files


def symmetry_tail(source: bytes, composite: bytes) -> dict[str, object]:
    if not composite.startswith(source):
        raise InputViewError("candidate stru_out is not byte-prefixed by legacy stru_out")
    tail = composite[len(source) :]
    if not source.endswith(b"\n"):
        if not tail.startswith(b"\n"):
            raise InputViewError("candidate stru_out has no separator after source bytes")
        tail = tail[1:]
    tokens = tail.split()
    if len(tokens) < 2:
        raise InputViewError("candidate stru_out has no symmetry tail")
    try:
        count = int(tokens[0])
    except ValueError as error:
        raise InputViewError("symmetry tail count is not an integer") from error
    convention = tokens[1].decode("ascii", errors="strict").lower()
    if count <= 1:
        raise InputViewError("symmetry tail must contain more than identity")
    if convention != "row":
        raise InputViewError(f"symmetry tail convention is {convention!r}, expected 'row'")
    expected_tokens = 2 + 12 * count
    if len(tokens) != expected_tokens:
        raise InputViewError(
            f"symmetry tail has {len(tokens)} tokens, expected {expected_tokens}"
        )
    return {
        "source_prefix_byte_identical": True,
        "symmetry_operation_count": count,
        "symmetry_convention": convention,
        "symmetry_tail_byte_count": len(tail),
    }


def validate_contracts(
    source_contract: bytes,
    candidate_contract: bytes,
    source_stru_sha: str,
    candidate_stru_sha: str,
) -> dict[str, object]:
    header = b"# librpa-qsgw-input-contract-v1\n"
    if not source_contract.startswith(header) or not candidate_contract.startswith(header):
        raise InputViewError("unsupported QSGW input contract header")
    source_row = (
        b"reader_static " + source_stru_sha.encode("ascii") + b" stru_out"
    )
    candidate_row = (
        b"reader_static " + candidate_stru_sha.encode("ascii") + b" stru_out"
    )
    if source_contract.count(source_row) != 1:
        raise InputViewError("legacy contract does not bind its stru_out exactly once")
    if candidate_contract.count(candidate_row) != 1:
        raise InputViewError("candidate contract does not bind its stru_out exactly once")
    if source_contract.count(b" stru_out") != 1:
        raise InputViewError("legacy contract has ambiguous stru_out rows")
    if candidate_contract.count(b" stru_out") != 1:
        raise InputViewError("candidate contract has ambiguous stru_out rows")
    restored = candidate_contract.replace(
        candidate_stru_sha.encode("ascii"), source_stru_sha.encode("ascii"), 1
    )
    if restored != source_contract:
        raise InputViewError("contracts differ outside the bound stru_out SHA256")
    return {
        "contracts_unchanged_except_stru_sha256": True,
        "source_contract_sha256": sha256_bytes(source_contract),
        "candidate_contract_sha256": sha256_bytes(candidate_contract),
    }


def validate(legacy_view: Path, candidate_view: Path) -> dict[str, object]:
    legacy = input_files(legacy_view)
    candidate = input_files(candidate_view)
    legacy_names = set(legacy)
    candidate_names = set(candidate)
    if legacy_names != candidate_names:
        raise InputViewError(
            "input view file sets differ: "
            f"legacy_only={sorted(legacy_names - candidate_names)}, "
            f"candidate_only={sorted(candidate_names - legacy_names)}"
        )
    missing = ALLOWED_DIFFERENCES - legacy_names
    if missing:
        raise InputViewError(f"input views lack required metadata files: {sorted(missing)}")

    legacy_hashes = {name: sha256(path) for name, path in legacy.items()}
    candidate_hashes = {name: sha256(path) for name, path in candidate.items()}
    differences = {
        name for name in legacy_names if legacy_hashes[name] != candidate_hashes[name]
    }
    if differences != ALLOWED_DIFFERENCES:
        raise InputViewError(
            "input view SHA256 differences are not exactly the reader metadata allowlist: "
            f"actual={sorted(differences)}, expected={sorted(ALLOWED_DIFFERENCES)}"
        )

    source_stru = legacy["stru_out"].read_bytes()
    candidate_stru = candidate["stru_out"].read_bytes()
    source_stru_sha = legacy_hashes["stru_out"]
    candidate_stru_sha = candidate_hashes["stru_out"]
    tail_report = symmetry_tail(source_stru, candidate_stru)
    contract_report = validate_contracts(
        legacy["qsgw_input.contract"].read_bytes(),
        candidate["qsgw_input.contract"].read_bytes(),
        source_stru_sha,
        candidate_stru_sha,
    )

    return {
        "schema": SCHEMA,
        "passed": True,
        "legacy_view": str(legacy_view),
        "candidate_view": str(candidate_view),
        "file_count": len(legacy_names),
        "common_physical_file_count": len(legacy_names - ALLOWED_DIFFERENCES),
        "allowed_reader_metadata_differences": sorted(ALLOWED_DIFFERENCES),
        "actual_sha256_differences": sorted(differences),
        "common_file_sha256_identical": True,
        "source_stru_out_sha256": source_stru_sha,
        "candidate_stru_out_sha256": candidate_stru_sha,
        **tail_report,
        **contract_report,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("legacy_view", type=Path)
    parser.add_argument("candidate_view", type=Path)
    parser.add_argument("report", type=Path)
    args = parser.parse_args()
    try:
        report = validate(args.legacy_view, args.candidate_view)
        status = 0
    except (OSError, UnicodeError, ValueError) as error:
        report = {"schema": SCHEMA, "passed": False, "error": str(error)}
        status = 1
    args.report.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return status


if __name__ == "__main__":
    raise SystemExit(main())
