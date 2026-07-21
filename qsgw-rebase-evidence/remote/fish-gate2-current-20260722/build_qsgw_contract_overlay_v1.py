#!/usr/bin/env python3
"""Bind a frozen QSGW contract to a derived reader-static stru_out."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


class ContractOverlayError(ValueError):
    pass


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def build(
    source_contract: Path,
    source_stru: Path,
    overlay_stru: Path,
    output_contract: Path,
    report_path: Path,
) -> dict[str, object]:
    source_bytes = source_contract.read_bytes()
    source_stru_sha = _sha256(source_stru)
    overlay_stru_sha = _sha256(overlay_stru)
    if source_stru_sha == overlay_stru_sha:
        raise ContractOverlayError("source and overlay stru_out hashes are equal")
    if not source_bytes.startswith(b"# librpa-qsgw-input-contract-v1\n"):
        raise ContractOverlayError("unsupported QSGW input contract header")
    if source_bytes.count(b"role sha256 file\n") != 1:
        raise ContractOverlayError("QSGW input contract role header is not unique")

    target = (
        b"reader_static "
        + source_stru_sha.encode("ascii")
        + b" stru_out"
    )
    replacement = (
        b"reader_static "
        + overlay_stru_sha.encode("ascii")
        + b" stru_out"
    )
    if source_bytes.count(target) != 1:
        raise ContractOverlayError(
            "QSGW input contract does not bind exactly one source stru_out"
        )
    if source_bytes.count(b" stru_out") != 1:
        raise ContractOverlayError("QSGW input contract has ambiguous stru_out rows")
    output_bytes = source_bytes.replace(target, replacement, 1)
    if len(output_bytes) != len(source_bytes):
        raise ContractOverlayError("derived contract byte length changed")
    changed = [
        index
        for index, (before, after) in enumerate(zip(source_bytes, output_bytes))
        if before != after
    ]
    if not changed:
        raise ContractOverlayError("derived contract did not change")
    output_contract.write_bytes(output_bytes)
    output_sha = _sha256_bytes(output_bytes)
    report = {
        "schema": "librpa-qsgw-input-contract-stru-overlay-v1",
        "passed": True,
        "source_contract_sha256": _sha256_bytes(source_bytes),
        "output_contract_sha256": output_sha,
        "source_stru_out_sha256": source_stru_sha,
        "overlay_stru_out_sha256": overlay_stru_sha,
        "output_length_bytes": len(output_bytes),
        "changed_byte_count": len(changed),
        "first_changed_byte": changed[0],
        "last_changed_byte": changed[-1],
        "replacement_count": 1,
        "unchanged_except_stru_sha256": (
            output_bytes.replace(
                overlay_stru_sha.encode("ascii"),
                source_stru_sha.encode("ascii"),
                1,
            )
            == source_bytes
        ),
    }
    if report["unchanged_except_stru_sha256"] is not True:
        raise ContractOverlayError("derived contract changed unrelated bytes")
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source_contract", type=Path)
    parser.add_argument("source_stru", type=Path)
    parser.add_argument("overlay_stru", type=Path)
    parser.add_argument("output_contract", type=Path)
    parser.add_argument("report", type=Path)
    args = parser.parse_args()
    try:
        report = build(
            args.source_contract,
            args.source_stru,
            args.overlay_stru,
            args.output_contract,
            args.report,
        )
        status = 0
    except (OSError, ValueError) as error:
        report = {
            "schema": "librpa-qsgw-input-contract-stru-overlay-v1",
            "passed": False,
            "error": str(error),
        }
        args.report.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        status = 1
    print(json.dumps(report, indent=2, sort_keys=True))
    return status


if __name__ == "__main__":
    raise SystemExit(main())
