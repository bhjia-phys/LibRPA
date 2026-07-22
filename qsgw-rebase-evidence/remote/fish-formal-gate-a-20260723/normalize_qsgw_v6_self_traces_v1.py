#!/usr/bin/env python3
"""Normalize native QSGW v6 headers for frozen v5 self validators."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


HEADER = re.compile(r"^(#\s*)(\S+)(?:\s+)(.*?)(\r?\n)?$")
REPLACEMENTS = {
    "qsgw_contract_version": "5",
    "symmetry": "input_kstar_live",
}


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _numeric_bytes(text: str) -> bytes:
    return "".join(
        line for line in text.splitlines(keepends=True)
        if not line.lstrip().startswith("#")
    ).encode("utf-8")


def _contract_headers(text: str) -> dict[str, str]:
    headers: dict[str, str] = {}
    for line in text.splitlines():
        match = HEADER.match(line)
        if match is None:
            continue
        key = match.group(2)
        if key in headers:
            raise ValueError(f"duplicate QSGW contract header {key}")
        headers[key] = match.group(3)
    return headers


def normalize_trace(text: str) -> str:
    headers = _contract_headers(text)
    if headers.get("qsgw_contract_version") != "6":
        raise ValueError("native self trace must use QSGW contract version 6")
    if headers.get("symmetry") != "exx_on_gw_on_rpa_on":
        raise ValueError("native self trace must be symmetry-on for EXX/GW/RPA")

    remaining = set(REPLACEMENTS)
    output: list[str] = []
    for line in text.splitlines(keepends=True):
        match = HEADER.match(line)
        if match is None or match.group(2) not in REPLACEMENTS:
            output.append(line)
            continue
        key = match.group(2)
        newline = match.group(4) or ""
        output.append(f"# {key} {REPLACEMENTS[key]}{newline}")
        remaining.discard(key)
    if remaining:
        raise ValueError(f"cannot normalize missing headers {sorted(remaining)}")

    normalized = "".join(output)
    if _numeric_bytes(normalized) != _numeric_bytes(text):
        raise ValueError("normalization changed numerical trace rows")
    return normalized


def normalize_files(
    *,
    matrix_input: Path,
    eigenvalue_input: Path,
    iteration_input: Path,
    matrix_output: Path,
    eigenvalue_output: Path,
    iteration_output: Path,
    report_output: Path,
) -> dict[str, object]:
    pairs = (
        ("matrix", matrix_input, matrix_output),
        ("eigenvalues", eigenvalue_input, eigenvalue_output),
        ("iterations", iteration_input, iteration_output),
    )
    traces: list[dict[str, object]] = []
    for name, source, destination in pairs:
        source_bytes = source.read_bytes()
        source_text = source_bytes.decode("utf-8")
        normalized = normalize_trace(source_text)
        normalized_bytes = normalized.encode("utf-8")
        destination.write_bytes(normalized_bytes)
        traces.append(
            {
                "name": name,
                "input": str(source),
                "input_sha256": _sha256_bytes(source_bytes),
                "output": str(destination),
                "output_sha256": _sha256_bytes(normalized_bytes),
                "numeric_rows_sha256": _sha256_bytes(_numeric_bytes(source_text)),
                "numeric_rows_unchanged": (
                    _numeric_bytes(source_text) == _numeric_bytes(normalized)
                ),
            }
        )

    report: dict[str, object] = {
        "passed": all(bool(item["numeric_rows_unchanged"]) for item in traces),
        "scope": "contract_headers_only_numeric_rows_unchanged",
        "input_contract_version": 6,
        "output_contract_version": 5,
        "input_symmetry": "exx_on_gw_on_rpa_on",
        "output_symmetry": "input_kstar_live",
        "trace_count": len(traces),
        "traces": traces,
    }
    report_output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("matrix_input", type=Path)
    result.add_argument("eigenvalue_input", type=Path)
    result.add_argument("iteration_input", type=Path)
    result.add_argument("matrix_output", type=Path)
    result.add_argument("eigenvalue_output", type=Path)
    result.add_argument("iteration_output", type=Path)
    result.add_argument("report_output", type=Path)
    return result


def main() -> int:
    args = parser().parse_args()
    report = normalize_files(
        matrix_input=args.matrix_input,
        eigenvalue_input=args.eigenvalue_input,
        iteration_input=args.iteration_input,
        matrix_output=args.matrix_output,
        eigenvalue_output=args.eigenvalue_output,
        iteration_output=args.iteration_output,
        report_output=args.report_output,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
