#!/usr/bin/env python3
"""Audit native ABACUS vxck compatibility in frozen legacy and candidate readers."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


EXPECTED_RAW_EXACT847_SHA256 = (
    "a932e60fa1b96e10a44eaa2f07da23bf57bbbb1c36600b1d4f9861a5d843e68e"
)
EXPECTED_SCHEME_A_SHA256 = (
    "34e5c93fe12259f4838469b0e19b2c2316d4b6871ba9d6c9da5b3c057a01ab34"
)


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def extract_block(source: str, start_marker: str, end_marker: str) -> str:
    start = source.index(start_marker)
    end = source.index(end_marker, start)
    return source[start:end]


def require_fragments(source: str, fragments: list[str], label: str) -> None:
    missing = [fragment for fragment in fragments if fragment not in source]
    if missing:
        raise ValueError(f"{label}: missing required source fragments: {missing}")


def provenance_path(path: Path, repo: Path) -> str:
    try:
        return path.relative_to(repo).as_posix()
    except ValueError:
        return str(path)


def build_report(repo: Path, legacy_raw: Path, legacy_scheme_a: Path) -> dict[str, object]:
    repo = repo.resolve(strict=True)
    legacy_raw = legacy_raw.resolve(strict=True)
    legacy_scheme_a = legacy_scheme_a.resolve(strict=True)
    candidate_vxc = repo / "src/qsgw/vxc_io.cpp"
    candidate_driver = repo / "driver/tasks/qsgw.cpp"
    candidate_test = repo / "src/test/test_qsgw_vxc_io.cpp"
    for path in (candidate_vxc, candidate_driver, candidate_test):
        if not path.is_file():
            raise ValueError(f"missing candidate source observer: {path}")

    raw_sha = sha256_file(legacy_raw)
    scheme_a_sha = sha256_file(legacy_scheme_a)
    if raw_sha != EXPECTED_RAW_EXACT847_SHA256:
        raise ValueError(f"raw exact-847 source SHA256 mismatch: {raw_sha}")
    if scheme_a_sha != EXPECTED_SCHEME_A_SHA256:
        raise ValueError(f"Scheme-A source SHA256 mismatch: {scheme_a_sha}")

    raw_source = legacy_raw.read_text(encoding="utf-8", errors="replace")
    scheme_a_source = legacy_scheme_a.read_text(encoding="utf-8", errors="replace")
    parser_start = "bool read_abacus_upper_triangle_matrix"
    parser_end = "ComplexMatrix matz_to_complex_matrix"
    raw_parser = extract_block(raw_source, parser_start, parser_end)
    scheme_a_parser = extract_block(scheme_a_source, parser_start, parser_end)
    raw_parser_sha = sha256_bytes(raw_parser.encode("utf-8"))
    scheme_a_parser_sha = sha256_bytes(scheme_a_parser.encode("utf-8"))
    if raw_parser != scheme_a_parser:
        raise ValueError("Scheme-A harness changed the exact-847 ABACUS matrix reader")
    require_fragments(
        raw_source,
        [
            'rows_regex("^\\\\s*#\\\\s*rows',
            'cols_regex("^\\\\s*#\\\\s*columns',
            "values.size() == upper_count",
            "parsed(row, col) = scale * values[index++]",
            'oss_vxc_text_k << "vxck"',
        ],
        "exact-847 reader",
    )

    candidate_vxc_source = candidate_vxc.read_text(
        encoding="utf-8", errors="replace"
    )
    candidate_driver_source = candidate_driver.read_text(
        encoding="utf-8", errors="replace"
    )
    candidate_test_source = candidate_test.read_text(
        encoding="utf-8", errors="replace"
    )
    require_fragments(
        candidate_vxc_source,
        [
            'lowered.rfind("# rows", 0)',
            'lowered.rfind("# columns", 0)',
            "saw_row_marker",
            "0.5 * row_it->second[offset]",
            "Legacy ABACUS triangular Vxc entry count mismatch",
        ],
        "candidate reader",
    )
    require_fragments(
        candidate_driver_source,
        ["read_abacus_vxc_ha(stream, matrix_path)"],
        "candidate QSGW driver",
    )
    require_fragments(
        candidate_test_source,
        ['"# rows 2\\n"', '"Row 1\\n"', 'read_abacus_vxc_ha(input, "synthetic-vxck")'],
        "candidate Vxc unit test",
    )

    return {
        "schema": "librpa-native-vxck-reader-compatibility-v1",
        "source_contract_passed": True,
        "legacy_parser_unchanged_by_scheme_a": True,
        "both_readers_accept_native_comment_row_schema": True,
        "both_readers_convert_rydberg_to_hartree": True,
        "native_file_alias_required": False,
        "candidate_manifest_should_bind_native_vxck": True,
        "numerical_runtime_parity_pending": True,
        "legacy": {
            "commit": "8476213f66c68efb43404713eacbd04966820f26",
            "raw_source": provenance_path(legacy_raw, repo),
            "raw_source_sha256": raw_sha,
            "scheme_a_source": provenance_path(legacy_scheme_a, repo),
            "scheme_a_source_sha256": scheme_a_sha,
            "reader_slice_sha256": raw_parser_sha,
            "scheme_a_reader_slice_sha256": scheme_a_parser_sha,
            "reader_slice_bytes": len(raw_parser.encode("utf-8")),
            "energy_conversion": "Ry_to_Ha_scale_0.5",
        },
        "candidate": {
            "vxc_io_source": provenance_path(candidate_vxc.resolve(), repo),
            "vxc_io_source_sha256": sha256_file(candidate_vxc),
            "driver_source": provenance_path(candidate_driver.resolve(), repo),
            "driver_source_sha256": sha256_file(candidate_driver),
            "unit_test_source": provenance_path(candidate_test.resolve(), repo),
            "unit_test_source_sha256": sha256_file(candidate_test),
            "energy_conversion": "Ry_to_Ha_scale_0.5",
        },
    }


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    default_repo = script_dir.parents[2]
    oracle_root = (
        default_repo
        / "qsgw-rebase-evidence/remote/fish-gate-a-symmetry-20260720"
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=default_repo)
    parser.add_argument(
        "--legacy-raw",
        type=Path,
        default=oracle_root / "oracle-source-audit-v1/exact847/task_qsgw_band_0.cpp",
    )
    parser.add_argument(
        "--legacy-scheme-a",
        type=Path,
        default=oracle_root / "patch-work-scheme-a-v3/task_qsgw_band_0.cpp",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    report = build_report(args.repo, args.legacy_raw, args.legacy_scheme_a)
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
