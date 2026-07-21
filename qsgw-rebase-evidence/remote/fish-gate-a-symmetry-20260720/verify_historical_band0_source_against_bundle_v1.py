#!/usr/bin/env python3
"""Compare files still present in a historical run with a frozen bundle manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def compare(manifest: Path, source_root: Path) -> dict[str, object]:
    matching: list[str] = []
    missing: list[str] = []
    mismatches: list[dict[str, str]] = []
    total = 0

    for raw_line in manifest.read_text(encoding="utf-8").splitlines():
        if not raw_line.strip():
            continue
        expected, relative = raw_line.split(maxsplit=1)
        relative = relative.removeprefix("dataset/")
        total += 1
        source = source_root / relative
        if not source.is_file():
            missing.append(relative)
            continue
        observed = sha256_file(source)
        if observed == expected:
            matching.append(relative)
        else:
            mismatches.append(
                {
                    "path": relative,
                    "expected_sha256": expected,
                    "observed_sha256": observed,
                }
            )

    return {
        "schema": "librpa-historical-band0-source-bundle-check-v1",
        "manifest": str(manifest.resolve()),
        "source_root": str(source_root.resolve()),
        "manifest_entries": total,
        "existing_entries": len(matching) + len(mismatches),
        "matching_entries": len(matching),
        "mismatch_count": len(mismatches),
        "missing_count": len(missing),
        "matching_paths": matching,
        "mismatches": mismatches,
        "missing_paths": missing,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("source_root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--minimum-existing", type=int, default=1)
    args = parser.parse_args()

    result = compare(args.manifest, args.source_root)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, sort_keys=True))
    if result["mismatch_count"] != 0:
        return 1
    if result["existing_entries"] < args.minimum_existing:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
