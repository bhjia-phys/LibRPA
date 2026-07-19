#!/usr/bin/env python3
import argparse
import hashlib
import json
import pathlib
import re
import sys


TABLE_HEADER = (
    r"State\s+occ\s+e_mf\s+v_xc\s+v_exx\s+ReSigc\s+ImSigc\s+e_qp"
)


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=pathlib.Path, required=True)
    parser.add_argument("--candidate", type=pathlib.Path, required=True)
    parser.add_argument("--upstream", type=pathlib.Path, required=True)
    parser.add_argument("--official", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()

    regression_root = args.source / "regression_tests"
    sys.path.insert(0, str(regression_root))
    from backend.validate import Validate

    validate_path = regression_root / "backend" / "validate.py"
    table_path = regression_root / "backend" / "comparisons" / "cmp_table.py"

    def compare(reference: pathlib.Path, label: str) -> dict:
        validator = Validate(
            name=label,
            file=args.candidate.name,
            comparison="cmp_table.abs_diff(0,precision=16)",
            headers="2",
            rows="18",
            regex=TABLE_HEADER,
            occurences=None,
            binary_extract=None,
            file_test=args.candidate.name,
            file_refr=reference.name,
        )
        passed, message = validator.evaluate(
            str(args.candidate.parent), str(reference.parent)
        )
        match = re.search(
            r"max abs diff = (\S+) .* over (\d+) table rows", message
        )
        return {
            "comparison": label,
            "passed": bool(passed),
            "maximum_absolute_difference": (
                float(match.group(1)) if match else None
            ),
            "table_rows": int(match.group(2)) if match else None,
            "message": message,
            "reference": str(reference),
            "reference_sha256": sha256(reference),
        }

    results = [
        compare(args.upstream, "candidate-vs-upstream"),
        compare(args.official, "candidate-vs-official-reference"),
    ]
    payload = {
        "schema": "librpa-g0w0-clean-head-comparison-v1",
        "candidate": str(args.candidate),
        "candidate_sha256": sha256(args.candidate),
        "extractor": {
            "implementation": str(validate_path),
            "sha256": sha256(validate_path),
            "regex": TABLE_HEADER,
            "headers": 2,
            "rows_per_table": 18,
        },
        "comparator": {
            "implementation": str(table_path),
            "sha256": sha256(table_path),
            "function": "cmp_table.abs_diff",
            "absolute_tolerance": 0.0,
        },
        "results": results,
        "passed": all(item["passed"] for item in results),
    }
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
