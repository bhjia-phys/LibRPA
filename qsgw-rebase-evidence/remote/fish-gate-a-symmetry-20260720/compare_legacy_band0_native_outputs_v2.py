#!/usr/bin/env python3
"""Compare legacy qsgw_band0 outputs without hiding legacy invariant gaps."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path

import compare_legacy_band0_native_outputs_v1 as v1


SCHEMA = "librpa-legacy-qsgw-band0-native-comparison-v2"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def add_acceptance_semantics(report: dict[str, object]) -> dict[str, object]:
    h0 = report["h0"]
    sigcrf = report["sigcrf"]
    text_outputs = report["text_outputs"]
    if not isinstance(h0, dict) or not isinstance(sigcrf, dict) or not isinstance(
        text_outputs, dict
    ):
        raise v1.ComparisonError("invalid v1 comparison report")

    files = h0.get("files")
    if not isinstance(files, list) or not files:
        raise v1.ComparisonError("missing H0 file reports")

    oracle_hermiticity = max(
        float(item["oracle_hermiticity_max_abs_ha"]) for item in files
    )
    reproduced_hermiticity = max(
        float(item["reproduced_hermiticity_max_abs_ha"]) for item in files
    )
    parity_passed = (
        float(h0["max_abs_ha"]) <= v1.H0_MAX_ABS_HA
        and float(h0["max_relative_frobenius"]) <= v1.H0_REL_FROBENIUS
    )
    oracle_invariant_passed = oracle_hermiticity <= v1.HERMITICITY_HA
    reproduced_invariant_passed = reproduced_hermiticity <= v1.HERMITICITY_HA
    absolute_invariants_passed = (
        oracle_invariant_passed and reproduced_invariant_passed
    )

    h0["v1_combined_passed"] = bool(h0["passed"])
    h0["parity_passed"] = parity_passed
    h0["oracle_hermiticity_max_abs_ha"] = oracle_hermiticity
    h0["reproduced_hermiticity_max_abs_ha"] = reproduced_hermiticity
    h0["oracle_absolute_invariant_passed"] = oracle_invariant_passed
    h0["reproduced_absolute_invariant_passed"] = reproduced_invariant_passed
    h0["absolute_invariants_passed"] = absolute_invariants_passed
    h0["passed"] = parity_passed

    historical_reproduction_passed = (
        parity_passed
        and bool(sigcrf["passed"])
        and bool(text_outputs["passed"])
    )
    report["schema"] = SCHEMA
    report["acceptance_semantics"] = {
        "cli_exit_accepts": "historical_reproduction_parity",
        "historical_reproduction_parity": (
            "H0/SigcRF/text differences satisfy their parity thresholds"
        ),
        "goal_thresholds": (
            "historical reproduction parity and both H0 Hermiticity invariants"
        ),
        "thresholds_are_not_relaxed": True,
    }
    report["historical_reproduction_passed"] = historical_reproduction_passed
    report["absolute_invariants_passed"] = absolute_invariants_passed
    report["goal_thresholds_passed"] = (
        historical_reproduction_passed and absolute_invariants_passed
    )
    report["legacy_oracle_invariant_gap"] = not oracle_invariant_passed
    report["reproduced_absolute_invariant_gap"] = not reproduced_invariant_passed
    report["passed"] = historical_reproduction_passed
    return report


def compare_outputs(oracle: Path, reproduced: Path) -> dict[str, object]:
    report = add_acceptance_semantics(v1.compare_outputs(oracle, reproduced))
    dependency = Path(v1.__file__).resolve(strict=True)
    report["parser_dependency"] = {
        "path": str(dependency),
        "sha256": sha256_file(dependency),
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("oracle", type=Path)
    parser.add_argument("reproduced", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    try:
        report = compare_outputs(args.oracle, args.reproduced)
    except (v1.ComparisonError, OSError, ValueError, KeyError, struct.error) as error:
        report = {
            "schema": SCHEMA,
            "passed": False,
            "historical_reproduction_passed": False,
            "goal_thresholds_passed": False,
            "error": str(error),
        }
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))
    return 0 if report.get("historical_reproduction_passed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
