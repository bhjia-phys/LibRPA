#!/usr/bin/env python3
"""Classify an exact847 SigmaC restart/projection probe."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


SCHEMA = "librpa-exact847-sigma-projection-probe-v1"


class ClassificationError(ValueError):
    pass


def classify(
    component_report: dict[str, object],
    max_abs_tolerance_ha: float,
    relative_frobenius_tolerance: float,
) -> dict[str, object]:
    if component_report.get("diagnostic_complete") is not True:
        raise ClassificationError("component comparison is incomplete")

    try:
        frequency_difference = float(
            component_report["frequency_grid_max_abs_ha"]
        )
        sigma_overall = component_report["sigma_c_iw"]["overall"]
        max_abs_ha = float(sigma_overall["max_abs_ha"])
        relative_frobenius = float(sigma_overall["relative_frobenius"])
    except (KeyError, TypeError, ValueError) as error:
        raise ClassificationError(
            "component comparison has no usable SigmaC metrics"
        ) from error

    frequency_tolerance_ha = 1.0e-14
    passed = (
        frequency_difference <= frequency_tolerance_ha
        and max_abs_ha <= max_abs_tolerance_ha
        and relative_frobenius <= relative_frobenius_tolerance
    )
    return {
        "schema": SCHEMA,
        "diagnostic_complete": True,
        "acceptance": "diagnostic_only",
        "frequency_grid_max_abs_ha": frequency_difference,
        "sigma_c_iw_max_abs_ha": max_abs_ha,
        "sigma_c_iw_relative_frobenius": relative_frobenius,
        "thresholds": {
            "frequency_grid_max_abs_ha": frequency_tolerance_ha,
            "sigma_c_iw_max_abs_ha": max_abs_tolerance_ha,
            "sigma_c_iw_relative_frobenius": relative_frobenius_tolerance,
        },
        "sigma_projection_parity_passed": passed,
        "inference": (
            "candidate_projection_matches_given_legacy_sigcrf"
            if passed
            else "candidate_projection_or_legacy_sigcrf_reader_differs"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("component_report", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--max-abs-tolerance-ha", type=float, default=1.0e-6)
    parser.add_argument(
        "--relative-frobenius-tolerance", type=float, default=1.0e-8
    )
    args = parser.parse_args()

    try:
        component_report = json.loads(args.component_report.read_text())
        report = classify(
            component_report,
            args.max_abs_tolerance_ha,
            args.relative_frobenius_tolerance,
        )
    except (ClassificationError, OSError, json.JSONDecodeError) as error:
        report = {
            "schema": SCHEMA,
            "diagnostic_complete": False,
            "error": str(error),
        }
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))
    return 0 if report.get("diagnostic_complete") else 2


if __name__ == "__main__":
    raise SystemExit(main())
