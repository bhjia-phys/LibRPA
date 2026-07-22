#!/usr/bin/env python3
"""Validate that two rejected legacy prefixes are numerically equivalent."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


SCHEMA = "librpa-qsgw-gate-a-overlap-diagnostic-v1"
EXPECTED_COMPONENTS = {
    "electron_count",
    "exx",
    "fermi_energy_ha",
    "gap_ha",
    "h0",
    "mixed_h",
    "occupation",
    "raw_h",
    "sigma_c_iw",
    "vc",
    "vxc_dft",
}
EXPECTED_CONTRACT_DIFFERENCES = {"qsgw_min_iter", "qsgw_max_iter"}


class DiagnosticError(ValueError):
    pass


def validate(report: dict[str, object], tolerance_ha: float) -> dict[str, object]:
    if tolerance_ha <= 0.0:
        raise DiagnosticError("tolerance must be positive")
    components = report.get("components")
    state = report.get("state")
    contract = report.get("contract")
    if not isinstance(components, dict) or not isinstance(state, dict):
        raise DiagnosticError("base comparison lacks components or state")
    if not isinstance(contract, dict):
        raise DiagnosticError("base comparison lacks contract report")
    names = set(components)
    if names != EXPECTED_COMPONENTS:
        raise DiagnosticError(
            "unexpected component set: "
            f"actual={sorted(names)}, expected={sorted(EXPECTED_COMPONENTS)}"
        )

    maxima: dict[str, float] = {}
    for name in sorted(EXPECTED_COMPONENTS):
        row = components[name]
        if not isinstance(row, dict) or "max_abs_diff" not in row:
            raise DiagnosticError(f"component {name} lacks max_abs_diff")
        value = float(row["max_abs_diff"])
        if value > tolerance_ha:
            raise DiagnosticError(
                f"component {name} differs by {value}, tolerance {tolerance_ha}"
            )
        maxima[name] = value

    state_limits = {
        "max_eigenvalue_abs_diff_ha": tolerance_ha,
        "max_alignment_unitarity_residual": 1e-10,
        "max_rotation_relative_residual": 1e-10,
        "max_velocity_relative_residual": 1e-10,
        "max_wfc_relative_residual": 1e-10,
    }
    state_maxima: dict[str, float] = {}
    for name, limit in state_limits.items():
        if name not in state:
            raise DiagnosticError(f"state report lacks {name}")
        value = float(state[name])
        if value > limit:
            raise DiagnosticError(f"state {name} is {value}, tolerance {limit}")
        state_maxima[name] = value

    differences = contract.get("differences")
    if not isinstance(differences, dict):
        raise DiagnosticError("contract report lacks differences")
    if set(differences) != EXPECTED_CONTRACT_DIFFERENCES:
        raise DiagnosticError(
            "unexpected contract differences: "
            f"actual={sorted(differences)}, "
            f"expected={sorted(EXPECTED_CONTRACT_DIFFERENCES)}"
        )

    return {
        "schema": SCHEMA,
        "passed": True,
        "conclusion": "overlap_alias_does_not_explain_rejected_prefix",
        "matrix_tolerance_ha": tolerance_ha,
        "component_max_abs_differences": maxima,
        "state_maxima": state_maxima,
        "expected_contract_differences": sorted(EXPECTED_CONTRACT_DIFFERENCES),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("comparison", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--tolerance-ha", type=float, default=2e-9)
    args = parser.parse_args()
    try:
        source = json.loads(args.comparison.read_text(encoding="utf-8"))
        result = validate(source, args.tolerance_ha)
        status = 0
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        result = {"schema": SCHEMA, "passed": False, "error": str(error)}
        status = 1
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    args.output.write_text(text, encoding="utf-8", newline="\n")
    print(text, end="")
    return status


if __name__ == "__main__":
    raise SystemExit(main())
