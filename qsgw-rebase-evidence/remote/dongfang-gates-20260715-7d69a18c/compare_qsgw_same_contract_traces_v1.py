#!/usr/bin/env python3
"""Compare two QSGW traces with the same legacy contract."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import compare_qsgw_component_traces_v3 as base


HARTREE_TO_EV = 27.211386245988
FREQUENCY_TOLERANCE_HA = 1.0e-10


def summarize_physical_blocks(
    old_rows: dict[tuple[Any, ...], tuple[float, complex]],
    new_rows: dict[tuple[Any, ...], tuple[float, complex]],
) -> dict[str, dict[str, float | int]]:
    state_components = {
        key[2]
        for key in old_rows
        if key[2].startswith("wfc_spinor")
        or key[2].startswith("velocity_")
        or key[2] == "rotation_u"
    }
    old_keys = {key for key in old_rows if key[2] not in state_components}
    new_keys = {key for key in new_rows if key[2] not in state_components}
    if old_keys != new_keys:
        raise ValueError("physical component row keys differ")

    blocks: dict[tuple[Any, ...], dict[str, float | int]] = {}
    for key in sorted(old_keys):
        old_frequency, old_value = old_rows[key]
        new_frequency, new_value = new_rows[key]
        if abs(old_frequency - new_frequency) > FREQUENCY_TOLERANCE_HA:
            raise ValueError(f"frequency differs for row {key}")
        block_key = key[:6] + (old_frequency,)
        difference = abs(old_value - new_value)
        block = blocks.setdefault(
            block_key,
            {"count": 0, "max_abs": 0.0, "diff_norm2": 0.0, "ref_norm2": 0.0},
        )
        block["count"] = int(block["count"]) + 1
        block["max_abs"] = max(float(block["max_abs"]), difference)
        block["diff_norm2"] = float(block["diff_norm2"]) + difference**2
        block["ref_norm2"] = float(block["ref_norm2"]) + abs(old_value) ** 2

    components: dict[str, dict[str, float | int]] = {}
    for block_key, block in blocks.items():
        component = str(block_key[2])
        reference_norm = math.sqrt(float(block["ref_norm2"]))
        difference_norm = math.sqrt(float(block["diff_norm2"]))
        relative = (
            difference_norm / reference_norm
            if reference_norm != 0.0
            else (0.0 if difference_norm == 0.0 else math.inf)
        )
        entry = components.setdefault(
            component,
            {
                "block_count": 0,
                "row_count": 0,
                "max_abs_diff_ha": 0.0,
                "max_relative_frobenius": 0.0,
            },
        )
        entry["block_count"] = int(entry["block_count"]) + 1
        entry["row_count"] = int(entry["row_count"]) + int(block["count"])
        entry["max_abs_diff_ha"] = max(
            float(entry["max_abs_diff_ha"]), float(block["max_abs"])
        )
        entry["max_relative_frobenius"] = max(
            float(entry["max_relative_frobenius"]), relative
        )
    return components


def compare(
    old_text: str,
    new_text: str,
    iterations: list[int],
    channel: int,
    matrix_max_abs_tolerance_ha: float,
    matrix_relative_tolerance: float,
    eigenvalue_tolerance_ha: float,
    gap_tolerance_ev: float,
    state_tolerance: float,
) -> dict[str, Any]:
    base_report = base.compare_trace_text(
        old_text,
        new_text,
        iterations,
        channel=channel,
        frequency_tolerance=1.0e-10,
        eigenvalue_tolerance=eigenvalue_tolerance_ha,
        degeneracy_tolerance=1.0e-8,
        state_tolerance=state_tolerance,
        contract_mode="oracle",
    )
    selected = set(iterations)
    old_contract = base.parse_contract(old_text, "old trace")
    new_contract = base.parse_contract(new_text, "new trace")
    old_rows = base.parse_rows(old_text, "old trace", selected, channel)
    new_rows = base.parse_rows(new_text, "new trace", selected, channel)
    old_rows, _ = base._drop_optional_zero_velocity_rows(old_rows, old_contract)
    new_rows, _ = base._drop_optional_zero_velocity_rows(new_rows, new_contract)
    components = summarize_physical_blocks(old_rows, new_rows)

    component_passed = True
    for metrics in components.values():
        passed = (
            float(metrics["max_abs_diff_ha"]) <= matrix_max_abs_tolerance_ha
            and float(metrics["max_relative_frobenius"]) <= matrix_relative_tolerance
        )
        metrics["max_abs_tolerance_ha"] = matrix_max_abs_tolerance_ha
        metrics["relative_frobenius_tolerance"] = matrix_relative_tolerance
        metrics["passed"] = passed
        component_passed = component_passed and passed

    gap_differences = [
        abs(old_rows[key][1].real - new_rows[key][1].real) * HARTREE_TO_EV
        for key in old_rows
        if key[2] == "gap_ha"
    ]
    max_gap_difference_ev = max(gap_differences, default=0.0)
    gap_passed = max_gap_difference_ev <= gap_tolerance_ev
    return {
        "passed": bool(base_report["passed"] and component_passed and gap_passed),
        "iterations": iterations,
        "channel": channel,
        "base_report": base_report,
        "components": components,
        "max_gap_difference_ev": max_gap_difference_ev,
        "gap_tolerance_ev": gap_tolerance_ev,
        "gap_passed": gap_passed,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("old_trace", type=Path)
    parser.add_argument("new_trace", type=Path)
    parser.add_argument("output_json", type=Path)
    parser.add_argument("--iterations", default="0:2")
    parser.add_argument("--channel", type=int, default=0)
    parser.add_argument("--matrix-max-abs-tolerance-ha", type=float, default=1.0e-8)
    parser.add_argument("--matrix-relative-tolerance", type=float, default=1.0e-8)
    parser.add_argument("--eigenvalue-tolerance-ha", type=float, default=1.0e-6)
    parser.add_argument("--gap-tolerance-ev", type=float, default=1.0e-5)
    parser.add_argument("--state-tolerance", type=float, default=1.0e-10)
    args = parser.parse_args()

    try:
        report = compare(
            args.old_trace.read_text(encoding="utf-8"),
            args.new_trace.read_text(encoding="utf-8"),
            base.parse_iterations(args.iterations),
            args.channel,
            args.matrix_max_abs_tolerance_ha,
            args.matrix_relative_tolerance,
            args.eigenvalue_tolerance_ha,
            args.gap_tolerance_ev,
            args.state_tolerance,
        )
    except Exception as error:
        report = {"passed": False, "error": str(error)}
    report["old_trace"] = str(args.old_trace.resolve())
    report["new_trace"] = str(args.new_trace.resolve())
    args.output_json.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
