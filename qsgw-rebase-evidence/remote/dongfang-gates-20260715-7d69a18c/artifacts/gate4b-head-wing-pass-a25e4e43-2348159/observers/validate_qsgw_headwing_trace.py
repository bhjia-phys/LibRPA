#!/usr/bin/env python3
"""Validate live same-grid velocity and analytic head tensors in QSGW traces."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from compare_qsgw_component_traces_v3 import _matrix_groups, parse_iterations, parse_rows


class HeadwingTraceError(ValueError):
    """Raised when a QSGW head/wing trace is structurally incomplete."""


def _header(text: str) -> dict[str, str]:
    accepted = {
        "qsgw_contract_version",
        "fixed_basis",
        "live_update",
        "velocity",
        "headwing",
        "symmetry",
        "hartree",
        "band",
        "qsgw_input_contract",
        "qsgw_input_contract_sha256",
        "qsgw_mixer",
        "qsgw_mixing_beta",
    }
    values: dict[str, str] = {}
    for line_number, line in enumerate(text.splitlines(), 1):
        stripped = line.strip()
        if not stripped.startswith("#"):
            continue
        fields = stripped[1:].strip().split(None, 1)
        if len(fields) != 2 or fields[0] not in accepted:
            continue
        if fields[0] in values:
            raise HeadwingTraceError(
                f"trace:{line_number}: duplicate header {fields[0]}"
            )
        values[fields[0]] = fields[1]
    required = {
        "qsgw_contract_version": "5",
        "fixed_basis": "immutable_mf0",
        "live_update": "eigenvalues_wfc",
        "velocity": "fixed_basis_rotation",
        "headwing": "scf_grid_analytic_live",
        "symmetry": "unsupported_full_bz_only",
        "hartree": "disabled_stage1",
        "band": "disabled_stage1",
    }
    differences = {
        key: {"expected": expected, "actual": values.get(key)}
        for key, expected in required.items()
        if values.get(key) != expected
    }
    if differences:
        raise HeadwingTraceError(f"invalid head/wing trace contract: {differences}")
    contract_sha = values.get("qsgw_input_contract_sha256", "")
    if len(contract_sha) != 64 or any(
        character not in "0123456789abcdef" for character in contract_sha
    ):
        raise HeadwingTraceError("invalid input-contract SHA256 header")
    return values


def _maximum_abs(matrix: np.ndarray) -> float:
    return float(np.max(np.abs(matrix))) if matrix.size else 0.0


def _relative(difference: np.ndarray, *references: np.ndarray) -> float:
    scale = max((float(np.linalg.norm(value)) for value in references), default=0.0)
    return float(np.linalg.norm(difference) / max(scale, 1.0e-300))


def validate(
    matrix_text: str,
    iterations: list[int],
    *,
    channel: int,
    invariant_tolerance: float,
    head_live_change_minimum: float,
    initial_replay_tolerance: float = 1.0e-10,
) -> dict[str, object]:
    if not iterations or iterations != list(range(iterations[-1] + 1)):
        raise HeadwingTraceError("iterations must be continuous and begin at zero")
    if iterations[-1] < 1:
        raise HeadwingTraceError("at least one QSGW update is required")
    if channel != 0:
        raise HeadwingTraceError("same-grid head/wing requires grid channel zero")
    for label, value in (
        ("invariant", invariant_tolerance),
        ("initial replay", initial_replay_tolerance),
        ("head live-change minimum", head_live_change_minimum),
    ):
        if not math.isfinite(value) or value < 0.0:
            raise HeadwingTraceError(f"invalid {label} tolerance")

    contract = _header(matrix_text)
    selected = set(iterations)
    rows = parse_rows(matrix_text, "QSGW head/wing trace", selected, channel)
    matrices = _matrix_groups(rows)
    velocity_components = ("velocity_x", "velocity_y", "velocity_z")
    initial_velocity_keys = {
        (component, spin, kpoint)
        for iteration, component, spin, kpoint, frequency_index in matrices
        if iteration == 0
        and component in velocity_components
        and frequency_index == -1
    }
    if not initial_velocity_keys:
        raise HeadwingTraceError("trace has no initial velocity blocks")
    expected_velocity_keys = {
        (component, spin, kpoint)
        for component, spin, kpoint in initial_velocity_keys
    }
    for spin, kpoint in {
        (spin, kpoint) for _component, spin, kpoint in initial_velocity_keys
    }:
        observed = {
            component
            for component, row_spin, row_kpoint in initial_velocity_keys
            if (row_spin, row_kpoint) == (spin, kpoint)
        }
        if observed != set(velocity_components):
            raise HeadwingTraceError(
                f"incomplete initial velocity at spin/k-point {(spin, kpoint)}"
            )

    maximum_velocity_rotation = 0.0
    maximum_velocity_hermiticity = 0.0
    maximum_rotation_unitarity = 0.0
    velocity_block_count = 0
    for iteration in iterations:
        observed_keys = {
            (component, spin, kpoint)
            for row_iteration, component, spin, kpoint, frequency_index
            in matrices
            if row_iteration == iteration
            and component in velocity_components
            and frequency_index == -1
        }
        if observed_keys != expected_velocity_keys:
            raise HeadwingTraceError(
                f"velocity block layout changed at iteration {iteration}"
            )
        for component, spin, kpoint in sorted(expected_velocity_keys):
            reference = matrices[(0, component, spin, kpoint, -1)]
            live = matrices[(iteration, component, spin, kpoint, -1)]
            if reference.shape != live.shape or reference.shape[0] != reference.shape[1]:
                raise HeadwingTraceError("velocity matrix shape is invalid")
            maximum_velocity_hermiticity = max(
                maximum_velocity_hermiticity,
                _maximum_abs(live - live.conj().T),
            )
            if iteration == 0:
                predicted = reference
            else:
                rotation_key = (iteration, "rotation_u", spin, kpoint, -1)
                if rotation_key not in matrices:
                    raise HeadwingTraceError(
                        f"missing fixed-basis rotation {rotation_key}"
                    )
                unitary = matrices[rotation_key]
                if unitary.shape != reference.shape:
                    raise HeadwingTraceError("rotation/velocity shape mismatch")
                identity = np.eye(unitary.shape[0], dtype=np.complex128)
                maximum_rotation_unitarity = max(
                    maximum_rotation_unitarity,
                    _maximum_abs(unitary.conj().T @ unitary - identity),
                )
                predicted = unitary.conj().T @ reference @ unitary
            maximum_velocity_rotation = max(
                maximum_velocity_rotation,
                _relative(live - predicted, reference, live),
            )
            velocity_block_count += 1

    head_layout: set[tuple[int, int, int]] | None = None
    head_by_iteration: dict[int, dict[tuple[int, int, int], np.ndarray]] = {}
    head_frequencies: dict[tuple[int, int, int], float] = {}
    maximum_head_hermiticity = 0.0
    for iteration in iterations:
        blocks = {
            (spin, kpoint, frequency_index): matrix
            for (
                row_iteration,
                component,
                spin,
                kpoint,
                frequency_index,
            ), matrix in matrices.items()
            if row_iteration == iteration and component == "head_tensor"
        }
        if not blocks:
            raise HeadwingTraceError(
                f"missing analytic head tensor at iteration {iteration}"
            )
        layout = set(blocks)
        if head_layout is None:
            head_layout = layout
        elif layout != head_layout:
            raise HeadwingTraceError(
                f"head tensor layout changed at iteration {iteration}"
            )
        for key, matrix in blocks.items():
            if matrix.shape != (3, 3):
                raise HeadwingTraceError("head tensor must be 3x3")
            maximum_head_hermiticity = max(
                maximum_head_hermiticity,
                _maximum_abs(matrix - matrix.conj().T),
            )
            block = (iteration, "head_tensor", key[0], key[1], key[2])
            frequencies = {
                frequency
                for row_key, (frequency, _value) in rows.items()
                if (row_key[0], row_key[2], row_key[3], row_key[4], row_key[5])
                == block
            }
            if len(frequencies) != 1:
                raise HeadwingTraceError(
                    f"inconsistent head frequency for block {block}"
                )
            frequency = next(iter(frequencies))
            if key in head_frequencies and head_frequencies[key] != frequency:
                raise HeadwingTraceError("head frequency changed across iterations")
            head_frequencies[key] = frequency
        head_by_iteration[iteration] = blocks

    assert head_layout is not None
    initial_replay = max(
        _maximum_abs(head_by_iteration[1][key] - head_by_iteration[0][key])
        for key in head_layout
    )
    post_update_change = 0.0
    for iteration in iterations[2:]:
        post_update_change = max(
            post_update_change,
            max(
                _maximum_abs(
                    head_by_iteration[iteration][key]
                    - head_by_iteration[iteration - 1][key]
                )
                for key in head_layout
            ),
        )
    live_change_passed = (
        len(iterations) < 3
        or post_update_change >= head_live_change_minimum
    )
    initial_replay_passed = initial_replay <= initial_replay_tolerance
    passed = bool(
        maximum_velocity_rotation <= invariant_tolerance
        and maximum_velocity_hermiticity <= invariant_tolerance
        and maximum_rotation_unitarity <= invariant_tolerance
        and maximum_head_hermiticity <= invariant_tolerance
        and initial_replay_passed
        and live_change_passed
    )
    return {
        "passed": passed,
        "iterations": iterations,
        "channel": channel,
        "input_contract_sha256": contract["qsgw_input_contract_sha256"],
        "velocity_rotation_formula": "U_dagger_v0_U",
        "velocity_block_count": velocity_block_count,
        "velocity_rotation_relative_residual": maximum_velocity_rotation,
        "velocity_hermiticity_max_abs": maximum_velocity_hermiticity,
        "rotation_unitarity_max_abs": maximum_rotation_unitarity,
        "head_block_count": len(head_layout) * len(iterations),
        "head_frequency_count": len({key[2] for key in head_layout}),
        "head_hermiticity_max_abs": maximum_head_hermiticity,
        "head_iteration0_to1_max_abs_change": initial_replay,
        "head_initial_replay_tolerance": initial_replay_tolerance,
        "head_initial_replay_passed": initial_replay_passed,
        "head_post_update_max_abs_change": post_update_change,
        "head_live_change_minimum": head_live_change_minimum,
        "head_live_change_passed": live_change_passed,
        "invariant_tolerance": invariant_tolerance,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("trace", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--iterations", default="0:5")
    parser.add_argument("--channel", type=int, default=0)
    parser.add_argument("--invariant-tolerance", type=float, default=1.0e-10)
    parser.add_argument(
        "--initial-replay-tolerance", type=float, default=1.0e-10
    )
    parser.add_argument(
        "--head-live-change-minimum", type=float, default=0.0
    )
    args = parser.parse_args()
    try:
        report = validate(
            args.trace.read_text(encoding="utf-8"),
            parse_iterations(args.iterations),
            channel=args.channel,
            invariant_tolerance=args.invariant_tolerance,
            initial_replay_tolerance=args.initial_replay_tolerance,
            head_live_change_minimum=args.head_live_change_minimum,
        )
        status = 0 if report["passed"] else 2
    except (OSError, ValueError) as error:
        report = {"passed": False, "error": str(error)}
        status = 2
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="ascii",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return status


if __name__ == "__main__":
    raise SystemExit(main())
