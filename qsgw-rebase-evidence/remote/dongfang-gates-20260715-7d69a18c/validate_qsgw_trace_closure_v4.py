#!/usr/bin/env python3
"""Validate internal Hamiltonian closure in a current LibRPA QSGW trace.

Revision v4 accepts current (contract version 5) traces with an enabled
Hartree delta-density update (``hartree delta_density``), validating the
``hartree_coulomb``/``hartree_normalization`` modes fail-closed, instead
of hard-rejecting any current trace whose Hartree mode is enabled. The
numerical Hartree checks (iteration-one zero reference, post-reference
response, delta_vh in closure and Hermiticity) are unchanged from v3.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections.abc import Iterable
from pathlib import Path

import numpy as np

from compare_qsgw_component_traces import (
    parse_contract,
    parse_iterations,
    parse_rows,
)


class TraceClosureError(ValueError):
    """Raised when a QSGW trace violates its numerical contract."""


MATRIX_COMPONENTS = (
    "h0",
    "vxc_dft",
    "exx",
    "vc",
    "delta_vh",
    "raw_h",
    "mixed_h",
)


def _parse_stage_one_contract(text: str) -> dict[str, str]:
    keys = {
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
    hartree_option_keys = {"hartree_coulomb", "hartree_normalization"}
    contract: dict[str, str] = {}
    for line_number, line in enumerate(text.splitlines(), 1):
        line = line.strip()
        if not line.startswith("#"):
            continue
        fields = line[1:].strip().split(None, 1)
        if len(fields) != 2 or fields[0] not in keys | hartree_option_keys:
            continue
        key, value = fields
        if key in contract:
            raise TraceClosureError(
                f"QSGW trace:{line_number}: duplicate contract key {key}"
            )
        contract[key] = value.strip()
    missing = sorted(keys - contract.keys())
    if missing:
        raise TraceClosureError(
            f"QSGW stage-one trace is missing contract keys {missing}"
        )
    expected = {
        "qsgw_contract_version": "5",
        "fixed_basis": "immutable_mf0",
        "live_update": "eigenvalues_wfc",
        "band": "disabled_stage1",
    }
    differences = {
        key: {"expected": value, "actual": contract[key]}
        for key, value in expected.items()
        if contract[key] != value
    }
    if differences:
        raise TraceClosureError(
            f"invalid QSGW stage-one trace contract: {differences}"
        )
    hartree_mode = contract["hartree"]
    if hartree_mode not in {"disabled_stage1", "delta_density"}:
        raise TraceClosureError(
            "invalid QSGW stage-one trace contract: unsupported hartree mode "
            f"{hartree_mode!r}"
        )
    hartree_enabled = hartree_mode == "delta_density"
    if hartree_enabled:
        missing_options = sorted(
            key for key in hartree_option_keys if key not in contract
        )
        if missing_options:
            raise TraceClosureError(
                "Hartree-enabled stage-one trace is missing contract keys "
                f"{missing_options}"
            )
        if contract["hartree_coulomb"] != "truncated":
            raise TraceClosureError(
                "unsupported hartree_coulomb mode "
                f"{contract['hartree_coulomb']!r}"
            )
        if (
            contract["hartree_normalization"]
            != "legacy_extra_inverse_nk"
        ):
            raise TraceClosureError(
                "unsupported hartree_normalization mode "
                f"{contract['hartree_normalization']!r}"
            )
    elif hartree_option_keys & contract.keys():
        raise TraceClosureError(
            "Hartree-disabled stage-one trace unexpectedly declares "
            "hartree_coulomb/hartree_normalization"
        )
    if contract["symmetry"] not in {
        "disabled_stage1",
        "input_kstar_live",
        "unsupported_full_bz_only",
    }:
        raise TraceClosureError(
            "invalid QSGW stage-one trace contract: unsupported symmetry mode "
            f"{contract['symmetry']!r}"
        )
    headwing_modes = {
        ("disabled_stage1", "disabled_stage1"),
        ("fixed_basis_rotation", "scf_grid_analytic_live"),
    }
    mode = (contract["velocity"], contract["headwing"])
    if mode not in headwing_modes:
        raise TraceClosureError(
            "invalid QSGW stage-one trace contract: velocity/headwing mode "
            f"{mode!r} is not a supported pair"
        )
    if contract["qsgw_mixer"] not in {"none", "linear"}:
        raise TraceClosureError("unsupported QSGW stage-one mixer")
    beta = float(contract["qsgw_mixing_beta"])
    if not math.isfinite(beta) or not 0.0 < beta <= 1.0:
        raise TraceClosureError("invalid QSGW stage-one mixing beta")
    if not contract["qsgw_input_contract"]:
        raise TraceClosureError("empty QSGW stage-one input contract path")
    if not re.fullmatch(
        r"[0-9a-fA-F]{64}", contract["qsgw_input_contract_sha256"]
    ):
        raise TraceClosureError("invalid QSGW stage-one input contract SHA256")
    return {
        **contract,
        "task": "qsgw",
        "qsgw_update_hartree": "1" if hartree_enabled else "0",
    }


def _parse_closure_contract(
    text: str, require_current_contract: bool
) -> dict[str, str]:
    version = None
    for line in text.splitlines():
        fields = line.strip().split()
        if len(fields) == 3 and fields[:2] == ["#", "qsgw_contract_version"]:
            version = fields[2]
            break
    if version == "5":
        return _parse_stage_one_contract(text)
    return parse_contract(
        text, "QSGW trace", require_current=require_current_contract
    )


def _require_continuous_iterations(iterations: Iterable[int]) -> list[int]:
    selected = sorted(set(iterations))
    if not selected or selected != list(range(selected[-1] + 1)):
        raise TraceClosureError(
            "trace closure requires continuous iterations beginning at zero"
        )
    if selected[-1] < 1:
        raise TraceClosureError("trace closure requires at least one QSGW update")
    return selected


def _matrix_blocks(rows, iteration: int, component: str) -> set[tuple[int, int]]:
    return {
        (key[3], key[4])
        for key in rows
        if key[0] == iteration and key[2] == component and key[5] == -1
    }


def _read_matrix(rows, iteration: int, component: str,
                 spin: int, kpoint: int) -> np.ndarray:
    entries = {
        (key[6], key[7]): value[1]
        for key, value in rows.items()
        if key[0] == iteration
        and key[2] == component
        and key[3] == spin
        and key[4] == kpoint
        and key[5] == -1
    }
    label = (
        f"iteration {iteration} component {component} "
        f"spin {spin} kpoint {kpoint}"
    )
    if not entries:
        raise TraceClosureError(f"missing matrix block: {label}")
    if any(row < 0 or column < 0 for row, column in entries):
        raise TraceClosureError(f"negative matrix index: {label}")
    nrows = max(row for row, _ in entries) + 1
    ncolumns = max(column for _, column in entries) + 1
    if nrows != ncolumns:
        raise TraceClosureError(f"non-square matrix block: {label}")
    expected = {(row, column) for row in range(nrows)
                for column in range(ncolumns)}
    if set(entries) != expected:
        raise TraceClosureError(f"incomplete matrix block: {label}")
    result = np.empty((nrows, ncolumns), dtype=np.complex128)
    for (row, column), value in entries.items():
        result[row, column] = value
    return result


def _materialize_upper_triangle_hermitian(matrix: np.ndarray) -> np.ndarray:
    """Match assemble_effective_hamiltonian's legacy UPLO='U' contract."""
    result = np.array(matrix, dtype=np.complex128, copy=True)
    diagonal = np.diag_indices_from(result)
    result[diagonal] = result[diagonal].real
    lower = np.tril_indices_from(result, k=-1)
    result[lower] = result.T.conj()[lower]
    return result


def _effective_matrix_for_contract(
    matrix: np.ndarray, component: str, legacy_upper_triangle: bool
) -> np.ndarray:
    if legacy_upper_triangle and component in {"raw_h", "mixed_h"}:
        return _materialize_upper_triangle_hermitian(matrix)
    return matrix


def _maximum_hermiticity_residual(
    rows, iterations, blocks, legacy_upper_triangle: bool
) -> float:
    maximum = 0.0
    for iteration in iterations:
        components = ("h0", "vxc_dft") if iteration == 0 else (
            "raw_h", "mixed_h"
        )
        if _matrix_blocks(rows, iteration, "delta_vh"):
            components += ("delta_vh",)
        for component in components:
            component_blocks = _matrix_blocks(rows, iteration, component)
            if component_blocks != blocks:
                raise TraceClosureError(
                    f"{component} matrix blocks differ from h0 at iteration "
                    f"{iteration}"
                )
            for spin, kpoint in blocks:
                value = _read_matrix(
                    rows, iteration, component, spin, kpoint
                )
                value = _effective_matrix_for_contract(
                    value, component, legacy_upper_triangle
                )
                maximum = max(
                    maximum,
                    float(np.max(np.abs(value - value.conj().T))),
                )
    return maximum


def validate_trace_text(
    text: str,
    iterations: Iterable[int],
    *,
    channel: int = 0,
    require_current_contract: bool = True,
    closure_tolerance_ha: float = 1.0e-10,
    hermiticity_tolerance_ha: float = 1.0e-10,
    hartree_reference_tolerance_ha: float = 1.0e-12,
    hartree_response_minimum_ha: float | None = None,
) -> dict[str, object]:
    selected = _require_continuous_iterations(iterations)
    if channel not in (0, 1):
        raise TraceClosureError("QSGW trace channel must be zero or one")
    for name, value in (
        ("closure", closure_tolerance_ha),
        ("Hermiticity", hermiticity_tolerance_ha),
        ("Hartree reference", hartree_reference_tolerance_ha),
    ):
        if not math.isfinite(value) or value < 0.0:
            raise TraceClosureError(f"invalid {name} tolerance")
    if hartree_response_minimum_ha is not None and (
        not math.isfinite(hartree_response_minimum_ha)
        or hartree_response_minimum_ha <= 0.0
    ):
        raise TraceClosureError("invalid Hartree response minimum")

    contract = _parse_closure_contract(text, require_current_contract)
    legacy_upper_triangle = contract["qsgw_contract_version"] == "4"
    if channel == 1 and contract["task"] != "qsgw_band":
        raise TraceClosureError("band channel requires task=qsgw_band")
    rows = parse_rows(text, "QSGW trace", set(selected), channel)
    blocks = _matrix_blocks(rows, 0, "h0")
    if not blocks:
        raise TraceClosureError("iteration zero has no h0 matrix blocks")

    hermiticity = _maximum_hermiticity_residual(
        rows, selected, blocks, legacy_upper_triangle
    )
    if hermiticity > hermiticity_tolerance_ha:
        raise TraceClosureError(
            "QSGW matrix is not Hermitian: maximum residual "
            f"{hermiticity:.6e} Ha"
        )

    hartree_enabled = int(contract["qsgw_update_hartree"]) != 0
    hartree_iteration_one = None
    hartree_post_reference = None
    if hartree_enabled:
        for iteration in selected[1:]:
            if _matrix_blocks(rows, iteration, "delta_vh") != blocks:
                raise TraceClosureError(
                    f"Hartree-enabled trace is missing delta_vh at iteration {iteration}"
                )
        hartree_iteration_one = max(
            float(np.max(np.abs(
                _read_matrix(rows, 1, "delta_vh", spin, kpoint)
            )))
            for spin, kpoint in blocks
        )
        if hartree_iteration_one > hartree_reference_tolerance_ha:
            raise TraceClosureError(
                "iteration-one Hartree delta does not vanish against the "
                f"immutable reference: {hartree_iteration_one:.6e} Ha"
            )
        if selected[-1] >= 2:
            hartree_post_reference = max(
                float(np.max(np.abs(
                    _read_matrix(
                        rows, iteration, "delta_vh", spin, kpoint
                    )
                )))
                for iteration in selected[2:]
                for spin, kpoint in blocks
            )
    else:
        for iteration in selected[1:]:
            if _matrix_blocks(rows, iteration, "delta_vh"):
                raise TraceClosureError(
                    "Hartree-disabled trace unexpectedly contains delta_vh"
                )

    if hartree_response_minimum_ha is not None:
        if not hartree_enabled:
            raise TraceClosureError(
                "post-reference Hartree response requires Hartree to be enabled"
            )
        if hartree_post_reference is None:
            raise TraceClosureError(
                "post-reference Hartree response requires iteration two"
            )
        if hartree_post_reference < hartree_response_minimum_ha:
            raise TraceClosureError(
                "post-reference Hartree response is below the required "
                f"minimum: {hartree_post_reference:.6e} Ha < "
                f"{hartree_response_minimum_ha:.6e} Ha"
            )

    effective_maximum = 0.0
    linear_maximum = 0.0
    linear_mixing = contract["qsgw_mixer"] == "linear"
    beta = float(contract["qsgw_mixing_beta"])
    if linear_mixing and not 0.0 < beta <= 1.0:
        raise TraceClosureError("invalid linear mixing beta in trace contract")

    previous_mixed = {
        block: _read_matrix(rows, 0, "h0", *block)
        for block in blocks
    }
    vxc = {
        block: _read_matrix(rows, 0, "vxc_dft", *block)
        for block in blocks
    }
    for iteration in selected[1:]:
        for component in ("exx", "vc", "raw_h", "mixed_h"):
            if _matrix_blocks(rows, iteration, component) != blocks:
                raise TraceClosureError(
                    f"{component} matrix blocks differ from h0 at iteration "
                    f"{iteration}"
                )
        for block in blocks:
            h0 = _read_matrix(rows, 0, "h0", *block)
            exx = _read_matrix(rows, iteration, "exx", *block)
            correlation = _read_matrix(rows, iteration, "vc", *block)
            raw = _read_matrix(rows, iteration, "raw_h", *block)
            mixed = _read_matrix(rows, iteration, "mixed_h", *block)
            raw = _effective_matrix_for_contract(
                raw, "raw_h", legacy_upper_triangle
            )
            mixed = _effective_matrix_for_contract(
                mixed, "mixed_h", legacy_upper_triangle
            )
            expected_raw = h0 - vxc[block] + exx + correlation
            if hartree_enabled:
                expected_raw += _read_matrix(
                    rows, iteration, "delta_vh", *block
                )
            expected_raw = _materialize_upper_triangle_hermitian(expected_raw)
            effective_maximum = max(
                effective_maximum,
                float(np.max(np.abs(raw - expected_raw))),
            )
            if linear_mixing:
                expected_mixed = (
                    (1.0 - beta) * previous_mixed[block] + beta * raw
                )
                linear_maximum = max(
                    linear_maximum,
                    float(np.max(np.abs(mixed - expected_mixed))),
                )
            previous_mixed[block] = mixed

    if effective_maximum > closure_tolerance_ha:
        raise TraceClosureError(
            "effective Hamiltonian closure failed: maximum residual "
            f"{effective_maximum:.6e} Ha"
        )
    if linear_mixing and linear_maximum > closure_tolerance_ha:
        raise TraceClosureError(
            "linear mixing recurrence failed: maximum residual "
            f"{linear_maximum:.6e} Ha"
        )

    return {
        "passed": True,
        "qsgw_contract_version": int(contract["qsgw_contract_version"]),
        "qsgw_input_contract_sha256": contract.get(
            "qsgw_input_contract_sha256"
        ),
        "effective_hamiltonian_semantics": (
            "legacy_upper_triangle_authoritative"
            if legacy_upper_triangle
            else "explicit_full_hermitian"
        ),
        "channel": channel,
        "iterations": selected,
        "matrix_block_count": len(blocks),
        "hartree_enabled": hartree_enabled,
        "effective_hamiltonian_max_abs_ha": effective_maximum,
        "linear_mixing_max_abs_ha": linear_maximum if linear_mixing else None,
        "hermiticity_max_abs_ha": hermiticity,
        "hartree_iteration_one_max_abs_ha": hartree_iteration_one,
        "hartree_post_reference_max_abs_ha": hartree_post_reference,
        "closure_tolerance_ha": closure_tolerance_ha,
        "hermiticity_tolerance_ha": hermiticity_tolerance_ha,
        "hartree_reference_tolerance_ha": hartree_reference_tolerance_ha,
        "hartree_response_minimum_ha": hartree_response_minimum_ha,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("trace", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--iterations", default="0:5")
    parser.add_argument("--channel", type=int, choices=(0, 1), default=0)
    parser.add_argument("--legacy-contract", action="store_true")
    parser.add_argument("--closure-tolerance-ha", type=float, default=1e-10)
    parser.add_argument("--hermiticity-tolerance-ha", type=float, default=1e-10)
    parser.add_argument(
        "--hartree-reference-tolerance-ha", type=float, default=1e-12
    )
    parser.add_argument("--hartree-response-minimum-ha", type=float)
    args = parser.parse_args()

    try:
        report = validate_trace_text(
            args.trace.read_text(encoding="utf-8"),
            parse_iterations(args.iterations),
            channel=args.channel,
            require_current_contract=not args.legacy_contract,
            closure_tolerance_ha=args.closure_tolerance_ha,
            hermiticity_tolerance_ha=args.hermiticity_tolerance_ha,
            hartree_reference_tolerance_ha=(
                args.hartree_reference_tolerance_ha
            ),
            hartree_response_minimum_ha=args.hartree_response_minimum_ha,
        )
        status = 0
    except (OSError, ValueError) as error:
        report = {"passed": False, "error": str(error)}
        status = 1
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return status


if __name__ == "__main__":
    raise SystemExit(main())
