#!/usr/bin/env python3
"""Validate fixed-basis diagonalization and live-state updates in a QSGW trace."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from compare_qsgw_component_traces import (
    _matrix_groups,
    parse_iterations,
    parse_rows,
)
from validate_qsgw_trace_closure import _parse_closure_contract


HA2EV = 27.211386245988


class FixedBasisError(ValueError):
    """Raised when a QSGW trace violates fixed-basis semantics."""


def _parse_eigenvalues(
    text: str, iterations: set[int], channel: int
) -> dict[tuple[int, int, int, int], float]:
    values: dict[tuple[int, int, int, int], float] = {}
    for line_number, line in enumerate(text.splitlines(), 1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        fields = stripped.split()
        if len(fields) != 9:
            raise FixedBasisError(
                f"eigenvalue trace:{line_number}: expected 9 columns"
            )
        iteration = int(fields[0])
        row_channel = int(fields[1])
        if iteration not in iterations or row_channel != channel:
            continue
        key = (iteration, int(fields[2]), int(fields[3]), int(fields[7]))
        value = float(fields[8]) / HA2EV
        if key in values:
            raise FixedBasisError(
                f"eigenvalue trace:{line_number}: duplicate key {key}"
            )
        if not math.isfinite(value):
            raise FixedBasisError(
                f"eigenvalue trace:{line_number}: non-finite energy"
            )
        values[key] = value
    if not values:
        raise FixedBasisError("eigenvalue trace has no selected rows")
    return values


def _parse_band_out(path: Path) -> dict[tuple[int, int, int], float]:
    lines = [line.split() for line in path.read_text(encoding="utf-8").splitlines()]
    if len(lines) < 5 or any(not fields for fields in lines[:5]):
        raise FixedBasisError(f"{path}: truncated band_out header")
    n_kpoints = int(lines[0][0])
    n_spins = int(lines[1][0])
    n_bands = int(lines[2][0])
    position = 5
    values: dict[tuple[int, int, int], float] = {}
    for kpoint in range(n_kpoints):
        for spin in range(n_spins):
            if position >= len(lines) or len(lines[position]) != 2:
                raise FixedBasisError(f"{path}: missing k-point/spin header")
            observed_kpoint, observed_spin = map(int, lines[position])
            position += 1
            if (observed_kpoint, observed_spin) != (kpoint + 1, spin + 1):
                raise FixedBasisError(f"{path}: unexpected k-point/spin ordering")
            for band in range(n_bands):
                if position >= len(lines) or len(lines[position]) < 4:
                    raise FixedBasisError(f"{path}: truncated band block")
                fields = lines[position]
                position += 1
                if int(fields[0]) != band + 1:
                    raise FixedBasisError(f"{path}: unexpected band ordering")
                energy_ha = float(fields[2])
                if not math.isfinite(energy_ha):
                    raise FixedBasisError(f"{path}: non-finite band energy")
                values[(spin, kpoint, band)] = energy_ha
    if any(fields for fields in lines[position:]):
        raise FixedBasisError(f"{path}: unexpected trailing rows")
    return values


def _maximum_abs(matrix: np.ndarray) -> float:
    return float(np.max(np.abs(matrix))) if matrix.size else 0.0


def _relative_frobenius(difference: np.ndarray, reference: np.ndarray) -> float:
    return float(
        np.linalg.norm(difference) / max(np.linalg.norm(reference), 1.0e-300)
    )


def validate(
    matrix_text: str,
    eigenvalue_text: str,
    iterations: list[int],
    *,
    channel: int,
    band_out: Path,
    eigenvalue_tolerance_ha: float,
    invariant_tolerance: float,
) -> dict[str, object]:
    if not iterations or iterations != list(range(iterations[-1] + 1)):
        raise FixedBasisError("iterations must be continuous and begin at zero")
    if iterations[-1] < 1:
        raise FixedBasisError("at least one QSGW update is required")
    if channel != 0:
        raise FixedBasisError("this validator currently requires the grid channel")
    for name, value in (
        ("eigenvalue", eigenvalue_tolerance_ha),
        ("invariant", invariant_tolerance),
    ):
        if not math.isfinite(value) or value < 0.0:
            raise FixedBasisError(f"invalid {name} tolerance")

    matrix_contract = _parse_closure_contract(matrix_text, True)
    eigenvalue_contract = _parse_closure_contract(eigenvalue_text, True)
    contract_keys = (
        "qsgw_contract_version",
        "fixed_basis",
        "live_update",
        "qsgw_input_contract_sha256",
        "qsgw_mixer",
        "hartree",
        "band",
    )
    differences = {
        key: (matrix_contract.get(key), eigenvalue_contract.get(key))
        for key in contract_keys
        if matrix_contract.get(key) != eigenvalue_contract.get(key)
    }
    if differences:
        raise FixedBasisError(f"trace contracts differ: {differences}")

    selected = set(iterations)
    rows = parse_rows(matrix_text, "QSGW matrix trace", selected, channel)
    matrices = _matrix_groups(rows)
    trace_eigenvalues = _parse_eigenvalues(eigenvalue_text, selected, channel)
    input_eigenvalues = _parse_band_out(band_out)

    blocks = sorted(
        (spin, kpoint)
        for iteration, component, spin, kpoint, frequency in matrices
        if iteration == 0 and component == "h0" and frequency == -1
    )
    if not blocks:
        raise FixedBasisError("iteration zero has no h0 blocks")
    wfc_components = sorted(
        {
            component
            for iteration, component, _spin, _kpoint, frequency in matrices
            if iteration == 0
            and component.startswith("wfc_spinor")
            and frequency == -1
        }
    )
    if not wfc_components:
        raise FixedBasisError("iteration zero has no wavefunction blocks")

    rotation_unitarity = 0.0
    diagonalization_offdiagonal = 0.0
    diagonalization_eigenvalue = 0.0
    trace_eigenvalue = 0.0
    input_eigenvalue = 0.0
    wfc_rotation = 0.0
    matrix_block_count = 0

    for spin, kpoint in blocks:
        h0_key = (0, "h0", spin, kpoint, -1)
        h0 = matrices[h0_key]
        if h0.shape[0] != h0.shape[1]:
            raise FixedBasisError(f"non-square h0 block {(spin, kpoint)}")
        dimension = h0.shape[0]
        reference_wfc = {
            component: matrices[(0, component, spin, kpoint, -1)]
            for component in wfc_components
        }

        for iteration in iterations:
            h_component = "h0" if iteration == 0 else "mixed_h"
            h_key = (iteration, h_component, spin, kpoint, -1)
            if h_key not in matrices:
                raise FixedBasisError(f"missing Hamiltonian block {h_key}")
            hamiltonian = matrices[h_key]
            eigenvalues = np.linalg.eigvalsh(hamiltonian)
            if eigenvalues.size != dimension:
                raise FixedBasisError(f"Hamiltonian dimension changed at {h_key}")
            for band, value in enumerate(eigenvalues):
                trace_key = (iteration, spin, kpoint, band)
                if trace_key not in trace_eigenvalues:
                    raise FixedBasisError(f"missing eigenvalue row {trace_key}")
                trace_eigenvalue = max(
                    trace_eigenvalue,
                    abs(float(value) - trace_eigenvalues[trace_key]),
                )
                if iteration == 0:
                    input_key = (spin, kpoint, band)
                    if input_key not in input_eigenvalues:
                        raise FixedBasisError(f"missing band_out row {input_key}")
                    input_eigenvalue = max(
                        input_eigenvalue,
                        abs(trace_eigenvalues[trace_key] - input_eigenvalues[input_key]),
                    )

            if iteration == 0:
                continue
            unitary_key = (iteration, "rotation_u", spin, kpoint, -1)
            if unitary_key not in matrices:
                raise FixedBasisError(f"missing rotation block {unitary_key}")
            unitary = matrices[unitary_key]
            identity = np.eye(dimension, dtype=np.complex128)
            rotation_unitarity = max(
                rotation_unitarity,
                _maximum_abs(unitary.conj().T @ unitary - identity),
            )
            diagonal = unitary.conj().T @ hamiltonian @ unitary
            diagonalization_offdiagonal = max(
                diagonalization_offdiagonal,
                _maximum_abs(diagonal - np.diag(np.diag(diagonal))),
            )
            diagonalization_eigenvalue = max(
                diagonalization_eigenvalue,
                _maximum_abs(np.diag(diagonal).real - eigenvalues),
            )
            for component, reference in reference_wfc.items():
                live_key = (iteration, component, spin, kpoint, -1)
                if live_key not in matrices:
                    raise FixedBasisError(f"missing live wavefunction block {live_key}")
                live = matrices[live_key]
                predicted = unitary.T @ reference
                if live.shape != predicted.shape:
                    raise FixedBasisError(f"wavefunction shape changed at {live_key}")
                wfc_rotation = max(
                    wfc_rotation,
                    _relative_frobenius(live - predicted, live),
                )
            matrix_block_count += 1

    passed = (
        trace_eigenvalue <= eigenvalue_tolerance_ha
        and input_eigenvalue <= eigenvalue_tolerance_ha
        and diagonalization_eigenvalue <= eigenvalue_tolerance_ha
        and rotation_unitarity <= invariant_tolerance
        and diagonalization_offdiagonal <= invariant_tolerance
        and wfc_rotation <= invariant_tolerance
    )
    return {
        "passed": passed,
        "iterations": iterations,
        "channel": channel,
        "matrix_block_count": matrix_block_count,
        "qsgw_contract_version": int(matrix_contract["qsgw_contract_version"]),
        "qsgw_input_contract_sha256": matrix_contract[
            "qsgw_input_contract_sha256"
        ],
        "rotation_unitarity_max_abs": rotation_unitarity,
        "diagonalization_offdiagonal_max_abs_ha": diagonalization_offdiagonal,
        "diagonalization_eigenvalue_max_abs_ha": diagonalization_eigenvalue,
        "trace_eigenvalue_max_abs_ha": trace_eigenvalue,
        "input_eigenvalue_max_abs_ha": input_eigenvalue,
        "fixed_basis_wfc_rotation_relative_frobenius": wfc_rotation,
        "eigenvalue_tolerance_ha": eigenvalue_tolerance_ha,
        "invariant_tolerance": invariant_tolerance,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("matrix_trace", type=Path)
    parser.add_argument("eigenvalue_trace", type=Path)
    parser.add_argument("band_out", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--iterations", default="0:1")
    parser.add_argument("--channel", type=int, default=0)
    parser.add_argument("--eigenvalue-tolerance-ha", type=float, default=1e-10)
    parser.add_argument("--invariant-tolerance", type=float, default=1e-10)
    args = parser.parse_args()
    try:
        report = validate(
            args.matrix_trace.read_text(encoding="utf-8"),
            args.eigenvalue_trace.read_text(encoding="utf-8"),
            parse_iterations(args.iterations),
            channel=args.channel,
            band_out=args.band_out,
            eigenvalue_tolerance_ha=args.eigenvalue_tolerance_ha,
            invariant_tolerance=args.invariant_tolerance,
        )
        status = 0 if report["passed"] else 1
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
