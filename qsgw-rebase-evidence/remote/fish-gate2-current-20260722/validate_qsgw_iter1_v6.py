#!/usr/bin/env python3
"""Validate current-v6 QSGW initial state and first-update invariants."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path
from types import ModuleType

import numpy as np


HA2EV = 27.211386245988


class ValidationError(ValueError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_module(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location("gate2_cmp_qsgw", path)
    if spec is None or spec.loader is None:
        raise ValidationError(f"cannot load Python module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _matrix(blocks: dict, key: tuple) -> np.ndarray:
    if key not in blocks:
        raise ValidationError(f"missing matrix block {key}")
    return np.asarray(blocks[key][1], dtype=np.complex128)


def _layout(blocks: dict, iteration: int, component: str) -> set[tuple[int, int]]:
    return {
        (key[3], key[4])
        for key in blocks
        if key[0] == iteration
        and key[1] == 0
        and key[2] == component
        and key[5] == -1
    }


def _max_abs(value: np.ndarray) -> float:
    return float(np.max(np.abs(value))) if value.size else 0.0


def _relative(difference: np.ndarray, reference: np.ndarray) -> float:
    return float(
        np.linalg.norm(difference) / max(np.linalg.norm(reference), 1.0e-300)
    )


def _parse_band_out(path: Path) -> dict[str, object]:
    rows = [line.split() for line in path.read_text(encoding="utf-8").splitlines()]
    if len(rows) < 5 or any(not row for row in rows[:5]):
        raise ValidationError("band_out header is truncated")
    nk, nspin, nband = (int(rows[index][0]) for index in range(3))
    efermi = float(rows[4][0])
    if min(nk, nspin, nband) <= 0 or not math.isfinite(efermi):
        raise ValidationError("band_out header is invalid")
    position = 5
    energies: dict[tuple[int, int, int], float] = {}
    occupations: dict[tuple[int, int, int], float] = {}
    for kpoint in range(nk):
        for spin in range(nspin):
            if position >= len(rows) or len(rows[position]) != 2:
                raise ValidationError("band_out k-point/spin header is missing")
            observed = tuple(map(int, rows[position]))
            position += 1
            if observed != (kpoint + 1, spin + 1):
                raise ValidationError("band_out k-point/spin ordering changed")
            for band in range(nband):
                if position >= len(rows) or len(rows[position]) < 4:
                    raise ValidationError("band_out band row is truncated")
                fields = rows[position]
                position += 1
                if int(fields[0]) != band + 1:
                    raise ValidationError("band_out band ordering changed")
                key = (spin, kpoint, band)
                occupations[key] = float(fields[1]) / nk
                energies[key] = float(fields[2])
    if any(row for row in rows[position:]):
        raise ValidationError("band_out has unexpected trailing rows")
    return {
        "nk": nk,
        "nspin": nspin,
        "nband": nband,
        "efermi_ha": efermi,
        "energies": energies,
        "occupations": occupations,
    }


def _require_contract(contract: dict[str, object], contract_sha: str) -> None:
    expected = {
        "qsgw_contract_version": 6,
        "fixed_basis": "immutable_mf0",
        "live_update": "eigenvalues_wfc",
        "velocity": "disabled_stage1",
        "headwing": "disabled_stage1",
        "symmetry": "exx_on_gw_on_rpa_on",
        "hartree": "disabled_stage1",
        "band": "disabled_stage1",
        "h_qsgw_cut": "disabled_non_band",
        "qsgw_mixer": "none",
        "qsgw_input_contract_sha256": contract_sha,
    }
    differences = {
        key: {"expected": value, "actual": contract.get(key)}
        for key, value in expected.items()
        if contract.get(key) != value
    }
    if differences:
        raise ValidationError(f"QSGW v6 contract mismatch: {differences}")
    if not math.isclose(
        float(contract["qsgw_mixing_beta"]), 0.2, rel_tol=0.0, abs_tol=1e-15
    ):
        raise ValidationError("configured QSGW beta differs from 0.2")


def validate(
    *,
    matrix_trace: Path,
    eigenvalue_trace: Path,
    iteration_trace: Path,
    band_out: Path,
    input_contract: Path,
    cmp_qsgw_path: Path,
    closure_tolerance_ha: float = 1e-10,
    invariant_tolerance: float = 1e-10,
    eigenvalue_tolerance_ha: float = 1e-10,
    initial_tolerance: float = 1e-10,
) -> dict[str, object]:
    tolerances = (
        closure_tolerance_ha,
        invariant_tolerance,
        eigenvalue_tolerance_ha,
        initial_tolerance,
    )
    if any(not math.isfinite(value) or value < 0.0 for value in tolerances):
        raise ValidationError("validation tolerances must be finite and nonnegative")

    cmp_qsgw = _load_module(cmp_qsgw_path)
    matrix_text = matrix_trace.read_text(encoding="utf-8")
    eigenvalue_text = eigenvalue_trace.read_text(encoding="utf-8")
    iteration_text = iteration_trace.read_text(encoding="utf-8")
    contract = cmp_qsgw._require_same_contract(
        matrix_text, eigenvalue_text, "matrix/eigenvalue traces"
    )
    cmp_qsgw._require_same_contract(
        matrix_text, iteration_text, "matrix/iteration traces"
    )
    contract_sha = _sha256(input_contract)
    _require_contract(contract, contract_sha)

    blocks = cmp_qsgw._parse_matrix_trace(matrix_text, "QSGW matrix trace")
    cmp_qsgw._validate_matrix_trajectory(blocks, contract, "QSGW matrix trace")
    eigenvalues = cmp_qsgw._parse_eigenvalue_trace(
        eigenvalue_text, "QSGW eigenvalue trace"
    )
    cmp_qsgw._validate_eigenvalue_trajectory(
        eigenvalues, contract, "QSGW eigenvalue trace"
    )
    summaries = cmp_qsgw._parse_iteration_summary(
        iteration_text, "QSGW iteration trace"
    )
    matrix_iterations = sorted({key[0] for key in blocks})
    eigenvalue_iterations = sorted({key[0] for key in eigenvalues})
    if matrix_iterations != [0, 1] or eigenvalue_iterations != [0, 1]:
        raise ValidationError("QSGW trace window must be exactly iterations 0:1")
    if sorted(summaries) != [0, 1]:
        raise ValidationError("QSGW iteration summary window must be 0:1")

    producer = _parse_band_out(band_out)
    reference_layout = _layout(blocks, 0, "h0")
    if not reference_layout or _layout(blocks, 0, "vxc_dft") != reference_layout:
        raise ValidationError("iteration-zero h0/vxc layout differs")
    update_components = ("exx", "vc", "raw_h", "mixed_h", "rotation_u")
    if any(_layout(blocks, 1, name) != reference_layout for name in update_components):
        raise ValidationError("iteration-one update layout differs from h0")
    if _layout(blocks, 1, "delta_vh"):
        raise ValidationError("Hartree-disabled Gate2 contains delta_vh")

    wfc_components = sorted(
        {
            key[2]
            for key in blocks
            if key[0] == 0
            and key[1] == 0
            and key[2].startswith("wfc_spinor")
            and key[5] == -1
        }
    )
    if not wfc_components:
        raise ValidationError("iteration zero has no wavefunction blocks")

    raw_closure = 0.0
    raw_closure_relative = 0.0
    none_mixing = 0.0
    hermiticity = 0.0
    rotation_unitarity = 0.0
    diagonalization_offdiagonal = 0.0
    diagonalization_eigenvalue = 0.0
    wfc_rotation = 0.0
    trace_eigenvalue = 0.0
    input_eigenvalue = 0.0
    input_occupation = 0.0
    electron_count_difference = 0.0
    block_dimensions: set[int] = set()

    traced_occupations: dict[int, dict[tuple[int, int, int], float]] = {
        0: {},
        1: {},
    }
    for iteration in (0, 1):
        occupation_layout = _layout(blocks, iteration, "occupation")
        if occupation_layout != reference_layout:
            raise ValidationError("occupation layout differs from h0")
        for spin, kpoint in sorted(reference_layout):
            occupation = _matrix(
                blocks, (iteration, 0, "occupation", spin, kpoint, -1)
            )
            if occupation.shape[0] != 1 or _max_abs(occupation.imag) > 1e-14:
                raise ValidationError("occupation block is not a real row vector")
            for band, value in enumerate(occupation.real[0]):
                traced_occupations[iteration][(spin, kpoint, band)] = float(value)
        count = sum(traced_occupations[iteration].values())
        electron_count_difference = max(
            electron_count_difference,
            abs(count - float(summaries[iteration]["electron_count"])),
        )

    producer_occupations = producer["occupations"]
    if traced_occupations[0].keys() != producer_occupations.keys():
        raise ValidationError("iteration-zero occupation state set changed")
    input_occupation = max(
        abs(traced_occupations[0][key] - producer_occupations[key])
        for key in producer_occupations
    )
    efermi_difference = abs(
        float(summaries[0]["efermi_eV"]) / HA2EV
        - float(producer["efermi_ha"])
    )

    for spin, kpoint in sorted(reference_layout):
        h0 = _matrix(blocks, (0, 0, "h0", spin, kpoint, -1))
        vxc = _matrix(blocks, (0, 0, "vxc_dft", spin, kpoint, -1))
        exx = _matrix(blocks, (1, 0, "exx", spin, kpoint, -1))
        correlation = _matrix(blocks, (1, 0, "vc", spin, kpoint, -1))
        raw = _matrix(blocks, (1, 0, "raw_h", spin, kpoint, -1))
        mixed = _matrix(blocks, (1, 0, "mixed_h", spin, kpoint, -1))
        rotation = _matrix(blocks, (1, 0, "rotation_u", spin, kpoint, -1))
        dimension = h0.shape[0]
        if any(matrix.shape != h0.shape for matrix in (vxc, exx, correlation, raw, mixed, rotation)):
            raise ValidationError("Hamiltonian block shapes differ")
        block_dimensions.add(dimension)
        expected_raw = h0 - vxc + exx + correlation
        raw_closure = max(raw_closure, _max_abs(raw - expected_raw))
        raw_closure_relative = max(
            raw_closure_relative, _relative(raw - expected_raw, expected_raw)
        )
        none_mixing = max(none_mixing, _max_abs(mixed - raw))
        for matrix in (h0, vxc, exx, correlation, raw, mixed):
            hermiticity = max(hermiticity, _max_abs(matrix - matrix.conj().T))
        identity = np.eye(dimension, dtype=np.complex128)
        rotation_unitarity = max(
            rotation_unitarity,
            _max_abs(rotation.conj().T @ rotation - identity),
        )
        diagonal = rotation.conj().T @ mixed @ rotation
        diagonalization_offdiagonal = max(
            diagonalization_offdiagonal,
            _max_abs(diagonal - np.diag(np.diag(diagonal))),
        )
        solved = np.linalg.eigvalsh(mixed)
        diagonalization_eigenvalue = max(
            diagonalization_eigenvalue,
            _max_abs(np.diag(diagonal).real - solved),
        )
        for iteration, expected_values in ((0, np.linalg.eigvalsh(h0)), (1, solved)):
            for band, value in enumerate(expected_values):
                key = (iteration, 0, spin, kpoint, band)
                if key not in eigenvalues:
                    raise ValidationError(f"missing eigenvalue trace row {key}")
                trace_value = float(eigenvalues[key][1]) / HA2EV
                trace_eigenvalue = max(trace_eigenvalue, abs(trace_value - value))
                if iteration == 0:
                    producer_key = (spin, kpoint, band)
                    input_eigenvalue = max(
                        input_eigenvalue,
                        abs(trace_value - producer["energies"][producer_key]),
                    )
        for component in wfc_components:
            reference = _matrix(blocks, (0, 0, component, spin, kpoint, -1))
            live = _matrix(blocks, (1, 0, component, spin, kpoint, -1))
            predicted = rotation.T @ reference
            if live.shape != predicted.shape:
                raise ValidationError("live wavefunction shape changed")
            wfc_rotation = max(
                wfc_rotation, _relative(live - predicted, live)
            )

    passed = (
        raw_closure <= closure_tolerance_ha
        and raw_closure_relative <= 1e-8
        and none_mixing <= closure_tolerance_ha
        and hermiticity <= invariant_tolerance
        and rotation_unitarity <= invariant_tolerance
        and diagonalization_offdiagonal <= invariant_tolerance
        and diagonalization_eigenvalue <= eigenvalue_tolerance_ha
        and wfc_rotation <= invariant_tolerance
        and trace_eigenvalue <= eigenvalue_tolerance_ha
        and input_eigenvalue <= eigenvalue_tolerance_ha
        and input_occupation <= initial_tolerance
        and efermi_difference <= initial_tolerance
        and electron_count_difference <= initial_tolerance
    )
    return {
        "schema": "librpa-qsgw-iter1-v6-invariants-v1",
        "passed": passed,
        "semantic_iteration_zero": "immutable_initial_state",
        "semantic_first_self_energy": "trace_iteration_1_channel_0",
        "iterations": [0, 1],
        "qsgw_contract_version": contract["qsgw_contract_version"],
        "qsgw_input_contract_sha256": contract_sha,
        "matrix_block_count": len(reference_layout),
        "matrix_dimensions": sorted(block_dimensions),
        "raw_h_closure_max_abs_ha": raw_closure,
        "raw_h_closure_relative_frobenius": raw_closure_relative,
        "none_mixer_max_abs_ha": none_mixing,
        "hermiticity_max_abs_ha": hermiticity,
        "rotation_unitarity_max_abs": rotation_unitarity,
        "diagonalization_offdiagonal_max_abs_ha": diagonalization_offdiagonal,
        "diagonalization_eigenvalue_max_abs_ha": diagonalization_eigenvalue,
        "fixed_basis_wfc_rotation_relative_frobenius": wfc_rotation,
        "trace_eigenvalue_max_abs_ha": trace_eigenvalue,
        "input_eigenvalue_max_abs_ha": input_eigenvalue,
        "input_occupation_max_abs": input_occupation,
        "input_efermi_max_abs_ha": efermi_difference,
        "trace_summary_electron_count_max_abs": electron_count_difference,
        "tolerances": {
            "closure_max_abs_ha": closure_tolerance_ha,
            "closure_relative_frobenius": 1e-8,
            "invariant": invariant_tolerance,
            "eigenvalue_ha": eigenvalue_tolerance_ha,
            "initial_state": initial_tolerance,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("matrix_trace", type=Path)
    parser.add_argument("eigenvalue_trace", type=Path)
    parser.add_argument("iteration_trace", type=Path)
    parser.add_argument("band_out", type=Path)
    parser.add_argument("input_contract", type=Path)
    parser.add_argument("cmp_qsgw", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--closure-tolerance-ha", type=float, default=1e-10)
    parser.add_argument("--invariant-tolerance", type=float, default=1e-10)
    parser.add_argument("--eigenvalue-tolerance-ha", type=float, default=1e-10)
    parser.add_argument("--initial-tolerance", type=float, default=1e-10)
    args = parser.parse_args()
    try:
        report = validate(
            matrix_trace=args.matrix_trace,
            eigenvalue_trace=args.eigenvalue_trace,
            iteration_trace=args.iteration_trace,
            band_out=args.band_out,
            input_contract=args.input_contract,
            cmp_qsgw_path=args.cmp_qsgw,
            closure_tolerance_ha=args.closure_tolerance_ha,
            invariant_tolerance=args.invariant_tolerance,
            eigenvalue_tolerance_ha=args.eigenvalue_tolerance_ha,
            initial_tolerance=args.initial_tolerance,
        )
        status = 0 if report["passed"] else 2
    except (OSError, ValueError) as error:
        report = {
            "schema": "librpa-qsgw-iter1-v6-invariants-v1",
            "passed": False,
            "error": str(error),
        }
        status = 1
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return status


if __name__ == "__main__":
    raise SystemExit(main())
