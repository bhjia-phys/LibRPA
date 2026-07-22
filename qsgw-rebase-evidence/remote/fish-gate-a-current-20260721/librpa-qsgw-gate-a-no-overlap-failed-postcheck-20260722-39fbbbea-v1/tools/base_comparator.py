#!/usr/bin/env python3
"""Compare legacy and upstream-port QSGW traces component by component."""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import numpy as np


HA2EV = 27.211386245988


PHYSICAL_TOLERANCES = {
    "h0": 1.0e-8,
    "vxc_dft": 1.0e-8,
    "sigma_c_iw": 1.0e-8,
    "exx": 1.0e-8,
    "vc": 1.0e-8,
    "delta_vh": 1.0e-8,
    "raw_h": 1.0e-8,
    "mixed_h": 1.0e-9,
    "occupation": 1.0e-12,
    "fermi_energy_ha": 1.0e-9,
    "electron_count": 1.0e-10,
    "gap_ha": 1.0e-9,
}

CONTRACT_KEYS = (
    "qsgw_contract_version",
    "task",
    "fixed_basis",
    "qsgw_mixer",
    "qsgw_mixing_beta",
    "qsgw_min_iter",
    "qsgw_max_iter",
    "starting_vxc",
    "vxc_basis",
    "qsgw_update_hartree",
    "qsgw_hartree_coulomb",
    "qsgw_hartree_normalization",
    "use_symmetry_gw",
    "use_symmetry_exx",
    "replace_w_head",
    "option_dielect_func",
    "nfreq",
    "n_params_anacon",
    "n_params_anacon_resample",
    "anacon_nfreq",
    "anacon_tfgrids_type",
    "use_shrink_abfs",
    "use_fullcoul_exx",
    "use_fullcoul_eps",
    "use_fullcoul_wc",
    "constants_choice",
    "ac_policy",
)

CURRENT_SHARED_CONTRACT_KEYS = (
    "qsgw_mixing_history",
    "qsgw_convergence_tolerance_ev",
    "qsgw_vxc_scf_manifest",
)

CURRENT_HEADWING_CONTRACT_KEYS = (
    "velocity_basis_max_relative_wfc_residual",
    "velocity_basis_max_phase_deviation",
)

BAND_ONLY_CONTRACT_KEY = "qsgw_vxc_band_manifest"
ALL_CONTRACT_KEYS = frozenset(
    CONTRACT_KEYS
    + CURRENT_SHARED_CONTRACT_KEYS
    + CURRENT_HEADWING_CONTRACT_KEYS
    + (BAND_ONLY_CONTRACT_KEY,)
)


def parse_iterations(spec: str) -> list[int]:
    if ":" in spec:
        start_text, stop_text = spec.split(":", 1)
        start = int(start_text)
        stop = int(stop_text)
        if start < 0 or stop < start:
            raise ValueError("invalid iteration range")
        return list(range(start, stop + 1))
    values = sorted({int(value) for value in spec.split(",")})
    if not values or values[0] < 0:
        raise ValueError("invalid iteration list")
    return values


def parse_contract(
    text: str,
    label: str,
    *,
    require_current: bool = False,
) -> dict[str, str]:
    contract: dict[str, str] = {}
    for line_number, line in enumerate(text.splitlines(), 1):
        line = line.strip()
        if not line.startswith("#"):
            continue
        fields = line[1:].strip().split(None, 1)
        if len(fields) != 2 or fields[0] not in ALL_CONTRACT_KEYS:
            continue
        key, value = fields
        if key in contract:
            raise ValueError(f"{label}:{line_number}: duplicate contract key {key}")
        contract[key] = value.strip()
    required = list(CONTRACT_KEYS)
    if require_current:
        required.extend(CURRENT_SHARED_CONTRACT_KEYS)
    missing = [key for key in required if key not in contract]
    if missing:
        raise ValueError(f"{label}: missing contract keys {missing}")
    if int(contract["qsgw_contract_version"]) != 4:
        raise ValueError(f"{label}: unsupported qsgw_contract_version")
    if contract["starting_vxc"] != "dft_only":
        raise ValueError(f"{label}: invalid starting_vxc contract")
    if contract["vxc_basis"] != "fixed_state":
        raise ValueError(f"{label}: invalid vxc_basis contract")
    if require_current:
        try:
            mixing_history = int(contract["qsgw_mixing_history"])
            convergence_tolerance = float(
                contract["qsgw_convergence_tolerance_ev"]
            )
        except ValueError as error:
            raise ValueError(
                f"{label}: invalid current QSGW contract value"
            ) from error
        if mixing_history < 2:
            raise ValueError(f"{label}: invalid qsgw_mixing_history contract")
        if not math.isfinite(convergence_tolerance) or convergence_tolerance <= 0.0:
            raise ValueError(
                f"{label}: invalid qsgw_convergence_tolerance_ev contract"
            )
        if not contract["qsgw_vxc_scf_manifest"]:
            raise ValueError(f"{label}: empty SCF Vxc manifest contract")
        if contract["task"] == "qsgw_band" and not contract.get(
            BAND_ONLY_CONTRACT_KEY
        ):
            raise ValueError(f"{label}: missing band Vxc manifest contract")

        needs_velocity_alignment = (
            int(contract["replace_w_head"]) == 1
            and int(contract["option_dielect_func"]) in (3, 4)
        )
        if needs_velocity_alignment:
            missing_velocity = [
                key
                for key in CURRENT_HEADWING_CONTRACT_KEYS
                if key not in contract
            ]
            if missing_velocity:
                raise ValueError(
                    f"{label}: missing velocity-basis contract keys "
                    f"{missing_velocity}"
                )
            residual = float(
                contract["velocity_basis_max_relative_wfc_residual"]
            )
            phase_deviation = float(
                contract["velocity_basis_max_phase_deviation"]
            )
            if not math.isfinite(residual) or not 0.0 <= residual <= 1.0e-8:
                raise ValueError(
                    f"{label}: invalid velocity-basis WFC residual contract"
                )
            if (
                not math.isfinite(phase_deviation)
                or not 0.0 <= phase_deviation <= 2.0 + 1.0e-12
            ):
                raise ValueError(
                    f"{label}: invalid velocity-basis phase contract"
                )
    return contract


def _normalized_contract_value(
    contract: dict[str, str], key: str
) -> object:
    value = contract[key]
    if key in {
        "qsgw_contract_version", "qsgw_min_iter", "qsgw_max_iter",
        "qsgw_update_hartree", "use_symmetry_gw", "use_symmetry_exx",
        "replace_w_head", "option_dielect_func", "nfreq",
        "n_params_anacon", "n_params_anacon_resample", "anacon_nfreq",
        "anacon_tfgrids_type", "use_shrink_abfs", "use_fullcoul_exx",
        "use_fullcoul_eps", "use_fullcoul_wc",
        "qsgw_mixing_history",
    }:
        integer = int(value)
        if key == "n_params_anacon" and integer == -1:
            return int(contract["nfreq"])
        return integer
    if key in {
        "qsgw_mixing_beta",
        "qsgw_convergence_tolerance_ev",
        "velocity_basis_max_relative_wfc_residual",
        "velocity_basis_max_phase_deviation",
    }:
        return float(value)
    return value


def compare_contracts(
    old: dict[str, str],
    new: dict[str, str],
    ignored_keys: frozenset[str] = frozenset(),
    keys: tuple[str, ...] = CONTRACT_KEYS,
) -> dict[str, object]:
    differences: dict[str, dict[str, object]] = {}
    hartree_detail_keys = {
        "qsgw_hartree_coulomb", "qsgw_hartree_normalization"
    }
    for key in keys:
        if key in ignored_keys or key in hartree_detail_keys:
            continue
        if key not in old or key not in new:
            differences[key] = {
                "old": old.get(key),
                "new": new.get(key),
            }
            continue
        old_value = _normalized_contract_value(old, key)
        new_value = _normalized_contract_value(new, key)
        equal = (
            math.isclose(old_value, new_value, rel_tol=0.0, abs_tol=1.0e-15)
            if isinstance(old_value, float) and isinstance(new_value, float)
            else old_value == new_value
        )
        if not equal:
            differences[key] = {"old": old_value, "new": new_value}

    if int(old["qsgw_update_hartree"]):
        for key in sorted(hartree_detail_keys - ignored_keys):
            if key not in old or key not in new:
                differences[key] = {
                    "old": old.get(key), "new": new.get(key)
                }
            elif old[key] != new[key]:
                differences[key] = {"old": old[key], "new": new[key]}
    return {"passed": not differences, "differences": differences}


RowKey = tuple[int, int, str, int, int, int, int, int]
RowValue = tuple[float, complex]

ITERATION_ZERO_COMPONENTS = {
    "h0", "vxc_dft",
}
ITERATIVE_COMPONENTS = {
    "sigma_c_iw", "exx", "vc", "raw_h", "mixed_h", "rotation_u",
}
GRID_AUXILIARY_COMPONENTS = {
    "occupation", "fermi_energy_ha", "electron_count", "gap_ha",
}
VELOCITY_COMPONENTS = frozenset({"velocity_x", "velocity_y", "velocity_z"})


def _velocity_required(contract: dict[str, str]) -> bool:
    return (
        int(contract["replace_w_head"]) != 0 and
        int(contract["option_dielect_func"]) in (3, 4)
    )


def _drop_optional_zero_velocity_rows(
    rows: dict[RowKey, RowValue],
    contract: dict[str, str],
) -> tuple[dict[RowKey, RowValue], int]:
    if _velocity_required(contract):
        return rows, 0
    velocity_keys = {
        key for key in rows if key[2] in VELOCITY_COMPONENTS
    }
    if not velocity_keys or any(rows[key][1] != 0.0 for key in velocity_keys):
        return rows, 0
    return {
        key: value for key, value in rows.items() if key not in velocity_keys
    }, len(velocity_keys)


def parse_rows(
    text: str, label: str, iterations: set[int], channel: int
) -> dict[RowKey, RowValue]:
    rows: dict[RowKey, RowValue] = {}
    for line_number, line in enumerate(text.splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) != 11:
            raise ValueError(f"{label}:{line_number}: expected 11 columns")
        iteration = int(fields[0])
        row_channel = int(fields[1])
        if iteration not in iterations or row_channel != channel:
            continue
        key: RowKey = (
            iteration, row_channel, fields[2], int(fields[3]),
            int(fields[4]), int(fields[5]), int(fields[7]), int(fields[8]),
        )
        frequency = float(fields[6])
        value = complex(float(fields[9]), float(fields[10]))
        if key in rows:
            raise ValueError(f"{label}:{line_number}: duplicate row key {key}")
        if not math.isfinite(frequency) or not (
            math.isfinite(value.real) and math.isfinite(value.imag)
        ):
            raise ValueError(f"{label}:{line_number}: non-finite trace value")
        rows[key] = (frequency, value)
    if not rows:
        raise ValueError(f"{label}: no selected trace rows")
    return rows


def validate_iteration_coverage(
    rows: dict[RowKey, RowValue],
    label: str,
    iterations: set[int],
    contract: dict[str, str],
) -> None:
    observed = {key[0] for key in rows}
    missing_iterations = sorted(iterations - observed)
    if missing_iterations:
        raise ValueError(
            f"{label}: missing selected iterations {missing_iterations}"
        )

    initial_wfc_components = {
        key[2]
        for key in rows
        if key[0] == 0 and key[2].startswith("wfc_spinor")
    }
    if not initial_wfc_components:
        raise ValueError(f"{label}: iteration 0 has no wavefunction components")

    channels = {key[1] for key in rows}
    if len(channels) != 1:
        raise ValueError(f"{label}: selected trace rows mix channels")
    is_grid_channel = next(iter(channels)) == 0

    hartree_enabled = int(contract["qsgw_update_hartree"]) != 0
    velocity_required = _velocity_required(contract)
    for iteration in sorted(iterations):
        components = {key[2] for key in rows if key[0] == iteration}
        required = set(
            ITERATION_ZERO_COMPONENTS
            if iteration == 0
            else ITERATIVE_COMPONENTS
        )
        if is_grid_channel:
            required.update(GRID_AUXILIARY_COMPONENTS)
        if iteration > 0 and hartree_enabled:
            required.add("delta_vh")
        if is_grid_channel and velocity_required:
            required.update({"velocity_x", "velocity_y", "velocity_z"})
        missing = sorted(required - components)
        if missing:
            raise ValueError(
                f"{label}: iteration {iteration} missing components {missing}"
            )
        wfc_components = {
            component
            for component in components
            if component.startswith("wfc_spinor")
        }
        if wfc_components != initial_wfc_components:
            raise ValueError(
                f"{label}: iteration {iteration} wavefunction component set differs"
            )


def _matrix_groups(
    rows: dict[RowKey, RowValue]
) -> dict[tuple[int, str, int, int, int], np.ndarray]:
    grouped: dict[tuple[int, str, int, int, int], list[tuple[int, int, complex]]] = \
        defaultdict(list)
    for key, (_frequency, value) in rows.items():
        iteration, _channel, component, spin, kpoint, frequency_index, row, column = key
        grouped[(iteration, component, spin, kpoint, frequency_index)].append(
            (row, column, value)
        )

    matrices: dict[tuple[int, str, int, int, int], np.ndarray] = {}
    for key, entries in grouped.items():
        nrows = max(entry[0] for entry in entries) + 1
        ncols = max(entry[1] for entry in entries) + 1
        if len(entries) != nrows * ncols:
            raise ValueError(f"incomplete matrix block {key}")
        matrix = np.empty((nrows, ncols), dtype=np.complex128)
        occupied: set[tuple[int, int]] = set()
        for row, column, value in entries:
            if (row, column) in occupied:
                raise ValueError(f"duplicate matrix element in block {key}")
            occupied.add((row, column))
            matrix[row, column] = value
        matrices[key] = matrix
    return matrices


def validate_auxiliary_layout(
    rows: dict[RowKey, RowValue],
    label: str,
    iterations: set[int],
) -> None:
    matrices = _matrix_groups(rows)
    baseline = {
        (spin, kpoint): matrix.shape
        for (iteration, component, spin, kpoint, frequency_index), matrix
        in matrices.items()
        if iteration == 0 and component == "h0" and frequency_index == -1
    }
    if not baseline or any(rows != columns for rows, columns in baseline.values()):
        raise ValueError(f"{label}: iteration 0 h0 layout is invalid")

    channels = {key[1] for key in rows}
    if len(channels) != 1:
        raise ValueError(f"{label}: selected trace rows mix channels")
    if next(iter(channels)) != 0:
        return

    scalar_components = {
        "fermi_energy_ha", "electron_count", "gap_ha"
    }
    for iteration in sorted(iterations):
        occupation = {
            (spin, kpoint): matrix
            for (row_iteration, component, spin, kpoint, frequency_index), matrix
            in matrices.items()
            if row_iteration == iteration and component == "occupation" and
            frequency_index == -1
        }
        if set(occupation) != set(baseline):
            raise ValueError(
                f"{label}: iteration {iteration} occupation block layout differs"
            )
        for block, matrix in occupation.items():
            if matrix.shape != (1, baseline[block][1]):
                raise ValueError(
                    f"{label}: iteration {iteration} occupation must be 1xnband"
                )
            if np.any(matrix.imag != 0.0) or np.any(matrix.real < -1.0e-14):
                raise ValueError(
                    f"{label}: iteration {iteration} occupation must be real and nonnegative"
                )

        for component in scalar_components:
            blocks = {
                (spin, kpoint, frequency_index): matrix
                for (
                    row_iteration, row_component, spin, kpoint,
                    frequency_index,
                ), matrix in matrices.items()
                if row_iteration == iteration and row_component == component
            }
            if set(blocks) != {(0, 0, -1)} or \
                    blocks[(0, 0, -1)].shape != (1, 1):
                raise ValueError(
                    f"{label}: iteration {iteration} {component} must be a real 1x1 scalar"
                )
            value = blocks[(0, 0, -1)][0, 0]
            if value.imag != 0.0:
                raise ValueError(
                    f"{label}: iteration {iteration} {component} must be real"
                )
            if component in {"electron_count", "gap_ha"} and \
                    value.real < -1.0e-14:
                raise ValueError(
                    f"{label}: iteration {iteration} {component} must be nonnegative"
                )


def _component_metrics(
    old_rows: dict[RowKey, RowValue],
    new_rows: dict[RowKey, RowValue],
    frequency_tolerance: float,
) -> dict[str, dict[str, float | int]]:
    state_components = {
        key[2] for key in old_rows
        if key[2].startswith("wfc_spinor") or
        key[2].startswith("velocity_") or key[2] == "rotation_u"
    }
    physical_keys = {
        key for key in old_rows
        if key[2] not in state_components
    }
    if physical_keys != {
        key for key in new_rows if key[2] not in state_components
    }:
        raise ValueError("physical component row keys differ")

    accumulators: dict[str, dict[str, float | int]] = {}
    for key in sorted(physical_keys):
        old_frequency, old_value = old_rows[key]
        new_frequency, new_value = new_rows[key]
        if abs(old_frequency - new_frequency) > frequency_tolerance:
            raise ValueError(f"frequency differs for row {key}")
        component = key[2]
        difference = abs(old_value - new_value)
        entry = accumulators.setdefault(
            component, {"count": 0, "max_abs_diff": 0.0, "sum_square": 0.0}
        )
        entry["count"] = int(entry["count"]) + 1
        entry["max_abs_diff"] = max(float(entry["max_abs_diff"]), difference)
        entry["sum_square"] = float(entry["sum_square"]) + difference * difference

    metrics: dict[str, dict[str, float | int]] = {}
    for component, entry in accumulators.items():
        count = int(entry["count"])
        tolerance = PHYSICAL_TOLERANCES.get(component, 1.0e-8)
        maximum = float(entry["max_abs_diff"])
        metrics[component] = {
            "count": count,
            "max_abs_diff": maximum,
            "rms_diff": math.sqrt(float(entry["sum_square"]) / count),
            "tolerance": tolerance,
            "passed": maximum <= tolerance,
        }
    required = {
        "h0", "vxc_dft", "sigma_c_iw", "exx", "vc", "raw_h", "mixed_h",
    }
    channels = {key[1] for key in old_rows}
    if len(channels) != 1:
        raise ValueError("selected trace rows mix channels")
    if next(iter(channels)) == 0:
        required.update(GRID_AUXILIARY_COMPONENTS)
    missing = sorted(required - set(metrics))
    if missing:
        raise ValueError(f"missing physical trace components {missing}")
    return metrics


def _degenerate_groups(eigenvalues: np.ndarray, tolerance: float) -> list[slice]:
    groups: list[slice] = []
    start = 0
    for index in range(1, eigenvalues.size):
        if abs(eigenvalues[index] - eigenvalues[index - 1]) > tolerance:
            groups.append(slice(start, index))
            start = index
    groups.append(slice(start, eigenvalues.size))
    return groups


def _relative_norm(difference: np.ndarray, *references: np.ndarray) -> float:
    scale = max([np.linalg.norm(reference) for reference in references] + [1.0e-30])
    return float(np.linalg.norm(difference) / scale)


LEGACY_UPPER_TRIANGLE_COMPONENTS = frozenset({"raw_h", "mixed_h"})


def _legacy_matrix_for_comparison(
    component: str, matrix: np.ndarray
) -> np.ndarray:
    if component not in LEGACY_UPPER_TRIANGLE_COMPONENTS:
        return matrix
    result = np.array(matrix, dtype=np.complex128, copy=True)
    diagonal = np.diag_indices_from(result)
    result[diagonal] = result[diagonal].real
    lower = np.tril_indices_from(result, k=-1)
    result[lower] = result.T.conj()[lower]
    return result


def _state_metrics(
    old_rows: dict[RowKey, RowValue],
    new_rows: dict[RowKey, RowValue],
    eigenvalue_tolerance: float,
    degeneracy_tolerance: float,
    state_tolerance: float,
    legacy_upper_triangle: bool = False,
) -> dict[str, float | int | bool]:
    old_matrices = _matrix_groups(old_rows)
    new_matrices = _matrix_groups(new_rows)
    old_components = {key[1] for key in old_matrices}
    new_components = {key[1] for key in new_matrices}
    state_components = {
        component for component in old_components
        if component.startswith("wfc_spinor") or
        component.startswith("velocity_") or component == "rotation_u"
    }
    if state_components != {
        component for component in new_components
        if component.startswith("wfc_spinor") or
        component.startswith("velocity_") or component == "rotation_u"
    }:
        raise ValueError("old/new state component sets differ")
    wfc_components = sorted(
        component for component in state_components
        if component.startswith("wfc_spinor")
    )
    if not wfc_components:
        raise ValueError("trace has no wavefunction components")
    velocity_components = ["velocity_x", "velocity_y", "velocity_z"]
    has_velocity = any(component in state_components for component in velocity_components)
    if has_velocity and not all(component in state_components for component in velocity_components):
        raise ValueError("trace has incomplete velocity components")

    state_blocks = sorted({
        (key[0], key[2], key[3]) for key in old_matrices
        if key[1] == wfc_components[0] and key[4] == -1
    })
    if not state_blocks:
        raise ValueError("trace has no wavefunction blocks")

    max_eigenvalue = 0.0
    max_wfc = 0.0
    max_unitarity = 0.0
    max_rotation = 0.0
    max_velocity = 0.0
    for iteration, spin, kpoint in state_blocks:
        h_component = "h0" if iteration == 0 else "mixed_h"
        matrix_key = (iteration, h_component, spin, kpoint, -1)
        if matrix_key not in old_matrices or matrix_key not in new_matrices:
            raise ValueError(f"missing state Hamiltonian block {matrix_key}")
        old_hamiltonian = old_matrices[matrix_key]
        if legacy_upper_triangle:
            old_hamiltonian = _legacy_matrix_for_comparison(
                h_component, old_hamiltonian
            )
        old_eigenvalues = np.linalg.eigvalsh(old_hamiltonian)
        new_eigenvalues = np.linalg.eigvalsh(new_matrices[matrix_key])
        if old_eigenvalues.shape != new_eigenvalues.shape:
            raise ValueError(f"state Hamiltonian dimensions differ for {matrix_key}")
        max_eigenvalue = max(
            max_eigenvalue,
            float(np.max(np.abs(old_eigenvalues - new_eigenvalues))),
        )

        old_wfc = np.concatenate([
            old_matrices[(iteration, component, spin, kpoint, -1)]
            for component in wfc_components
        ], axis=1)
        new_wfc = np.concatenate([
            new_matrices[(iteration, component, spin, kpoint, -1)]
            for component in wfc_components
        ], axis=1)
        if old_wfc.shape != new_wfc.shape or old_wfc.shape[0] != old_eigenvalues.size:
            raise ValueError(f"wavefunction shape differs for {(iteration, spin, kpoint)}")

        rotation = np.zeros(
            (old_eigenvalues.size, old_eigenvalues.size), dtype=np.complex128
        )
        for group in _degenerate_groups(old_eigenvalues, degeneracy_tolerance):
            old_group = old_wfc[group, :]
            new_group = new_wfc[group, :]
            least_squares = new_group @ np.linalg.pinv(old_group)
            left, _singular_values, right = np.linalg.svd(least_squares)
            rotation[group, group] = left @ right
        identity = np.eye(rotation.shape[0], dtype=np.complex128)
        max_unitarity = max(
            max_unitarity,
            _relative_norm(rotation.conj().T @ rotation - identity, identity),
        )
        max_wfc = max(
            max_wfc,
            _relative_norm(new_wfc - rotation @ old_wfc, old_wfc, new_wfc),
        )

        if iteration > 0:
            rotation_key = (iteration, "rotation_u", spin, kpoint, -1)
            old_unitary = old_matrices[rotation_key]
            new_unitary = new_matrices[rotation_key]
            predicted = old_unitary @ rotation.T
            max_rotation = max(
                max_rotation,
                _relative_norm(new_unitary - predicted, old_unitary, new_unitary),
            )

        if has_velocity:
            for component in velocity_components:
                velocity_key = (iteration, component, spin, kpoint, -1)
                old_velocity = old_matrices[velocity_key]
                new_velocity = new_matrices[velocity_key]
                predicted = rotation.conj() @ old_velocity @ rotation.T
                max_velocity = max(
                    max_velocity,
                    _relative_norm(
                        new_velocity - predicted, old_velocity, new_velocity
                    ),
                )

    return {
        "block_count": len(state_blocks),
        "max_eigenvalue_abs_diff_ha": max_eigenvalue,
        "eigenvalue_tolerance_ha": eigenvalue_tolerance,
        "max_wfc_relative_residual": max_wfc,
        "max_alignment_unitarity_residual": max_unitarity,
        "max_rotation_relative_residual": max_rotation,
        "max_velocity_relative_residual": max_velocity,
        "state_tolerance": state_tolerance,
        "passed": (
            max_eigenvalue <= eigenvalue_tolerance and
            max_wfc <= state_tolerance and
            max_unitarity <= state_tolerance and
            max_rotation <= state_tolerance and
            max_velocity <= state_tolerance
        ),
    }


def _parse_header_values(
    text: str, label: str, keys: frozenset[str]
) -> dict[str, str]:
    values: dict[str, str] = {}
    for line_number, line in enumerate(text.splitlines(), 1):
        line = line.strip()
        if not line.startswith("#"):
            continue
        fields = line[1:].strip().split(None, 1)
        if len(fields) != 2 or fields[0] not in keys:
            continue
        key, value = fields
        if key in values:
            raise ValueError(f"{label}:{line_number}: duplicate header key {key}")
        values[key] = value.strip()
    missing = sorted(keys - values.keys())
    if missing:
        raise ValueError(f"{label}: missing header keys {missing}")
    return values


CURRENT_V5_KEYS = frozenset({
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
})


def _parse_current_v5_header(text: str, label: str) -> dict[str, str]:
    values = _parse_header_values(text, label, CURRENT_V5_KEYS)
    expected = {
        "qsgw_contract_version": "5",
        "fixed_basis": "immutable_mf0",
        "live_update": "eigenvalues_wfc",
        "velocity": "disabled_stage1",
        "headwing": "disabled_stage1",
        "symmetry": "unsupported_full_bz_only",
        "hartree": "disabled_stage1",
        "band": "disabled_stage1",
        "qsgw_mixer": "none",
    }
    differences = {
        key: {"expected": expected_value, "actual": values[key]}
        for key, expected_value in expected.items()
        if values[key] != expected_value
    }
    if differences:
        raise ValueError(f"{label}: invalid current contract {differences}")
    if not values["qsgw_input_contract"]:
        raise ValueError(f"{label}: empty qsgw_input_contract")
    if not re.fullmatch(
        r"[0-9a-fA-F]{64}", values["qsgw_input_contract_sha256"]
    ):
        raise ValueError(f"{label}: invalid qsgw_input_contract_sha256")
    beta = float(values["qsgw_mixing_beta"])
    if not math.isfinite(beta) or beta <= 0.0:
        raise ValueError(f"{label}: invalid qsgw_mixing_beta")
    return values


def _validate_legacy_current_contract(
    old_text: str,
    current_texts: tuple[tuple[str, str], ...],
    final_iteration: int,
    expected_legacy_use_fullcoul_exx: bool,
    allow_iteration_prefix: bool,
) -> dict[str, object]:
    old = parse_contract(old_text, "legacy trace")
    old_expected = {
        "qsgw_contract_version": "4",
        "task": "qsgw",
        "fixed_basis": "immutable_reference",
        "qsgw_mixer": "linear",
        "qsgw_mixing_beta": "1",
        "starting_vxc": "dft_only",
        "vxc_basis": "fixed_state",
        "qsgw_update_hartree": "0",
        "qsgw_hartree_coulomb": "truncated",
        "qsgw_hartree_normalization": "legacy_extra_inverse_nk",
        "use_symmetry_gw": "0",
        "use_symmetry_exx": "0",
        "replace_w_head": "0",
        "option_dielect_func": "0",
        "nfreq": "6",
        "n_params_anacon": "6",
        "n_params_anacon_resample": "-1",
        "anacon_nfreq": "-1",
        "anacon_tfgrids_type": "-101",
        "use_shrink_abfs": "0",
        "use_fullcoul_exx": (
            "1" if expected_legacy_use_fullcoul_exx else "0"
        ),
        "use_fullcoul_eps": "1",
        "use_fullcoul_wc": "0",
        "constants_choice": "internal",
        "ac_policy": "direct_pade",
    }
    differences: dict[str, dict[str, object]] = {}
    for key, expected in old_expected.items():
        actual = old[key]
        if key == "qsgw_mixing_beta":
            equal = math.isclose(
                float(actual), float(expected), rel_tol=0.0, abs_tol=1.0e-15
            )
        else:
            equal = actual == expected
        if not equal:
            differences[key] = {"expected": expected, "actual": actual}
    try:
        declared_min_iteration = int(old["qsgw_min_iter"])
        declared_max_iteration = int(old["qsgw_max_iter"])
    except ValueError as error:
        raise ValueError(
            "legacy trace: qsgw_min_iter and qsgw_max_iter must be integers"
        ) from error
    if declared_min_iteration != declared_max_iteration:
        differences["qsgw_iteration_bounds"] = {
            "expected": "qsgw_min_iter == qsgw_max_iter",
            "actual": (
                f"{declared_min_iteration} != {declared_max_iteration}"
            ),
        }
    elif allow_iteration_prefix:
        if declared_max_iteration < final_iteration:
            differences["qsgw_max_iter"] = {
                "expected": f">= {final_iteration}",
                "actual": str(declared_max_iteration),
            }
    elif declared_max_iteration != final_iteration:
        differences["qsgw_min_iter"] = {
            "expected": str(final_iteration),
            "actual": str(declared_min_iteration),
        }
        differences["qsgw_max_iter"] = {
            "expected": str(final_iteration),
            "actual": str(declared_max_iteration),
        }
    if differences:
        raise ValueError(f"legacy trace: invalid direct-update contract {differences}")
    iteration_selection_mode = (
        "prefix" if declared_max_iteration > final_iteration else "exact"
    )

    current_headers = [
        (label, _parse_current_v5_header(text, label))
        for label, text in current_texts
    ]
    reference_label, reference = current_headers[0]
    for label, values in current_headers[1:]:
        mismatches = {
            key: {"expected": reference[key], "actual": values[key]}
            for key in sorted(CURRENT_V5_KEYS)
            if values[key] != reference[key]
        }
        if mismatches:
            raise ValueError(
                f"{label}: header differs from {reference_label}: {mismatches}"
            )
    return {
        "passed": True,
        "old_version": 4,
        "current_version": 5,
        "fixed_basis_mapping": "immutable_reference_to_immutable_mf0",
        "direct_update_mapping": "legacy_linear_beta_1_to_current_none",
        "iteration_selection_mode": iteration_selection_mode,
        "selected_final_iteration": final_iteration,
        "declared_final_iteration": declared_max_iteration,
        "legacy_effective_hamiltonian_semantics": (
            "upper_triangle_authoritative"
        ),
        "legacy_declared_use_fullcoul_exx": (
            expected_legacy_use_fullcoul_exx
        ),
        "current_input_contract": reference["qsgw_input_contract"],
        "current_input_contract_sha256": reference[
            "qsgw_input_contract_sha256"
        ],
    }


def _selected_common_rows(
    rows: dict[RowKey, RowValue],
    label: str,
    iterations: set[int],
) -> dict[RowKey, RowValue]:
    common_components = {
        "h0",
        "vxc_dft",
        "sigma_c_iw",
        "exx",
        "vc",
        "raw_h",
        "mixed_h",
        "rotation_u",
        "occupation",
    }
    selected = {
        key: value
        for key, value in rows.items()
        if key[2] in common_components or key[2].startswith("wfc_spinor")
    }
    wfc_at_zero = {
        key[2] for key in selected
        if key[0] == 0 and key[2].startswith("wfc_spinor")
    }
    if not wfc_at_zero:
        raise ValueError(f"{label}: no iteration-zero wavefunctions")
    for iteration in sorted(iterations):
        observed = {key[2] for key in selected if key[0] == iteration}
        required = {
            "h0", "vxc_dft", "occupation"
        } if iteration == 0 else {
            "sigma_c_iw", "exx", "vc", "raw_h", "mixed_h",
            "rotation_u", "occupation",
        }
        required.update(wfc_at_zero)
        missing = sorted(required - observed)
        if missing:
            raise ValueError(
                f"{label}: iteration {iteration} missing components {missing}"
            )
        iteration_wfc = {
            component for component in observed
            if component.startswith("wfc_spinor")
        }
        if iteration_wfc != wfc_at_zero:
            raise ValueError(
                f"{label}: iteration {iteration} wavefunction layout differs"
            )
    return selected


def _matrix_block_frequencies(
    rows: dict[RowKey, RowValue],
) -> dict[tuple[int, str, int, int, int], float]:
    frequencies: dict[tuple[int, str, int, int, int], float] = {}
    for key, (frequency, _value) in rows.items():
        block = (key[0], key[2], key[3], key[4], key[5])
        if block in frequencies and frequencies[block] != frequency:
            raise ValueError(
                f"matrix block {block} has inconsistent frequencies"
            )
        frequencies[block] = frequency
    return frequencies


def _legacy_current_component_metrics(
    old_rows: dict[RowKey, RowValue],
    current_rows: dict[RowKey, RowValue],
    max_abs_tolerance_ha: float,
    relative_tolerance: float,
    frequency_tolerance_ha: float,
) -> dict[str, dict[str, float | int | bool]]:
    old_matrices = _matrix_groups(old_rows)
    current_matrices = _matrix_groups(current_rows)
    old_frequencies = _matrix_block_frequencies(old_rows)
    current_frequencies = _matrix_block_frequencies(current_rows)
    state_components = {
        component for _iteration, component, _spin, _kpoint, _frequency
        in old_matrices
        if component.startswith("wfc_spinor") or component == "rotation_u"
    }
    old_physical = {
        key: value for key, value in old_matrices.items()
        if key[1] not in state_components
    }
    current_physical = {
        key: value for key, value in current_matrices.items()
        if key[1] not in state_components
    }
    if set(old_physical) != set(current_physical):
        missing = sorted(set(old_physical) - set(current_physical))[:5]
        extra = sorted(set(current_physical) - set(old_physical))[:5]
        raise ValueError(
            f"legacy/current matrix layout differs; missing={missing}, extra={extra}"
        )

    accumulators: dict[str, dict[str, float | int]] = {}
    for key in sorted(old_physical):
        old_frequency = old_frequencies[key]
        current_frequency = current_frequencies[key]
        frequency_difference = abs(old_frequency - current_frequency)
        if frequency_difference > frequency_tolerance_ha:
            raise ValueError(f"frequency differs for matrix block {key}")
        old_matrix = _legacy_matrix_for_comparison(
            key[1], old_physical[key]
        )
        current_matrix = current_physical[key]
        if old_matrix.shape != current_matrix.shape:
            raise ValueError(f"matrix shape differs for block {key}")
        difference = current_matrix - old_matrix
        maximum = float(np.max(np.abs(difference)))
        scale = max(
            float(np.linalg.norm(old_matrix)),
            float(np.linalg.norm(current_matrix)),
            1.0e-30,
        )
        relative = float(np.linalg.norm(difference) / scale)
        entry = accumulators.setdefault(
            key[1],
            {
                "block_count": 0,
                "element_count": 0,
                "max_abs_diff_ha": 0.0,
                "max_relative_frobenius": 0.0,
                "max_frequency_abs_diff_ha": 0.0,
            },
        )
        entry["block_count"] = int(entry["block_count"]) + 1
        entry["element_count"] = int(entry["element_count"]) + int(
            difference.size
        )
        entry["max_abs_diff_ha"] = max(
            float(entry["max_abs_diff_ha"]), maximum
        )
        entry["max_relative_frobenius"] = max(
            float(entry["max_relative_frobenius"]), relative
        )
        entry["max_frequency_abs_diff_ha"] = max(
            float(entry["max_frequency_abs_diff_ha"]), frequency_difference
        )

    metrics: dict[str, dict[str, float | int | bool]] = {}
    for component, entry in accumulators.items():
        component_abs_tolerance = (
            min(max_abs_tolerance_ha, 1.0e-12)
            if component == "occupation"
            else max_abs_tolerance_ha
        )
        maximum = float(entry["max_abs_diff_ha"])
        relative = float(entry["max_relative_frobenius"])
        metrics[component] = {
            **entry,
            "max_abs_tolerance_ha": component_abs_tolerance,
            "relative_frobenius_tolerance": relative_tolerance,
            "passed": (
                maximum <= component_abs_tolerance
                and relative <= relative_tolerance
            ),
        }
    return metrics


def _parse_current_eigenvalue_rows(
    text: str, iterations: set[int], channel: int
) -> dict[tuple[int, int, int, int], float]:
    rows: dict[tuple[int, int, int, int], float] = {}
    for line_number, line in enumerate(text.splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) != 9:
            raise ValueError(
                f"current eigenvalue trace:{line_number}: expected 9 columns"
            )
        iteration = int(fields[0])
        row_channel = int(fields[1])
        if iteration not in iterations or row_channel != channel:
            continue
        key = (iteration, int(fields[2]), int(fields[3]), int(fields[7]))
        energy_ha = float(fields[8]) / HA2EV
        if key in rows:
            raise ValueError(
                f"current eigenvalue trace:{line_number}: duplicate row {key}"
            )
        if not math.isfinite(energy_ha):
            raise ValueError(
                f"current eigenvalue trace:{line_number}: non-finite energy"
            )
        rows[key] = energy_ha
    if not rows:
        raise ValueError("current eigenvalue trace has no selected rows")
    return rows


def _eigenvalue_metrics(
    old_rows: dict[RowKey, RowValue],
    current_rows: dict[RowKey, RowValue],
    current_eigenvalues: dict[tuple[int, int, int, int], float],
    iterations: set[int],
    tolerance_ha: float,
) -> dict[str, float | int | bool]:
    old_matrices = _matrix_groups(old_rows)
    current_matrices = _matrix_groups(current_rows)
    old_vs_trace = 0.0
    current_vs_trace = 0.0
    old_vs_current = 0.0
    count = 0
    observed_keys: set[tuple[int, int, int, int]] = set()
    for iteration in sorted(iterations):
        component = "h0" if iteration == 0 else "mixed_h"
        blocks = sorted({
            (spin, kpoint)
            for row_iteration, row_component, spin, kpoint, frequency_index
            in old_matrices
            if row_iteration == iteration
            and row_component == component
            and frequency_index == -1
        })
        if not blocks:
            raise ValueError(
                f"legacy trace: missing {component} at iteration {iteration}"
            )
        for spin, kpoint in blocks:
            key = (iteration, component, spin, kpoint, -1)
            if key not in current_matrices:
                raise ValueError(f"current matrix trace: missing block {key}")
            old_values = np.linalg.eigvalsh(
                _legacy_matrix_for_comparison(component, old_matrices[key])
            )
            current_values = np.linalg.eigvalsh(current_matrices[key])
            if old_values.shape != current_values.shape:
                raise ValueError(f"eigenvalue block shape differs for {key}")
            trace_values = []
            for band in range(old_values.size):
                trace_key = (iteration, spin, kpoint, band)
                if trace_key not in current_eigenvalues:
                    raise ValueError(
                        f"current eigenvalue trace: missing row {trace_key}"
                    )
                observed_keys.add(trace_key)
                trace_values.append(current_eigenvalues[trace_key])
            trace_array = np.asarray(trace_values)
            old_vs_trace = max(
                old_vs_trace,
                float(np.max(np.abs(old_values - trace_array))),
            )
            current_vs_trace = max(
                current_vs_trace,
                float(np.max(np.abs(current_values - trace_array))),
            )
            old_vs_current = max(
                old_vs_current,
                float(np.max(np.abs(old_values - current_values))),
            )
            count += old_values.size
    if observed_keys != set(current_eigenvalues):
        extra = sorted(set(current_eigenvalues) - observed_keys)[:5]
        raise ValueError(f"current eigenvalue trace has extra rows {extra}")
    return {
        "count": count,
        "max_old_vs_current_abs_diff_ha": old_vs_trace,
        "max_current_matrix_vs_trace_abs_diff_ha": current_vs_trace,
        "max_old_matrix_vs_current_matrix_abs_diff_ha": old_vs_current,
        "tolerance_ha": tolerance_ha,
        "passed": (
            old_vs_trace <= tolerance_ha
            and current_vs_trace <= tolerance_ha
            and old_vs_current <= tolerance_ha
        ),
    }


def _parse_current_iteration_rows(
    text: str, iterations: set[int]
) -> dict[int, tuple[float, float, float]]:
    rows: dict[int, tuple[float, float, float]] = {}
    for line_number, line in enumerate(text.splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) < 7:
            raise ValueError(
                f"current iteration trace:{line_number}: expected at least 7 columns"
            )
        iteration = int(fields[0])
        if iteration not in iterations:
            continue
        values = (float(fields[4]), float(fields[5]), float(fields[6]))
        if iteration in rows:
            raise ValueError(
                f"current iteration trace:{line_number}: duplicate iteration"
            )
        if not all(math.isfinite(value) for value in values):
            raise ValueError(
                f"current iteration trace:{line_number}: non-finite scalar"
            )
        rows[iteration] = values
    if set(rows) != iterations:
        raise ValueError(
            "current iteration trace does not cover selected iterations"
        )
    return rows


def _legacy_scalar(
    matrices: dict[tuple[int, str, int, int, int], np.ndarray],
    iteration: int,
    component: str,
) -> float:
    key = (iteration, component, 0, 0, -1)
    if key not in matrices or matrices[key].shape != (1, 1):
        raise ValueError(f"legacy trace: invalid scalar block {key}")
    value = matrices[key][0, 0]
    if value.imag != 0.0 or not math.isfinite(float(value.real)):
        raise ValueError(f"legacy trace: invalid scalar value {key}")
    return float(value.real)


def _scalar_metrics(
    old_rows: dict[RowKey, RowValue],
    current_iterations: dict[int, tuple[float, float, float]],
    iterations: set[int],
    fermi_tolerance_ha: float,
    gap_tolerance_ev: float,
) -> dict[str, float | bool]:
    old_matrices = _matrix_groups(old_rows)
    maximum_fermi = 0.0
    maximum_gap = 0.0
    maximum_electron = 0.0
    for iteration in sorted(iterations):
        current_fermi_ev, current_gap_ev, current_electron = \
            current_iterations[iteration]
        old_fermi = _legacy_scalar(
            old_matrices, iteration, "fermi_energy_ha"
        )
        old_gap = _legacy_scalar(old_matrices, iteration, "gap_ha")
        old_electron = _legacy_scalar(
            old_matrices, iteration, "electron_count"
        )
        maximum_fermi = max(
            maximum_fermi, abs(old_fermi - current_fermi_ev / HA2EV)
        )
        maximum_gap = max(
            maximum_gap, abs(old_gap * HA2EV - current_gap_ev)
        )
        maximum_electron = max(
            maximum_electron, abs(old_electron - current_electron)
        )
    electron_tolerance = 1.0e-10
    return {
        "max_fermi_abs_diff_ha": maximum_fermi,
        "fermi_tolerance_ha": fermi_tolerance_ha,
        "max_gap_abs_diff_ev": maximum_gap,
        "gap_tolerance_ev": gap_tolerance_ev,
        "max_electron_count_abs_diff": maximum_electron,
        "electron_count_tolerance": electron_tolerance,
        "passed": (
            maximum_fermi <= fermi_tolerance_ha
            and maximum_gap <= gap_tolerance_ev
            and maximum_electron <= electron_tolerance
        ),
    }


def compare_legacy_v4_current_v5(
    *,
    old_matrix_text: str,
    current_matrix_text: str,
    current_eigenvalue_text: str,
    current_iteration_text: str,
    iterations: list[int],
    channel: int = 0,
    frequency_tolerance_ha: float = 1.0e-10,
    matrix_max_abs_tolerance_ha: float = 1.0e-8,
    matrix_relative_tolerance: float = 1.0e-8,
    eigenvalue_tolerance_ha: float = 1.0e-6,
    gap_tolerance_ev: float = 1.0e-5,
    degeneracy_tolerance_ha: float = 1.0e-8,
    state_tolerance: float = 1.0e-10,
    expected_legacy_use_fullcoul_exx: bool = True,
    allow_iteration_prefix: bool = False,
) -> dict[str, object]:
    if channel != 0:
        raise ValueError("legacy-v4/current-v5 comparison requires grid channel 0")
    if iterations != list(range(iterations[-1] + 1)) or len(iterations) < 2:
        raise ValueError(
            "legacy-v4/current-v5 comparison requires continuous iterations from zero"
        )
    if not isinstance(expected_legacy_use_fullcoul_exx, bool):
        raise ValueError("expected legacy use_fullcoul_exx must be a bool")
    if not isinstance(allow_iteration_prefix, bool):
        raise ValueError("allow_iteration_prefix must be a bool")
    for label, tolerance in (
        ("frequency", frequency_tolerance_ha),
        ("matrix max-abs", matrix_max_abs_tolerance_ha),
        ("matrix relative", matrix_relative_tolerance),
        ("eigenvalue", eigenvalue_tolerance_ha),
        ("gap", gap_tolerance_ev),
        ("degeneracy", degeneracy_tolerance_ha),
        ("state", state_tolerance),
    ):
        if not math.isfinite(tolerance) or tolerance < 0.0:
            raise ValueError(f"invalid {label} tolerance")
    selected = set(iterations)
    contract = _validate_legacy_current_contract(
        old_matrix_text,
        (
            ("current matrix trace", current_matrix_text),
            ("current eigenvalue trace", current_eigenvalue_text),
            ("current iteration trace", current_iteration_text),
        ),
        iterations[-1],
        expected_legacy_use_fullcoul_exx,
        allow_iteration_prefix,
    )
    old_all_rows = parse_rows(
        old_matrix_text, "legacy matrix trace", selected, channel
    )
    current_all_rows = parse_rows(
        current_matrix_text, "current matrix trace", selected, channel
    )
    old_rows = _selected_common_rows(
        old_all_rows, "legacy matrix trace", selected
    )
    current_rows = _selected_common_rows(
        current_all_rows, "current matrix trace", selected
    )
    if set(old_rows) != set(current_rows):
        missing = sorted(set(old_rows) - set(current_rows))[:5]
        extra = sorted(set(current_rows) - set(old_rows))[:5]
        raise ValueError(
            f"legacy/current common row layout differs; missing={missing}, extra={extra}"
        )
    components = _legacy_current_component_metrics(
        old_rows,
        current_rows,
        matrix_max_abs_tolerance_ha,
        matrix_relative_tolerance,
        frequency_tolerance_ha,
    )
    state = _state_metrics(
        old_rows,
        current_rows,
        eigenvalue_tolerance_ha,
        degeneracy_tolerance_ha,
        state_tolerance,
        legacy_upper_triangle=True,
    )
    current_eigenvalues = _parse_current_eigenvalue_rows(
        current_eigenvalue_text, selected, channel
    )
    eigenvalues = _eigenvalue_metrics(
        old_rows,
        current_rows,
        current_eigenvalues,
        selected,
        eigenvalue_tolerance_ha,
    )
    current_iterations = _parse_current_iteration_rows(
        current_iteration_text, selected
    )
    scalars = _scalar_metrics(
        old_all_rows,
        current_iterations,
        selected,
        eigenvalue_tolerance_ha,
        gap_tolerance_ev,
    )
    components_passed = all(
        bool(metric["passed"]) for metric in components.values()
    )
    return {
        "passed": bool(
            contract["passed"]
            and components_passed
            and state["passed"]
            and eigenvalues["passed"]
            and scalars["passed"]
        ),
        "iterations": iterations,
        "channel": channel,
        "contract_mode": "legacy_v4_to_current_v5",
        "contract": contract,
        "components": components,
        "state": state,
        "eigenvalues": eigenvalues,
        "scalars": scalars,
        "ignored_legacy_components": [
            "velocity_x", "velocity_y", "velocity_z",
            "fermi_energy_ha", "electron_count", "gap_ha",
        ],
    }


def compare_trace_text(
    old_text: str,
    new_text: str,
    iterations: list[int],
    channel: int = 0,
    frequency_tolerance: float = 1.0e-10,
    eigenvalue_tolerance: float = 1.0e-9,
    degeneracy_tolerance: float = 1.0e-8,
    state_tolerance: float = 5.0e-8,
    contract_mode: str = "oracle",
) -> dict[str, object]:
    if contract_mode not in {"oracle", "grid_metamorphic"}:
        raise ValueError(f"unsupported contract mode {contract_mode!r}")
    if contract_mode == "grid_metamorphic" and channel != 0:
        raise ValueError("grid_metamorphic contract mode requires channel 0")
    for label, tolerance in (
        ("frequency", frequency_tolerance),
        ("eigenvalue", eigenvalue_tolerance),
        ("degeneracy", degeneracy_tolerance),
        ("state", state_tolerance),
    ):
        if not math.isfinite(tolerance) or tolerance < 0.0:
            raise ValueError(f"invalid {label} tolerance")

    selected = set(iterations)
    require_current = contract_mode == "grid_metamorphic"
    old_contract = parse_contract(
        old_text, "old trace", require_current=require_current
    )
    new_contract = parse_contract(
        new_text, "new trace", require_current=require_current
    )
    ignored_contract_keys: frozenset[str] = frozenset()
    comparison_keys = list(CONTRACT_KEYS)
    for key in CURRENT_SHARED_CONTRACT_KEYS + CURRENT_HEADWING_CONTRACT_KEYS:
        if require_current or (key in old_contract and key in new_contract):
            comparison_keys.append(key)
    if contract_mode == "grid_metamorphic":
        if (
            old_contract["task"] != "qsgw"
            or new_contract["task"] != "qsgw_band"
        ):
            raise ValueError(
                "grid_metamorphic contract mode requires qsgw and qsgw_band traces"
            )
        ignored_contract_keys = frozenset({"task"})
    contract = compare_contracts(
        old_contract,
        new_contract,
        ignored_keys=ignored_contract_keys,
        keys=tuple(comparison_keys),
    )
    old_rows = parse_rows(old_text, "old trace", selected, channel)
    new_rows = parse_rows(new_text, "new trace", selected, channel)
    validate_iteration_coverage(
        old_rows, "old trace", selected, old_contract
    )
    validate_iteration_coverage(
        new_rows, "new trace", selected, new_contract
    )
    validate_auxiliary_layout(old_rows, "old trace", selected)
    validate_auxiliary_layout(new_rows, "new trace", selected)
    old_rows, old_zero_velocity_rows = _drop_optional_zero_velocity_rows(
        old_rows, old_contract
    )
    new_rows, new_zero_velocity_rows = _drop_optional_zero_velocity_rows(
        new_rows, new_contract
    )
    old_layout = {key for key in old_rows if key[2] not in {"rotation_u"}}
    new_layout = {key for key in new_rows if key[2] not in {"rotation_u"}}
    if old_layout != new_layout:
        missing = sorted(old_layout - new_layout)[:5]
        extra = sorted(new_layout - old_layout)[:5]
        raise ValueError(f"trace row layout differs; missing={missing}, extra={extra}")
    components = _component_metrics(old_rows, new_rows, frequency_tolerance)
    state = _state_metrics(
        old_rows, new_rows, eigenvalue_tolerance, degeneracy_tolerance,
        state_tolerance,
    )
    component_passed = all(bool(metric["passed"]) for metric in components.values())
    return {
        "passed": bool(contract["passed"] and component_passed and state["passed"]),
        "iterations": iterations,
        "channel": channel,
        "contract_mode": contract_mode,
        "contract": contract,
        "components": components,
        "state": state,
        "optional_zero_velocity_rows_ignored": {
            "old": old_zero_velocity_rows,
            "new": new_zero_velocity_rows,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("old_trace", type=Path)
    parser.add_argument("new_trace", type=Path)
    parser.add_argument("output_json", type=Path)
    parser.add_argument("--iterations", default="0:5")
    parser.add_argument("--channel", type=int, choices=(0, 1), default=0)
    parser.add_argument("--frequency-tolerance", type=float, default=1.0e-10)
    parser.add_argument(
        "--matrix-max-abs-tolerance-ha", type=float, default=1.0e-8
    )
    parser.add_argument(
        "--matrix-relative-tolerance", type=float, default=1.0e-8
    )
    parser.add_argument("--eigenvalue-tolerance", type=float, default=1.0e-9)
    parser.add_argument("--gap-tolerance-ev", type=float, default=1.0e-5)
    parser.add_argument("--degeneracy-tolerance", type=float, default=1.0e-8)
    parser.add_argument("--state-tolerance", type=float, default=5.0e-8)
    parser.add_argument("--current-eigenvalue-trace", type=Path)
    parser.add_argument("--current-iteration-trace", type=Path)
    parser.add_argument(
        "--expected-legacy-use-fullcoul-exx",
        choices=("0", "1"),
        default="1",
        help="declared use_fullcoul_exx value required in the legacy trace",
    )
    parser.add_argument(
        "--allow-iteration-prefix",
        action="store_true",
        help=(
            "allow a selected prefix when the legacy trace declares a longer "
            "forced run"
        ),
    )
    parser.add_argument(
        "--contract-mode",
        choices=(
            "oracle",
            "grid_metamorphic",
            "legacy_v4_to_current_v5",
        ),
        default="oracle",
    )
    args = parser.parse_args()

    try:
        if args.contract_mode == "legacy_v4_to_current_v5":
            if args.current_eigenvalue_trace is None:
                raise ValueError(
                    "legacy_v4_to_current_v5 requires "
                    "--current-eigenvalue-trace"
                )
            if args.current_iteration_trace is None:
                raise ValueError(
                    "legacy_v4_to_current_v5 requires "
                    "--current-iteration-trace"
                )
            report = compare_legacy_v4_current_v5(
                old_matrix_text=args.old_trace.read_text(encoding="utf-8"),
                current_matrix_text=args.new_trace.read_text(encoding="utf-8"),
                current_eigenvalue_text=(
                    args.current_eigenvalue_trace.read_text(encoding="utf-8")
                ),
                current_iteration_text=(
                    args.current_iteration_trace.read_text(encoding="utf-8")
                ),
                iterations=parse_iterations(args.iterations),
                channel=args.channel,
                frequency_tolerance_ha=args.frequency_tolerance,
                matrix_max_abs_tolerance_ha=(
                    args.matrix_max_abs_tolerance_ha
                ),
                matrix_relative_tolerance=args.matrix_relative_tolerance,
                eigenvalue_tolerance_ha=args.eigenvalue_tolerance,
                gap_tolerance_ev=args.gap_tolerance_ev,
                degeneracy_tolerance_ha=args.degeneracy_tolerance,
                state_tolerance=args.state_tolerance,
                expected_legacy_use_fullcoul_exx=(
                    args.expected_legacy_use_fullcoul_exx == "1"
                ),
                allow_iteration_prefix=args.allow_iteration_prefix,
            )
        else:
            report = compare_trace_text(
                args.old_trace.read_text(encoding="utf-8"),
                args.new_trace.read_text(encoding="utf-8"),
                parse_iterations(args.iterations),
                channel=args.channel,
                frequency_tolerance=args.frequency_tolerance,
                eigenvalue_tolerance=args.eigenvalue_tolerance,
                degeneracy_tolerance=args.degeneracy_tolerance,
                state_tolerance=args.state_tolerance,
                contract_mode=args.contract_mode,
            )
    except Exception as error:
        report = {"passed": False, "error": str(error)}
    report["old_trace"] = str(args.old_trace.resolve())
    report["new_trace"] = str(args.new_trace.resolve())
    if args.current_eigenvalue_trace is not None:
        report["current_eigenvalue_trace"] = str(
            args.current_eigenvalue_trace.resolve()
        )
    if args.current_iteration_trace is not None:
        report["current_iteration_trace"] = str(
            args.current_iteration_trace.resolve()
        )
    args.output_json.write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    raise SystemExit(0 if report["passed"] else 2)


if __name__ == "__main__":
    main()
