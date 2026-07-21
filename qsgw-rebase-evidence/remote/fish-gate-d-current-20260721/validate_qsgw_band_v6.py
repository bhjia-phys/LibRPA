#!/usr/bin/env python3
"""Validate a current-contract QSGW band run and its ABACUS H(R) export."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import numpy as np

import cmp_qsgw


SCHEMA = "librpa-qsgw-band-v6-validation-v1"
HA2EV = 27.211386245988


class BandValidationError(ValueError):
    pass


def _matrix(blocks, iteration, channel, component, spin, kpoint):
    key = (iteration, channel, component, spin, kpoint, -1)
    if key not in blocks:
        raise BandValidationError("missing matrix block {}".format(key))
    return np.asarray(blocks[key][1], dtype=np.complex128)


def _component_layout(blocks, iteration, channel, component):
    return {
        (key[3], key[4])
        for key in blocks
        if key[0] == iteration
        and key[1] == channel
        and key[2] == component
        and key[5] == -1
    }


def _max_abs(value):
    return float(np.max(np.abs(value))) if value.size else 0.0


def _relative_difference(actual, expected):
    numerator = float(np.linalg.norm(actual - expected))
    denominator = float(np.linalg.norm(expected))
    return numerator / denominator if denominator else numerator


def _check_difference(actual, expected, label, absolute_tolerance,
                      relative_tolerance):
    if actual.shape != expected.shape:
        raise BandValidationError("{} shape differs".format(label))
    maximum = _max_abs(actual - expected)
    relative = _relative_difference(actual, expected)
    if maximum > absolute_tolerance or relative > relative_tolerance:
        raise BandValidationError(
            "{} differs: max_abs={:.6e}, relative_frobenius={:.6e}".format(
                label, maximum, relative
            )
        )
    return maximum, relative


def _hermiticity(matrix):
    if matrix.shape[0] != matrix.shape[1]:
        return math.inf
    return _max_abs(matrix - matrix.conj().T)


def _unitarity(matrix):
    if matrix.shape[0] != matrix.shape[1]:
        return math.inf
    identity = np.eye(matrix.shape[0], dtype=np.complex128)
    return _max_abs(matrix @ matrix.conj().T - identity)


def _eigenvalues(eigen_rows, iteration, channel, spin, kpoint):
    values = {
        key[4]: item[1]
        for key, item in eigen_rows.items()
        if key[:4] == (iteration, channel, spin, kpoint)
    }
    if not values or set(values) != set(range(len(values))):
        raise BandValidationError(
            "incomplete eigenvalue block at iteration {}, channel {}, spin {}, kpoint {}"
            .format(iteration, channel, spin, kpoint)
        )
    return np.asarray([values[index] for index in range(len(values))])


def _coordinates(eigen_rows, iteration, channel, spin, kpoint):
    values = {
        item[0]
        for key, item in eigen_rows.items()
        if key[:4] == (iteration, channel, spin, kpoint)
    }
    if len(values) != 1:
        raise BandValidationError("inconsistent eigenvalue coordinates")
    return next(iter(values))


def _apply_cut(matrix, reference, previous_eigenvalues_ev, efermi_ev,
               mode, unoccupied_keep, shift_ha):
    result = np.array(matrix, dtype=np.complex128, copy=True)
    if mode == 0:
        return result, result.shape[0]
    occupied = int(np.count_nonzero(previous_eigenvalues_ev < efermi_ev))
    active_limit = min(result.shape[0], occupied + unoccupied_keep)
    shift = shift_ha if mode == 2 else 0.0
    for row in range(result.shape[0]):
        for column in range(result.shape[1]):
            if row < active_limit and column < active_limit:
                continue
            result[row, column] = (
                reference[row, row] + shift if row == column else 0.0
            )
    return result, active_limit


def _parse_band_table(path, n_kpoints, n_bands):
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="ascii").splitlines(), 1):
        fields = line.split()
        if not fields:
            continue
        if len(fields) != 4 + 2 * n_bands:
            raise BandValidationError(
                "{}:{} has {} columns, expected {}".format(
                    path, line_number, len(fields), 4 + 2 * n_bands
                )
            )
        try:
            index = int(fields[0])
            coordinate = tuple(float(value) for value in fields[1:4])
            occupation = np.asarray(
                [float(fields[4 + 2 * band]) for band in range(n_bands)]
            )
            energy = np.asarray(
                [float(fields[5 + 2 * band]) for band in range(n_bands)]
            )
        except ValueError as error:
            raise BandValidationError(
                "{}:{} contains invalid numeric data".format(path, line_number)
            ) from error
        if index != len(rows) + 1 or not np.all(np.isfinite(occupation)) or not np.all(np.isfinite(energy)):
            raise BandValidationError("{} has invalid row ordering or data".format(path))
        rows.append((coordinate, occupation, energy))
    if len(rows) != n_kpoints:
        raise BandValidationError(
            "{} has {} kpoints, expected {}".format(path, len(rows), n_kpoints)
        )
    return rows


def _table_gap(rows):
    occupied = []
    unoccupied = []
    for _coordinate, occupations, energies in rows:
        occupied.extend(energies[occupations > 0.0])
        unoccupied.extend(energies[occupations <= 0.0])
    if not occupied or not unoccupied:
        raise BandValidationError("band table does not contain both occupations")
    return float(min(unoccupied) - max(occupied))


def _parse_csr(path):
    lines = path.read_text(encoding="ascii").splitlines()
    if len(lines) < 3 or lines[0].strip() != "STEP: 0":
        raise BandValidationError("invalid ABACUS CSR header in {}".format(path))
    dimension_match = re.fullmatch(r"Matrix Dimension of H\(R\): (\d+)", lines[1].strip())
    count_match = re.fullmatch(r"Matrix number of H\(R\): (\d+)", lines[2].strip())
    if not dimension_match or not count_match:
        raise BandValidationError("invalid ABACUS CSR dimensions in {}".format(path))
    dimension = int(dimension_match.group(1))
    block_count = int(count_match.group(1))
    if dimension <= 0 or block_count <= 0:
        raise BandValidationError("empty ABACUS CSR in {}".format(path))
    blocks = {}
    offset = 3
    for _index in range(block_count):
        if offset + 3 >= len(lines):
            raise BandValidationError("truncated ABACUS CSR in {}".format(path))
        header = lines[offset].split()
        offset += 1
        if len(header) != 4:
            raise BandValidationError("invalid ABACUS CSR block header")
        cell = tuple(int(value) for value in header[:3])
        nnz = int(header[3])
        values = [float(value) for value in lines[offset].split()]
        columns = [int(value) for value in lines[offset + 1].split()]
        row_offsets = [int(value) for value in lines[offset + 2].split()]
        offset += 3
        if (
            cell in blocks
            or nnz <= 0
            or len(values) != nnz
            or len(columns) != nnz
            or len(row_offsets) != dimension + 1
            or row_offsets[0] != 0
            or row_offsets[-1] != nnz
            or any(left > right for left, right in zip(row_offsets, row_offsets[1:]))
            or any(column < 0 or column >= dimension for column in columns)
            or not all(math.isfinite(value) for value in values)
        ):
            raise BandValidationError("invalid ABACUS CSR block in {}".format(path))
        matrix = np.zeros((dimension, dimension), dtype=np.complex128)
        for row in range(dimension):
            for item in range(row_offsets[row], row_offsets[row + 1]):
                matrix[row, columns[item]] = values[item] / 2.0
        blocks[cell] = matrix
    if offset != len(lines) and any(line.strip() for line in lines[offset:]):
        raise BandValidationError("trailing ABACUS CSR data in {}".format(path))
    return dimension, blocks


def _reference_wavefunctions(blocks, channel, spin, kpoint):
    components = sorted(
        {
            key[2]
            for key in blocks
            if key[0] == 0
            and key[1] == channel
            and key[3] == spin
            and key[4] == kpoint
            and key[2].startswith("wfc_spinor")
        },
        key=lambda value: int(value[len("wfc_spinor"):]),
    )
    if not components:
        raise BandValidationError("missing reference wavefunctions")
    expected = ["wfc_spinor{}".format(index) for index in range(len(components))]
    if components != expected:
        raise BandValidationError("non-contiguous reference spinors")
    return np.concatenate(
        [_matrix(blocks, 0, channel, component, spin, kpoint) for component in components],
        axis=1,
    )


def _wavefunction_components(blocks, iteration, channel, spin, kpoint):
    components = sorted(
        {
            key[2]
            for key in blocks
            if key[0] == iteration
            and key[1] == channel
            and key[3] == spin
            and key[4] == kpoint
            and key[2].startswith("wfc_spinor")
            and key[5] == -1
        },
        key=lambda value: int(value[len("wfc_spinor"):]),
    )
    expected = ["wfc_spinor{}".format(index) for index in range(len(components))]
    if not components or components != expected:
        raise BandValidationError(
            "missing or non-contiguous wavefunctions at iteration {}, channel {}, "
            "spin {}, kpoint {}".format(iteration, channel, spin, kpoint)
        )
    return components


def _validate_csr_export(path, blocks, eigen_rows, iteration, spin, periods,
                         absolute_tolerance, relative_tolerance,
                         hermiticity_tolerance):
    dimension, real_space = _parse_csr(path)
    translation_hermiticity = 0.0
    for cell, matrix in real_space.items():
        reverse_candidates = [
            candidate
            for candidate in real_space
            if all(
                (left + right) % period == 0
                for left, right, period in zip(cell, candidate, periods)
            )
        ]
        if len(reverse_candidates) != 1:
            raise BandValidationError(
                "CSR has no unique periodic -R partner for {}".format(cell)
            )
        reverse = reverse_candidates[0]
        translation_hermiticity = max(
            translation_hermiticity,
            _max_abs(matrix - real_space[reverse].conj().T),
        )
    if translation_hermiticity > hermiticity_tolerance:
        raise BandValidationError(
            "CSR translational Hermiticity exceeds tolerance: {:.6e}".format(
                translation_hermiticity
            )
        )

    layout = sorted(_component_layout(blocks, iteration, 0, "mixed_h"))
    maximum = 0.0
    relative = 0.0
    for block_spin, kpoint in layout:
        if block_spin != spin:
            continue
        coordinate = _coordinates(eigen_rows, iteration, 0, spin, kpoint)
        ao_hamiltonian = np.zeros((dimension, dimension), dtype=np.complex128)
        for cell, matrix in real_space.items():
            phase = np.exp(2j * np.pi * sum(k * r for k, r in zip(coordinate, cell)))
            ao_hamiltonian += phase * matrix
        wavefunctions = _reference_wavefunctions(blocks, 0, spin, kpoint)
        if wavefunctions.shape[1] != dimension:
            raise BandValidationError("CSR dimension differs from the reference AO basis")
        projected = np.conj(wavefunctions) @ ao_hamiltonian @ wavefunctions.T
        observed = _matrix(blocks, iteration, 0, "mixed_h", spin, kpoint)
        block_maximum, block_relative = _check_difference(
            projected,
            observed,
            "CSR roundtrip spin {} kpoint {}".format(spin, kpoint),
            absolute_tolerance,
            relative_tolerance,
        )
        maximum = max(maximum, block_maximum)
        relative = max(relative, block_relative)
    return {
        "file": str(path),
        "dimension": dimension,
        "period": periods,
        "real_space_blocks": len(real_space),
        "translation_hermiticity_max_abs_ha": translation_hermiticity,
        "grid_roundtrip_max_abs_ha": maximum,
        "grid_roundtrip_relative_frobenius": relative,
    }


def validate_band_run(matrix_path, eigenvalue_path, summary_path, output_dir,
                      bz_sampling_path,
                      expected_iterations, expected_cut_mode,
                      expected_unoccupied_keep, expected_shift_ha,
                      closure_absolute_tolerance=1.0e-10,
                      matrix_relative_tolerance=1.0e-8,
                      hermiticity_tolerance=1.0e-10,
                      fixed_basis_tolerance=1.0e-10,
                      csr_absolute_tolerance=1.0e-8,
                      band_table_tolerance_ev=1.0e-5):
    matrix_text = matrix_path.read_text(encoding="ascii")
    eigenvalue_text = eigenvalue_path.read_text(encoding="ascii")
    summary_text = summary_path.read_text(encoding="ascii")
    bz_tokens = bz_sampling_path.read_text(encoding="ascii").split()
    try:
        periods = tuple(int(value) for value in bz_tokens[:3])
    except ValueError as error:
        raise BandValidationError("invalid bz_sampling_out grid") from error
    if len(periods) != 3 or any(value <= 0 for value in periods):
        raise BandValidationError("invalid bz_sampling_out grid")
    contracts = [
        cmp_qsgw._parse_contract(matrix_text, "matrix trace"),
        cmp_qsgw._parse_contract(eigenvalue_text, "eigenvalue trace"),
        cmp_qsgw._parse_contract(summary_text, "iteration trace"),
    ]
    if any(item != contracts[0] for item in contracts[1:]):
        raise BandValidationError("QSGW trace contracts differ")
    contract = contracts[0]
    expected_contract = {
        "qsgw_contract_version": 6,
        "band": "fixed_reference_operator_fourier_live",
        "h_qsgw_cut": "band_postprocess",
        "qsgw_band0_unoccupied_keep": expected_unoccupied_keep,
        "qsgw_band0_cut_mode": expected_cut_mode,
        "qsgw_band0_cut_shift_ha": expected_shift_ha,
        "headwing": "disabled_stage1",
    }
    for key, expected in expected_contract.items():
        actual = contract.get(key)
        if isinstance(expected, float):
            matches = math.isclose(float(actual), expected, rel_tol=0.0, abs_tol=1.0e-14)
        else:
            matches = actual == expected
        if not matches:
            raise BandValidationError(
                "QSGW contract {} differs: {} != {}".format(key, actual, expected)
            )

    blocks = cmp_qsgw._parse_matrix_trace(matrix_text, "matrix trace")
    eigen_rows = cmp_qsgw._parse_eigenvalue_trace(eigenvalue_text, "eigenvalue trace")
    summaries = cmp_qsgw._parse_iteration_summary(summary_text, "iteration trace")
    cmp_qsgw._validate_matrix_trajectory(blocks, contract, "matrix trace")
    cmp_qsgw._validate_eigenvalue_trajectory(eigen_rows, contract, "eigenvalue trace")
    iterations = sorted(summaries)
    if iterations != list(range(expected_iterations + 1)):
        raise BandValidationError("unexpected completed iteration range {}".format(iterations))
    if {key[0] for key in blocks} != set(iterations) or {key[0] for key in eigen_rows} != set(iterations):
        raise BandValidationError("trace iteration sets differ")

    channels = (0, 1)
    layouts = {
        channel: _component_layout(blocks, 0, channel, "h0")
        for channel in channels
    }
    closure_maximum = 0.0
    closure_relative = 0.0
    mixing_maximum = 0.0
    mixing_relative = 0.0
    residual_l2_difference = 0.0
    residual_max_difference = 0.0
    maximum_hermiticity = 0.0
    maximum_unitarity = 0.0
    maximum_trace_eigenvalue_difference = 0.0
    maximum_diagonalization_offdiagonal = 0.0
    maximum_diagonalization_eigenvalue_difference = 0.0
    maximum_wavefunction_rotation_absolute = 0.0
    maximum_wavefunction_rotation_relative = 0.0
    active_limits = {}
    current = {}
    for channel in channels:
        for spin, kpoint in layouts[channel]:
            h0 = _matrix(blocks, 0, channel, "h0", spin, kpoint)
            previous_eigenvalues = _eigenvalues(eigen_rows, 0, channel, spin, kpoint)
            h0_eigenvalues = np.linalg.eigvalsh(h0) * HA2EV
            maximum_trace_eigenvalue_difference = max(
                maximum_trace_eigenvalue_difference,
                _max_abs(h0_eigenvalues - previous_eigenvalues) / HA2EV,
            )
            current[(channel, spin, kpoint)], _active = _apply_cut(
                h0,
                h0,
                previous_eigenvalues,
                summaries[0]["efermi_eV"],
                expected_cut_mode,
                expected_unoccupied_keep,
                expected_shift_ha,
            )

    for iteration in range(1, expected_iterations + 1):
        grid_residual_square = 0.0
        grid_residual_maximum = 0.0
        for channel in channels:
            for spin, kpoint in layouts[channel]:
                h0 = _matrix(blocks, 0, channel, "h0", spin, kpoint)
                vxc = _matrix(blocks, 0, channel, "vxc_dft", spin, kpoint)
                exx = _matrix(blocks, iteration, channel, "exx", spin, kpoint)
                vc = _matrix(blocks, iteration, channel, "vc", spin, kpoint)
                uncut = h0 - vxc + exx + vc
                if contract["hartree"] == "delta_density":
                    uncut += _matrix(
                        blocks, iteration, channel, "delta_vh", spin, kpoint
                    )
                previous_eigenvalues = _eigenvalues(
                    eigen_rows, iteration - 1, channel, spin, kpoint
                )
                expected_raw, active_limit = _apply_cut(
                    uncut,
                    h0,
                    previous_eigenvalues,
                    summaries[iteration - 1]["efermi_eV"],
                    expected_cut_mode,
                    expected_unoccupied_keep,
                    expected_shift_ha,
                )
                active_limits[
                    "{}:{}:{}:{}".format(iteration, channel, spin, kpoint)
                ] = active_limit
                raw = _matrix(blocks, iteration, channel, "raw_h", spin, kpoint)
                block_maximum, block_relative = _check_difference(
                    raw,
                    expected_raw,
                    "raw closure iter {} channel {} spin {} kpoint {}".format(
                        iteration, channel, spin, kpoint
                    ),
                    closure_absolute_tolerance,
                    matrix_relative_tolerance,
                )
                closure_maximum = max(closure_maximum, block_maximum)
                closure_relative = max(closure_relative, block_relative)

                previous_input = current[(channel, spin, kpoint)]
                if channel == 0:
                    residual = raw - previous_input
                    grid_residual_square += float(np.vdot(residual, residual).real)
                    grid_residual_maximum = max(grid_residual_maximum, _max_abs(residual))
                if contract["qsgw_mixer"] == "linear":
                    mixed_before_cut = previous_input + contract["qsgw_mixing_beta"] * (
                        raw - previous_input
                    )
                    expected_mixed, _active = _apply_cut(
                        mixed_before_cut,
                        h0,
                        previous_eigenvalues,
                        summaries[iteration - 1]["efermi_eV"],
                        expected_cut_mode,
                        expected_unoccupied_keep,
                        expected_shift_ha,
                    )
                else:
                    expected_mixed = raw
                mixed = _matrix(blocks, iteration, channel, "mixed_h", spin, kpoint)
                block_maximum, block_relative = _check_difference(
                    mixed,
                    expected_mixed,
                    "mix closure iter {} channel {} spin {} kpoint {}".format(
                        iteration, channel, spin, kpoint
                    ),
                    closure_absolute_tolerance,
                    matrix_relative_tolerance,
                )
                mixing_maximum = max(mixing_maximum, block_maximum)
                mixing_relative = max(mixing_relative, block_relative)
                current[(channel, spin, kpoint)] = expected_mixed

                for component in ("h0", "vxc_dft", "exx", "vc", "raw_h", "mixed_h"):
                    source_iteration = 0 if component in ("h0", "vxc_dft") else iteration
                    maximum_hermiticity = max(
                        maximum_hermiticity,
                        _hermiticity(
                            _matrix(
                                blocks,
                                source_iteration,
                                channel,
                                component,
                                spin,
                                kpoint,
                            )
                        ),
                    )
                if contract["hartree"] == "delta_density":
                    maximum_hermiticity = max(
                        maximum_hermiticity,
                        _hermiticity(
                            _matrix(
                                blocks, iteration, channel, "delta_vh", spin, kpoint
                            )
                        ),
                    )
                maximum_unitarity = max(
                    maximum_unitarity,
                    _unitarity(
                        _matrix(
                            blocks, iteration, channel, "rotation_u", spin, kpoint
                        )
                    ),
                )
                rotation = _matrix(
                    blocks, iteration, channel, "rotation_u", spin, kpoint
                )
                diagonalized = rotation.conj().T @ mixed @ rotation
                mixed_eigenvalues = np.linalg.eigvalsh(mixed)
                traced_eigenvalues = _eigenvalues(
                    eigen_rows, iteration, channel, spin, kpoint
                ) / HA2EV
                maximum_trace_eigenvalue_difference = max(
                    maximum_trace_eigenvalue_difference,
                    _max_abs(mixed_eigenvalues - traced_eigenvalues),
                )
                maximum_diagonalization_offdiagonal = max(
                    maximum_diagonalization_offdiagonal,
                    _max_abs(diagonalized - np.diag(np.diag(diagonalized))),
                )
                maximum_diagonalization_eigenvalue_difference = max(
                    maximum_diagonalization_eigenvalue_difference,
                    _max_abs(np.diag(diagonalized).real - mixed_eigenvalues),
                )
                reference_components = _wavefunction_components(
                    blocks, 0, channel, spin, kpoint
                )
                live_components = _wavefunction_components(
                    blocks, iteration, channel, spin, kpoint
                )
                if live_components != reference_components:
                    raise BandValidationError(
                        "wavefunction spinor layout changed at iteration {}, channel {}, "
                        "spin {}, kpoint {}".format(iteration, channel, spin, kpoint)
                    )
                for component in reference_components:
                    reference_wavefunction = _matrix(
                        blocks, 0, channel, component, spin, kpoint
                    )
                    live_wavefunction = _matrix(
                        blocks, iteration, channel, component, spin, kpoint
                    )
                    predicted_wavefunction = rotation.T @ reference_wavefunction
                    block_maximum, block_relative = _check_difference(
                        live_wavefunction,
                        predicted_wavefunction,
                        "fixed-basis wavefunction rotation iter {} channel {} spin {} "
                        "kpoint {} component {}".format(
                            iteration, channel, spin, kpoint, component
                        ),
                        fixed_basis_tolerance,
                        fixed_basis_tolerance,
                    )
                    maximum_wavefunction_rotation_absolute = max(
                        maximum_wavefunction_rotation_absolute, block_maximum
                    )
                    maximum_wavefunction_rotation_relative = max(
                        maximum_wavefunction_rotation_relative, block_relative
                    )
        calculated_l2 = math.sqrt(grid_residual_square)
        residual_l2_difference = max(
            residual_l2_difference,
            abs(calculated_l2 - summaries[iteration]["residual_l2_Ha"]),
        )
        residual_max_difference = max(
            residual_max_difference,
            abs(grid_residual_maximum - summaries[iteration]["residual_max_Ha"]),
        )

    if maximum_hermiticity > hermiticity_tolerance:
        raise BandValidationError(
            "matrix Hermiticity exceeds tolerance: {:.6e}".format(maximum_hermiticity)
        )
    if maximum_unitarity > hermiticity_tolerance:
        raise BandValidationError(
            "rotation unitarity exceeds tolerance: {:.6e}".format(maximum_unitarity)
        )
    if maximum_trace_eigenvalue_difference > fixed_basis_tolerance:
        raise BandValidationError(
            "Hamiltonian eigenvalues differ from trace: {:.6e} Ha".format(
                maximum_trace_eigenvalue_difference
            )
        )
    if maximum_diagonalization_offdiagonal > fixed_basis_tolerance:
        raise BandValidationError(
            "rotated Hamiltonian offdiagonal exceeds tolerance: {:.6e} Ha".format(
                maximum_diagonalization_offdiagonal
            )
        )
    if maximum_diagonalization_eigenvalue_difference > fixed_basis_tolerance:
        raise BandValidationError(
            "rotated Hamiltonian diagonal differs from eigenvalues: {:.6e} Ha".format(
                maximum_diagonalization_eigenvalue_difference
            )
        )
    if residual_l2_difference > closure_absolute_tolerance or residual_max_difference > closure_absolute_tolerance:
        raise BandValidationError("iteration residual summary does not close")

    diagnostic_limits = {
        "basis_inverse_residual": 1.0e-10,
        "basis_condition_estimate": 1.0e12,
        "fourier_orthogonality_residual": 1.0e-10,
        "source_roundtrip_relative_error": 1.0e-10,
        "repaired_target_hermiticity_error": 1.0e-10,
    }
    diagnostics = {}
    for iteration in range(1, expected_iterations + 1):
        values = {}
        for component, limit in diagnostic_limits.items():
            matrix = _matrix(blocks, iteration, 1, component, 0, 0)
            if matrix.shape != (1, 1) or abs(matrix[0, 0].imag) > 1.0e-15:
                raise BandValidationError("invalid Fourier scalar {}".format(component))
            value = float(matrix[0, 0].real)
            if value < 0.0 or value > limit:
                raise BandValidationError(
                    "Fourier diagnostic {} exceeds tolerance: {:.6e} > {:.6e}"
                    .format(component, value, limit)
                )
            values[component] = value
        absolute_target = float(
            _matrix(blocks, iteration, 1, "target_hermiticity_error", 0, 0)[0, 0].real
        )
        relative_target = float(
            _matrix(blocks, iteration, 1, "target_relative_hermiticity_error", 0, 0)[0, 0].real
        )
        if min(absolute_target, relative_target) > 1.0e-10:
            raise BandValidationError("unrepaired Fourier target is outside both tolerances")
        values["target_hermiticity_error"] = absolute_target
        values["target_relative_hermiticity_error"] = relative_target
        diagnostics[str(iteration)] = values

    band_layout = sorted(layouts[1])
    spins = sorted({spin for spin, _kpoint in band_layout})
    n_band_kpoints = max(kpoint for _spin, kpoint in band_layout) + 1
    n_bands = _matrix(blocks, 0, 1, "h0", *band_layout[0]).shape[0]
    table_reports = {}
    csr_reports = []
    for iteration in range(1, expected_iterations + 1):
        table_reports[str(iteration)] = {}
        for spin in spins:
            names = {
                "ks": output_dir / "KS_band_spin_{}_{}.dat".format(spin + 1, iteration),
                "exx": output_dir / "EXX_band_spin_{}_{}.dat".format(spin + 1, iteration),
                "qsgw": output_dir / "QSGW_band_spin_{}_{}.dat".format(spin + 1, iteration),
            }
            tables = {
                label: _parse_band_table(path, n_band_kpoints, n_bands)
                for label, path in names.items()
            }
            for kpoint in range(n_band_kpoints):
                coordinate = _coordinates(eigen_rows, iteration, 1, spin, kpoint)
                h0 = _matrix(blocks, 0, 1, "h0", spin, kpoint)
                vxc = _matrix(blocks, 0, 1, "vxc_dft", spin, kpoint)
                exx = _matrix(blocks, iteration, 1, "exx", spin, kpoint)
                qsgw_eigenvalues = _eigenvalues(
                    eigen_rows, iteration, 1, spin, kpoint
                )
                reference_occupations = _matrix(
                    blocks, 0, 1, "occupation", spin, kpoint
                )[0].real
                occupation_scale = n_band_kpoints * len(spins)
                state_weight = float(np.max(reference_occupations))
                spinor_count = len(
                    {
                        key[2]
                        for key in blocks
                        if key[0] == 0
                        and key[1] == 1
                        and key[3] == spin
                        and key[4] == kpoint
                        and key[2].startswith("wfc_spinor")
                    }
                )
                if state_weight <= 1.0e-14:
                    state_weight = 2.0 / (occupation_scale * spinor_count)
                expected_occupations = {
                    "ks": reference_occupations * occupation_scale,
                    "exx": reference_occupations * occupation_scale,
                    "qsgw": np.where(
                        qsgw_eigenvalues <= summaries[iteration]["efermi_eV"],
                        state_weight * occupation_scale,
                        0.0,
                    ),
                }
                expected_energies = {
                    "ks": np.diag(h0).real * HA2EV,
                    "exx": np.diag(h0 - vxc + exx).real * HA2EV,
                    "qsgw": qsgw_eigenvalues,
                }
                for label, rows in tables.items():
                    row_coordinate, occupations, energies = rows[kpoint]
                    if max(abs(a - b) for a, b in zip(row_coordinate, coordinate)) > 1.0e-7:
                        raise BandValidationError("{} table coordinate differs".format(label))
                    if _max_abs(occupations - expected_occupations[label]) > band_table_tolerance_ev:
                        raise BandValidationError("{} table occupation differs".format(label))
                    if _max_abs(energies - expected_energies[label]) > band_table_tolerance_ev:
                        raise BandValidationError("{} table energy differs".format(label))
            table_reports[str(iteration)][str(spin)] = {
                label + "_gap_ev": _table_gap(table)
                for label, table in tables.items()
            }
            csr_path = output_dir / "hrs{}_nao_qsgw_iter_{:04d}.csr".format(
                spin + 1, iteration
            )
            csr_reports.append(
                _validate_csr_export(
                    csr_path,
                    blocks,
                    eigen_rows,
                    iteration,
                    spin,
                    periods,
                    csr_absolute_tolerance,
                    matrix_relative_tolerance,
                    hermiticity_tolerance,
                )
            )

    report = {
        "schema": SCHEMA,
        "passed": True,
        "contract": contract,
        "iterations": iterations,
        "active_limits": active_limits,
        "raw_closure_max_abs_ha": closure_maximum,
        "raw_closure_relative_frobenius": closure_relative,
        "mixing_closure_max_abs_ha": mixing_maximum,
        "mixing_closure_relative_frobenius": mixing_relative,
        "residual_l2_summary_max_abs_difference_ha": residual_l2_difference,
        "residual_max_summary_max_abs_difference_ha": residual_max_difference,
        "matrix_hermiticity_max_abs_ha": maximum_hermiticity,
        "rotation_unitarity_max_abs": maximum_unitarity,
        "trace_eigenvalue_max_abs_difference_ha": maximum_trace_eigenvalue_difference,
        "diagonalization_offdiagonal_max_abs_ha": maximum_diagonalization_offdiagonal,
        "diagonalization_eigenvalue_max_abs_difference_ha": (
            maximum_diagonalization_eigenvalue_difference
        ),
        "fixed_basis_wavefunction_rotation_max_abs": (
            maximum_wavefunction_rotation_absolute
        ),
        "fixed_basis_wavefunction_rotation_relative_frobenius": (
            maximum_wavefunction_rotation_relative
        ),
        "fourier_diagnostics": diagnostics,
        "band_tables": table_reports,
        "csr_exports": csr_reports,
        "thresholds": {
            "closure_absolute_ha": closure_absolute_tolerance,
            "matrix_relative_frobenius": matrix_relative_tolerance,
            "hermiticity": hermiticity_tolerance,
            "fixed_basis": fixed_basis_tolerance,
            "csr_absolute_ha": csr_absolute_tolerance,
            "band_table_ev": band_table_tolerance_ev,
        },
    }
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("matrix_trace", type=Path)
    parser.add_argument("eigenvalue_trace", type=Path)
    parser.add_argument("iteration_trace", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("bz_sampling", type=Path)
    parser.add_argument("report", type=Path)
    parser.add_argument("--expected-iterations", type=int, required=True)
    parser.add_argument("--expected-cut-mode", type=int, choices=(0, 1, 2), required=True)
    parser.add_argument("--expected-unoccupied-keep", type=int, required=True)
    parser.add_argument("--expected-shift-ha", type=float, required=True)
    args = parser.parse_args()
    try:
        report = validate_band_run(
            args.matrix_trace,
            args.eigenvalue_trace,
            args.iteration_trace,
            args.output_dir,
            args.bz_sampling,
            args.expected_iterations,
            args.expected_cut_mode,
            args.expected_unoccupied_keep,
            args.expected_shift_ha,
        )
    except (BandValidationError, ValueError, OSError) as error:
        report = {"schema": SCHEMA, "passed": False, "error": str(error)}
        args.report.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="ascii"
        )
        raise SystemExit(str(error))
    args.report.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="ascii"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
