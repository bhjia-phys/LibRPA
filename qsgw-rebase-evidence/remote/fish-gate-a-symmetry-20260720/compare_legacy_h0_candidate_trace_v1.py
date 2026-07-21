#!/usr/bin/env python3
"""Compare legacy H0 checkpoints with current QSGW fixed-basis traces."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
from collections import defaultdict
from pathlib import Path

import numpy as np

import compare_legacy_band0_native_outputs_v1 as native


SCHEMA = "librpa-legacy-h0-candidate-trace-comparison-v1"
HA2EV = 27.211386245988
MATRIX_MAX_ABS_HA = 1.0e-6
MATRIX_REL_FROBENIUS = 1.0e-8
EIGENVALUE_MAX_ABS_HA = 1.0e-6
GAP_MAX_ABS_EV = 1.0e-5
HERMITICITY_MAX_ABS_HA = 1.0e-10
ORTHOGONALITY_MAX_ABS = 1.0e-10


class ComparisonError(ValueError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_iterations(specification: str) -> list[int]:
    if ":" in specification:
        start_text, stop_text = specification.split(":", 1)
        start = int(start_text)
        stop = int(stop_text)
        if start < 1 or stop < start:
            raise ComparisonError("invalid iteration range")
        return list(range(start, stop + 1))
    values = sorted({int(value) for value in specification.split(",")})
    if not values or values[0] < 1:
        raise ComparisonError("invalid iteration list")
    return values


def require_contract_v5(path: Path) -> None:
    found = False
    for line in path.read_text().splitlines():
        if line.strip() == "# qsgw_contract_version 5":
            found = True
            break
    if not found:
        raise ComparisonError(f"missing contract-v5 header: {path}")


def materialize_blocks(
    rows: dict[tuple[int, str, int, int, int, int], complex],
    iterations: list[int],
    components: tuple[str, ...],
    n_spins: int,
    n_kpoints: int,
    n_bands: int,
) -> dict[str, dict[int, np.ndarray]]:
    result: dict[str, dict[int, np.ndarray]] = {}
    expected_count = n_spins * n_kpoints * n_bands * n_bands
    for component in components:
        result[component] = {}
        for iteration in iterations:
            selected = {
                (spin, kpoint, row, column): value
                for (row_iteration, row_component, spin, kpoint, row, column), value
                in rows.items()
                if row_iteration == iteration and row_component == component
            }
            if len(selected) != expected_count:
                raise ComparisonError(
                    f"{component} iteration {iteration} has {len(selected)} rows; "
                    f"expected {expected_count}"
                )
            blocks = np.empty(
                (n_spins, n_kpoints, n_bands, n_bands),
                dtype=np.complex128,
            )
            for spin in range(n_spins):
                for kpoint in range(n_kpoints):
                    for row in range(n_bands):
                        for column in range(n_bands):
                            key = (spin, kpoint, row, column)
                            if key not in selected:
                                raise ComparisonError(
                                    f"missing {component} row {iteration}:{key}"
                                )
                            blocks[spin, kpoint, row, column] = selected[key]
            result[component][iteration] = blocks
    return result


def read_candidate_matrices(
    path: Path,
    iterations: list[int],
    n_spins: int,
    n_kpoints: int,
    n_bands: int,
) -> dict[str, dict[int, np.ndarray]]:
    require_contract_v5(path)
    selected_iterations = set(iterations)
    components = ("mixed_h", "rotation_u")
    rows: dict[tuple[int, str, int, int, int, int], complex] = {}
    for line_number, raw_line in enumerate(path.read_text().splitlines(), 1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) != 11:
            raise ComparisonError(
                f"{path}:{line_number}: expected 11 matrix-trace columns"
            )
        try:
            iteration = int(fields[0])
            channel = int(fields[1])
        except ValueError as error:
            raise ComparisonError(
                f"{path}:{line_number}: invalid trace index"
            ) from error
        component = fields[2]
        if (
            iteration not in selected_iterations
            or channel != 0
            or component not in components
        ):
            continue
        try:
            spin = int(fields[3])
            kpoint = int(fields[4])
            frequency_index = int(fields[5])
            row = int(fields[7])
            column = int(fields[8])
            real = float(fields[9])
            imaginary = float(fields[10])
        except ValueError as error:
            raise ComparisonError(
                f"{path}:{line_number}: invalid selected matrix row"
            ) from error
        if frequency_index != -1:
            raise ComparisonError(
                f"{path}:{line_number}: non-static {component} row"
            )
        if not math.isfinite(real) or not math.isfinite(imaginary):
            raise ComparisonError(
                f"{path}:{line_number}: non-finite selected matrix row"
            )
        key = (iteration, component, spin, kpoint, row, column)
        if key in rows:
            raise ComparisonError(f"{path}:{line_number}: duplicate row {key}")
        rows[key] = complex(real, imaginary)
    return materialize_blocks(
        rows, iterations, components, n_spins, n_kpoints, n_bands
    )


def read_candidate_eigenvalues(
    path: Path,
    iterations: list[int],
    n_spins: int,
    n_kpoints: int,
    n_bands: int,
) -> dict[int, np.ndarray]:
    require_contract_v5(path)
    selected_iterations = set(iterations)
    values: dict[tuple[int, int, int, int], float] = {}
    for line_number, raw_line in enumerate(path.read_text().splitlines(), 1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) != 9:
            raise ComparisonError(
                f"{path}:{line_number}: expected 9 eigenvalue-trace columns"
            )
        try:
            iteration = int(fields[0])
            channel = int(fields[1])
        except ValueError as error:
            raise ComparisonError(
                f"{path}:{line_number}: invalid eigenvalue index"
            ) from error
        if iteration not in selected_iterations or channel != 0:
            continue
        try:
            spin = int(fields[2])
            kpoint = int(fields[3])
            band = int(fields[7])
            energy_ha = float(fields[8]) / HA2EV
        except ValueError as error:
            raise ComparisonError(
                f"{path}:{line_number}: invalid selected eigenvalue row"
            ) from error
        if not math.isfinite(energy_ha):
            raise ComparisonError(
                f"{path}:{line_number}: non-finite selected eigenvalue"
            )
        key = (iteration, spin, kpoint, band)
        if key in values:
            raise ComparisonError(
                f"{path}:{line_number}: duplicate eigenvalue row {key}"
            )
        values[key] = energy_ha

    result: dict[int, np.ndarray] = {}
    expected_count = n_spins * n_kpoints * n_bands
    for iteration in iterations:
        selected = {
            (spin, kpoint, band): value
            for (row_iteration, spin, kpoint, band), value in values.items()
            if row_iteration == iteration
        }
        if len(selected) != expected_count:
            raise ComparisonError(
                f"eigenvalue iteration {iteration} has {len(selected)} rows; "
                f"expected {expected_count}"
            )
        array = np.empty((n_spins, n_kpoints, n_bands), dtype=np.float64)
        for spin in range(n_spins):
            for kpoint in range(n_kpoints):
                for band in range(n_bands):
                    key = (spin, kpoint, band)
                    if key not in selected:
                        raise ComparisonError(
                            f"missing eigenvalue row {iteration}:{key}"
                        )
                    array[spin, kpoint, band] = selected[key]
        result[iteration] = array
    return result


def read_legacy_h0(
    checkpoint_root: Path,
    iteration: int,
    n_spins: int,
    n_kpoints: int,
    n_bands: int,
) -> np.ndarray:
    result = np.empty(
        (n_spins, n_kpoints, n_bands, n_bands), dtype=np.complex128
    )
    directory = checkpoint_root / f"iter_{iteration:05d}"
    for spin in range(n_spins):
        for kpoint in range(n_kpoints):
            path = directory / (
                f"H0_GW_spin_{spin + 1:02d}_k_{kpoint + 1:06d}.bin"
            )
            rows, columns, values = native.read_matz_binary(path)
            if (rows, columns) != (n_bands, n_bands):
                raise ComparisonError(
                    f"unexpected legacy H0 dimensions in {path}: "
                    f"{rows}x{columns}"
                )
            result[spin, kpoint] = np.asarray(
                values, dtype=np.complex128
            ).reshape((n_bands, n_bands))
    return result


def hermitize_legacy_upper(values: np.ndarray) -> np.ndarray:
    upper = np.triu(values)
    result = upper + np.swapaxes(np.triu(values, 1).conj(), -1, -2)
    indices = np.arange(result.shape[-1])
    result[..., indices, indices] = result[..., indices, indices].real
    return result


def hermiticity_max(values: np.ndarray) -> float:
    delta = values - np.swapaxes(values.conj(), -1, -2)
    return float(np.max(np.abs(delta)))


def orthogonality_max(values: np.ndarray) -> float:
    maximum = 0.0
    identity = np.eye(values.shape[-1], dtype=np.complex128)
    for spin in range(values.shape[0]):
        for kpoint in range(values.shape[1]):
            unitary = values[spin, kpoint]
            residual = unitary.conj().T @ unitary - identity
            maximum = max(maximum, float(np.max(np.abs(residual))))
    return maximum


def matrix_metrics(reference: np.ndarray, observed: np.ndarray) -> tuple[float, float]:
    difference = observed - reference
    maximum = float(np.max(np.abs(difference)))
    difference_norm = float(np.linalg.norm(difference.ravel()))
    scale = max(
        float(np.linalg.norm(reference.ravel())),
        float(np.linalg.norm(observed.ravel())),
        1.0e-30,
    )
    return maximum, difference_norm / scale


def eigensystem(values: np.ndarray) -> tuple[np.ndarray, float]:
    eigenvalues = np.empty(values.shape[:-1], dtype=np.float64)
    maximum_orthogonality = 0.0
    identity = np.eye(values.shape[-1], dtype=np.complex128)
    for spin in range(values.shape[0]):
        for kpoint in range(values.shape[1]):
            block_values, vectors = np.linalg.eigh(
                values[spin, kpoint], UPLO="U"
            )
            eigenvalues[spin, kpoint] = block_values
            residual = vectors.conj().T @ vectors - identity
            maximum_orthogonality = max(
                maximum_orthogonality,
                float(np.max(np.abs(residual))),
            )
    return eigenvalues, maximum_orthogonality


def band_gap_ev(eigenvalues: np.ndarray, occupied_bands: int) -> float:
    if occupied_bands < 1 or occupied_bands >= eigenvalues.shape[-1]:
        raise ComparisonError("occupied band count is outside the band range")
    valence_maximum = float(np.max(eigenvalues[..., occupied_bands - 1]))
    conduction_minimum = float(np.min(eigenvalues[..., occupied_bands]))
    return (conduction_minimum - valence_maximum) * HA2EV


def compare_iteration(
    legacy_raw: np.ndarray,
    candidate_h: np.ndarray,
    candidate_rotation: np.ndarray,
    candidate_eigenvalues: np.ndarray,
    occupied_bands: int,
) -> dict[str, object]:
    legacy_h = hermitize_legacy_upper(legacy_raw.copy())
    matrix_maximum, matrix_relative = matrix_metrics(legacy_h, candidate_h)
    legacy_eigenvalues, legacy_solver_orthogonality = eigensystem(legacy_raw)
    candidate_matrix_eigenvalues, candidate_solver_orthogonality = eigensystem(
        candidate_h
    )
    eigenvalue_maximum = float(
        np.max(np.abs(legacy_eigenvalues - candidate_eigenvalues))
    )
    candidate_trace_consistency = float(
        np.max(np.abs(candidate_matrix_eigenvalues - candidate_eigenvalues))
    )
    legacy_gap = band_gap_ev(legacy_eigenvalues, occupied_bands)
    candidate_gap = band_gap_ev(candidate_eigenvalues, occupied_bands)
    gap_difference = abs(candidate_gap - legacy_gap)
    legacy_raw_hermiticity = hermiticity_max(legacy_raw)
    candidate_hermiticity = hermiticity_max(candidate_h)
    candidate_rotation_orthogonality = orthogonality_max(candidate_rotation)

    parity_passed = (
        matrix_maximum <= MATRIX_MAX_ABS_HA
        and matrix_relative <= MATRIX_REL_FROBENIUS
        and eigenvalue_maximum <= EIGENVALUE_MAX_ABS_HA
        and candidate_trace_consistency <= EIGENVALUE_MAX_ABS_HA
        and gap_difference <= GAP_MAX_ABS_EV
    )
    candidate_invariants_passed = (
        candidate_hermiticity <= HERMITICITY_MAX_ABS_HA
        and candidate_rotation_orthogonality <= ORTHOGONALITY_MAX_ABS
        and candidate_solver_orthogonality <= ORTHOGONALITY_MAX_ABS
    )
    return {
        "matrix_max_abs_ha": matrix_maximum,
        "matrix_relative_frobenius": matrix_relative,
        "eigenvalue_max_abs_ha": eigenvalue_maximum,
        "candidate_trace_eigenvalue_consistency_max_abs_ha": (
            candidate_trace_consistency
        ),
        "legacy_gap_ev": legacy_gap,
        "candidate_gap_ev": candidate_gap,
        "gap_abs_diff_ev": gap_difference,
        "legacy_raw_hermiticity_max_abs_ha": legacy_raw_hermiticity,
        "legacy_upper_solver_orthogonality_max_abs": (
            legacy_solver_orthogonality
        ),
        "candidate_hermiticity_max_abs_ha": candidate_hermiticity,
        "candidate_rotation_orthogonality_max_abs": (
            candidate_rotation_orthogonality
        ),
        "candidate_solver_orthogonality_max_abs": (
            candidate_solver_orthogonality
        ),
        "legacy_oracle_invariant_gap": (
            legacy_raw_hermiticity > HERMITICITY_MAX_ABS_HA
        ),
        "parity_passed": parity_passed,
        "candidate_invariants_passed": candidate_invariants_passed,
        "passed": parity_passed and candidate_invariants_passed,
    }


def compare_outputs(
    checkpoint_root: Path,
    matrix_trace: Path,
    eigenvalue_trace: Path,
    iterations: list[int],
    n_spins: int,
    n_kpoints: int,
    n_bands: int,
    occupied_bands: int,
) -> dict[str, object]:
    checkpoint_root = checkpoint_root.resolve(strict=True)
    matrix_trace = matrix_trace.resolve(strict=True)
    eigenvalue_trace = eigenvalue_trace.resolve(strict=True)
    candidate_matrices = read_candidate_matrices(
        matrix_trace, iterations, n_spins, n_kpoints, n_bands
    )
    candidate_eigenvalues = read_candidate_eigenvalues(
        eigenvalue_trace, iterations, n_spins, n_kpoints, n_bands
    )
    reports: dict[str, dict[str, object]] = {}
    for iteration in iterations:
        reports[str(iteration)] = compare_iteration(
            read_legacy_h0(
                checkpoint_root, iteration, n_spins, n_kpoints, n_bands
            ),
            candidate_matrices["mixed_h"][iteration],
            candidate_matrices["rotation_u"][iteration],
            candidate_eigenvalues[iteration],
            occupied_bands,
        )
    parity_passed = all(report["parity_passed"] for report in reports.values())
    candidate_invariants_passed = all(
        report["candidate_invariants_passed"] for report in reports.values()
    )
    dependency = Path(native.__file__).resolve(strict=True)
    return {
        "schema": SCHEMA,
        "legacy_checkpoint_root": str(checkpoint_root),
        "candidate_matrix_trace": str(matrix_trace),
        "candidate_eigenvalue_trace": str(eigenvalue_trace),
        "iterations": iterations,
        "layout": {
            "n_spins": n_spins,
            "n_kpoints": n_kpoints,
            "n_bands": n_bands,
            "occupied_bands": occupied_bands,
        },
        "thresholds": {
            "matrix_max_abs_ha": MATRIX_MAX_ABS_HA,
            "matrix_relative_frobenius": MATRIX_REL_FROBENIUS,
            "eigenvalue_max_abs_ha": EIGENVALUE_MAX_ABS_HA,
            "gap_max_abs_ev": GAP_MAX_ABS_EV,
            "hermiticity_max_abs_ha": HERMITICITY_MAX_ABS_HA,
            "orthogonality_max_abs": ORTHOGONALITY_MAX_ABS,
        },
        "iteration_reports": reports,
        "legacy_oracle_invariant_gap": any(
            report["legacy_oracle_invariant_gap"] for report in reports.values()
        ),
        "parity_passed": parity_passed,
        "candidate_invariants_passed": candidate_invariants_passed,
        "passed": parity_passed and candidate_invariants_passed,
        "parser_dependency": {
            "path": str(dependency),
            "sha256": sha256_file(dependency),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("legacy_checkpoint_root", type=Path)
    parser.add_argument("candidate_matrix_trace", type=Path)
    parser.add_argument("candidate_eigenvalue_trace", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--iterations", default="1:2")
    parser.add_argument("--n-spins", type=int, default=1)
    parser.add_argument("--n-kpoints", type=int, default=8)
    parser.add_argument("--n-bands", type=int, default=44)
    parser.add_argument("--occupied-bands", type=int, default=4)
    args = parser.parse_args()
    try:
        if args.n_spins < 1 or args.n_kpoints < 1 or args.n_bands < 2:
            raise ComparisonError("invalid matrix layout")
        report = compare_outputs(
            args.legacy_checkpoint_root,
            args.candidate_matrix_trace,
            args.candidate_eigenvalue_trace,
            parse_iterations(args.iterations),
            args.n_spins,
            args.n_kpoints,
            args.n_bands,
            args.occupied_bands,
        )
    except (
        ComparisonError,
        native.ComparisonError,
        OSError,
        ValueError,
        KeyError,
        struct.error,
    ) as error:
        report = {"schema": SCHEMA, "passed": False, "error": str(error)}
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))
    return 0 if report.get("passed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
