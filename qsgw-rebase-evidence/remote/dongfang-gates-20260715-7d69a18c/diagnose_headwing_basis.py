#!/usr/bin/env python3

import argparse
import json
import math
import re
from pathlib import Path

import numpy as np


COMPLEX_TOKEN = re.compile(r"\(([^,]+),([^\)]+)\)")


def read_band_out(path):
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    if len(lines) < 6:
        raise ValueError(f"truncated band_out: {path}")
    n_kpoints, n_spins, n_bands, n_aos = [
        int(lines[index].strip()) for index in range(4)
    ]
    cursor = 5
    eigenvalues = np.empty((n_spins, n_kpoints, n_bands), dtype=float)
    occupations = np.empty_like(eigenvalues)
    for kpoint in range(n_kpoints):
        for spin in range(n_spins):
            header = lines[cursor].split()
            cursor += 1
            if len(header) != 2:
                raise ValueError(f"invalid band_out block header: {path}:{cursor}")
            for band in range(n_bands):
                fields = lines[cursor].split()
                cursor += 1
                if len(fields) < 4 or int(fields[0]) != band + 1:
                    raise ValueError(f"invalid band_out row: {path}:{cursor}")
                occupations[spin, kpoint, band] = float(fields[1])
                eigenvalues[spin, kpoint, band] = float(fields[2])
    if any(line.strip() for line in lines[cursor:]):
        raise ValueError(f"unexpected trailing band_out data: {path}")
    return {
        "n_kpoints": n_kpoints,
        "n_spins": n_spins,
        "n_bands": n_bands,
        "n_aos": n_aos,
        "efermi": float(lines[4].strip()),
        "eigenvalues": eigenvalues,
        "occupations": occupations,
    }


def read_wfc(path, n_bands, n_aos, expected_kpoint):
    tokens = Path(path).read_text(encoding="utf-8").split()
    expected = 1 + 2 * n_bands * n_aos
    if len(tokens) != expected:
        raise ValueError(
            f"WFC token count differs: {path}: {len(tokens)} != {expected}"
        )
    if int(tokens[0]) != expected_kpoint + 1:
        raise ValueError(f"WFC k-point index differs: {path}")
    values = np.asarray(
        [float(value.replace("D", "E").replace("d", "e")) for value in tokens[1:]],
        dtype=float,
    )
    coefficients = values[0::2] + 1j * values[1::2]
    # Legacy text order is AO-major with band as the inner index.
    return coefficients.reshape(n_aos, n_bands).T.copy()


def read_upper_overlap(path, dimension):
    text = Path(path).read_text(encoding="utf-8")
    first_line = text.splitlines()[0].strip()
    if int(first_line) != dimension:
        raise ValueError(f"overlap dimension differs: {path}")
    values = [
        complex(
            float(real.replace("D", "E").replace("d", "e")),
            float(imag.replace("D", "E").replace("d", "e")),
        )
        for real, imag in COMPLEX_TOKEN.findall(text)
    ]
    expected = dimension * (dimension + 1) // 2
    if len(values) != expected:
        raise ValueError(
            f"overlap packed count differs: {path}: {len(values)} != {expected}"
        )
    overlap = np.zeros((dimension, dimension), dtype=np.complex128)
    cursor = 0
    for row in range(dimension):
        for column in range(row, dimension):
            overlap[row, column] = values[cursor]
            overlap[column, row] = np.conj(values[cursor])
            cursor += 1
    return overlap


def read_velocity(path):
    tokens = Path(path).read_text(encoding="utf-8").split()
    if len(tokens) < 4:
        raise ValueError(f"truncated velocity matrix: {path}")
    n_kpoints, n_spins, n_bands, n_aos = map(int, tokens[:4])
    if n_bands != n_aos:
        raise ValueError(f"velocity matrix is not complete-band: {path}")
    result = np.empty(
        (n_spins, n_kpoints, 3, n_bands, n_bands), dtype=np.complex128
    )
    seen = set()
    cursor = 4
    matrix_size = n_bands * n_bands
    for _ in range(n_kpoints * n_spins * 3):
        direction, kpoint, spin = [int(value) - 1 for value in tokens[cursor:cursor + 3]]
        cursor += 3
        key = (spin, kpoint, direction)
        if key in seen:
            raise ValueError(f"duplicate velocity block: {path}: {key}")
        seen.add(key)
        raw = np.asarray(
            [
                float(value.replace("D", "E").replace("d", "e"))
                for value in tokens[cursor:cursor + 2 * matrix_size]
            ],
            dtype=float,
        )
        cursor += 2 * matrix_size
        values = raw[0::2] + 1j * raw[1::2]
        result[spin, kpoint, direction] = values.reshape(n_bands, n_bands)
    if cursor != len(tokens):
        raise ValueError(f"unexpected trailing velocity data: {path}")
    return result


def relative_frobenius(left, right):
    denominator = max(np.linalg.norm(left), np.linalg.norm(right), np.finfo(float).tiny)
    return float(np.linalg.norm(left - right) / denominator)


def maximum_abs(matrix):
    return float(np.max(np.abs(matrix)))


def phase_metrics(source, reference):
    residuals = []
    phases = []
    for band in range(reference.shape[0]):
        overlap = np.vdot(reference[band], source[band])
        if abs(overlap) == 0.0:
            phases.append(complex(float("nan"), float("nan")))
            residuals.append(float("inf"))
            continue
        phase = overlap / abs(overlap)
        denominator = max(
            np.vdot(source[band], source[band]).real,
            np.vdot(reference[band], reference[band]).real,
        )
        residual = np.linalg.norm(source[band] - phase * reference[band]) / math.sqrt(
            denominator
        )
        phases.append(phase)
        residuals.append(float(residual))
    return np.asarray(phases), np.asarray(residuals)


def hermiticity_metrics(matrix):
    delta = matrix - np.swapaxes(matrix.conj(), -1, -2)
    return {
        "maximum_absolute": maximum_abs(delta),
        "relative_frobenius": float(
            np.linalg.norm(delta) / max(np.linalg.norm(matrix), np.finfo(float).tiny)
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    parser.add_argument("output_json")
    parser.add_argument("output_tsv")
    args = parser.parse_args()

    dataset = Path(args.dataset).resolve()
    source_dir = dataset / "pyatb_librpa_df"
    reference_band = read_band_out(dataset / "band_out")
    source_band = read_band_out(source_dir / "band_out")
    shape_keys = ("n_kpoints", "n_spins", "n_bands", "n_aos")
    if any(reference_band[key] != source_band[key] for key in shape_keys):
        raise ValueError("reference and PyATB band dimensions differ")
    if reference_band["n_spins"] != 1:
        raise ValueError("diagnostic currently requires one spin channel")
    if reference_band["n_bands"] != reference_band["n_aos"]:
        raise ValueError("diagnostic requires a complete square WFC basis")

    n_kpoints = reference_band["n_kpoints"]
    n_bands = reference_band["n_bands"]
    identity = np.eye(n_bands, dtype=np.complex128)
    degeneracy_tolerances = (1.0e-8, 1.0e-7, 1.0e-6, 1.0e-5)
    per_kpoint = []
    per_band_rows = []
    transforms = []

    for kpoint in range(n_kpoints):
        reference = read_wfc(
            dataset / f"KS_eigenvector_{kpoint}.dat", n_bands, n_bands, kpoint
        )
        source = read_wfc(
            source_dir / f"KS_eigenvector_{kpoint}.dat", n_bands, n_bands, kpoint
        )
        overlap = read_upper_overlap(dataset / f"sks1k{kpoint + 1}_nao.txt", n_bands)

        reference_gram = reference.conj() @ overlap @ reference.T
        source_gram = source.conj() @ overlap @ source.T
        transform_direct = np.linalg.solve(reference.T, source.T).T
        # <r_a|s_i> = T_ia, so transpose the physical overlap table.
        transform_metric = (reference.conj() @ overlap @ source.T).T
        transforms.append(transform_metric)
        reconstructed = transform_metric @ reference
        transform_unitarity = transform_metric @ transform_metric.conj().T
        phases, phase_residuals = phase_metrics(source, reference)
        eigen_reference = reference_band["eigenvalues"][0, kpoint]
        eigen_source = source_band["eigenvalues"][0, kpoint]

        leakage = {}
        for tolerance in degeneracy_tolerances:
            allowed = np.abs(eigen_source[:, None] - eigen_reference[None, :]) <= tolerance
            rejected = np.where(allowed, 0.0, transform_metric)
            leakage[f"{tolerance:.0e}"] = {
                "maximum_absolute": maximum_abs(rejected),
                "relative_frobenius": float(
                    np.linalg.norm(rejected)
                    / max(np.linalg.norm(transform_metric), np.finfo(float).tiny)
                ),
            }

        dominant = np.argmax(np.abs(transform_metric), axis=1)
        for band in range(n_bands):
            per_band_rows.append(
                {
                    "kpoint": kpoint,
                    "source_band": band,
                    "reference_band": int(dominant[band]),
                    "dominant_amplitude": float(abs(transform_metric[band, dominant[band]])),
                    "phase_residual": float(phase_residuals[band]),
                    "eigenvalue_difference_ha": float(eigen_source[band] - eigen_reference[band]),
                }
            )

        offdiagonal = transform_metric.copy()
        np.fill_diagonal(offdiagonal, 0.0)
        per_kpoint.append(
            {
                "kpoint": kpoint,
                "reference_condition_number": float(np.linalg.cond(reference)),
                "overlap_condition_number": float(np.linalg.cond(overlap)),
                "reference_orthonormality_max_abs": maximum_abs(reference_gram - identity),
                "source_orthonormality_max_abs": maximum_abs(source_gram - identity),
                "transform_direct_vs_metric_relative_frobenius": relative_frobenius(
                    transform_direct, transform_metric
                ),
                "transform_reconstruction_relative_frobenius": relative_frobenius(
                    source, reconstructed
                ),
                "transform_unitarity_max_abs": maximum_abs(transform_unitarity - identity),
                "transform_unitarity_relative_frobenius": relative_frobenius(
                    transform_unitarity, identity
                ),
                "transform_offdiagonal_max_abs": maximum_abs(offdiagonal),
                "phase_residual_maximum": float(np.max(phase_residuals)),
                "phase_residual_band": int(np.argmax(phase_residuals)),
                "eigenvalue_max_abs_difference_ha": maximum_abs(
                    eigen_source - eigen_reference
                ),
                "degeneracy_block_leakage": leakage,
            }
        )

    velocity_reference_file = read_velocity(dataset / "velocity_matrix")
    velocity_source = read_velocity(source_dir / "velocity_matrix")
    if velocity_reference_file.shape != velocity_source.shape:
        raise ValueError("top-level and PyATB velocity dimensions differ")
    transformed_velocity = np.empty_like(velocity_source)
    for kpoint, transform in enumerate(transforms):
        for direction in range(3):
            transformed_velocity[0, kpoint, direction] = (
                transform.T
                @ velocity_source[0, kpoint, direction]
                @ transform.conj()
            )

    def aggregate_maximum(field):
        return max(record[field] for record in per_kpoint)

    worst_phase = max(per_band_rows, key=lambda record: record["phase_residual"])
    report = {
        "schema": "qsgw-headwing-basis-diagnostic-v1",
        "dataset": str(dataset),
        "dimensions": {
            "n_kpoints": n_kpoints,
            "n_spins": 1,
            "n_bands": n_bands,
            "n_aos": n_bands,
        },
        "eigenvalues": {
            "maximum_absolute_difference_ha": maximum_abs(
                source_band["eigenvalues"] - reference_band["eigenvalues"]
            ),
            "relative_frobenius": relative_frobenius(
                source_band["eigenvalues"], reference_band["eigenvalues"]
            ),
            "occupation_maximum_absolute_difference": maximum_abs(
                source_band["occupations"] - reference_band["occupations"]
            ),
        },
        "current_phase_only_check": {
            "maximum_relative_wfc_residual": worst_phase["phase_residual"],
            "worst_kpoint": worst_phase["kpoint"],
            "worst_band": worst_phase["source_band"],
            "threshold_in_candidate": 1.0e-8,
            "passes_candidate_threshold": worst_phase["phase_residual"] <= 1.0e-8,
        },
        "complete_basis_transform": {
            "maximum_reference_orthonormality_error": aggregate_maximum(
                "reference_orthonormality_max_abs"
            ),
            "maximum_source_orthonormality_error": aggregate_maximum(
                "source_orthonormality_max_abs"
            ),
            "maximum_direct_vs_metric_relative_frobenius": aggregate_maximum(
                "transform_direct_vs_metric_relative_frobenius"
            ),
            "maximum_reconstruction_relative_frobenius": aggregate_maximum(
                "transform_reconstruction_relative_frobenius"
            ),
            "maximum_unitarity_error": aggregate_maximum(
                "transform_unitarity_max_abs"
            ),
            "maximum_unitarity_relative_frobenius": aggregate_maximum(
                "transform_unitarity_relative_frobenius"
            ),
            "maximum_offdiagonal_amplitude": aggregate_maximum(
                "transform_offdiagonal_max_abs"
            ),
            "maximum_reference_condition_number": aggregate_maximum(
                "reference_condition_number"
            ),
            "maximum_overlap_condition_number": aggregate_maximum(
                "overlap_condition_number"
            ),
            "degeneracy_block_leakage_maxima": {
                f"{tolerance:.0e}": {
                    "maximum_absolute": max(
                        record["degeneracy_block_leakage"][f"{tolerance:.0e}"][
                            "maximum_absolute"
                        ]
                        for record in per_kpoint
                    ),
                    "relative_frobenius": max(
                        record["degeneracy_block_leakage"][f"{tolerance:.0e}"][
                            "relative_frobenius"
                        ]
                        for record in per_kpoint
                    ),
                }
                for tolerance in degeneracy_tolerances
            },
        },
        "velocity": {
            "top_level_vs_pyatb_maximum_absolute": maximum_abs(
                velocity_reference_file - velocity_source
            ),
            "top_level_vs_pyatb_relative_frobenius": relative_frobenius(
                velocity_reference_file, velocity_source
            ),
            "source_hermiticity": hermiticity_metrics(velocity_source),
            "transformed_hermiticity": hermiticity_metrics(transformed_velocity),
            "transformed_maximum_absolute": maximum_abs(transformed_velocity),
        },
        "per_kpoint": per_kpoint,
    }

    Path(args.output_json).write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with Path(args.output_tsv).open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(
            "kpoint\tsource_band\treference_band\tdominant_amplitude\t"
            "phase_residual\teigenvalue_difference_ha\n"
        )
        for row in per_band_rows:
            stream.write(
                f"{row['kpoint']}\t{row['source_band']}\t{row['reference_band']}\t"
                f"{row['dominant_amplitude']:.17e}\t{row['phase_residual']:.17e}\t"
                f"{row['eigenvalue_difference_ha']:.17e}\n"
            )

    print(json.dumps({
        "phase_max": report["current_phase_only_check"]["maximum_relative_wfc_residual"],
        "unitarity_max": report["complete_basis_transform"]["maximum_unitarity_error"],
        "reconstruction_rel": report["complete_basis_transform"][
            "maximum_reconstruction_relative_frobenius"
        ],
        "velocity_rel": report["velocity"]["top_level_vs_pyatb_relative_frobenius"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
