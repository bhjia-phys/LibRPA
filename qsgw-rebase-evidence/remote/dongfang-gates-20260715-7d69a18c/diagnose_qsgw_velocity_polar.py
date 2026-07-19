#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

import numpy as np

from diagnose_headwing_basis import (
    hermiticity_metrics,
    maximum_abs,
    read_band_out,
    read_velocity,
    read_wfc,
    relative_frobenius,
)


def nearest_unitary(matrix):
    left, singular_values, right_h = np.linalg.svd(
        matrix, full_matrices=False
    )
    return left @ right_h, singular_values


def unitarity_max_abs(matrix):
    identity = np.eye(matrix.shape[0], dtype=np.complex128)
    return maximum_abs(matrix @ matrix.conj().T - identity)


def transform_velocity(transform, velocity):
    return transform.T @ velocity @ transform.conj()


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
    reference_velocity = read_velocity(dataset / "velocity_matrix")
    source_velocity = read_velocity(source_dir / "velocity_matrix")
    if reference_velocity.shape != source_velocity.shape:
        raise ValueError("reference and PyATB velocity dimensions differ")

    raw_velocity = np.empty_like(source_velocity)
    polar_velocity = np.empty_like(source_velocity)
    per_kpoint = []

    for kpoint in range(n_kpoints):
        reference = read_wfc(
            dataset / f"KS_eigenvector_{kpoint}.dat",
            n_bands,
            n_bands,
            kpoint,
        )
        source = read_wfc(
            source_dir / f"KS_eigenvector_{kpoint}.dat",
            n_bands,
            n_bands,
            kpoint,
        )
        raw_transform = np.linalg.solve(reference.T, source.T).T
        polar_transform, singular_values = nearest_unitary(raw_transform)

        for direction in range(3):
            raw_velocity[0, kpoint, direction] = transform_velocity(
                raw_transform, source_velocity[0, kpoint, direction]
            )
            polar_velocity[0, kpoint, direction] = transform_velocity(
                polar_transform, source_velocity[0, kpoint, direction]
            )

        reference_block = reference_velocity[0, kpoint]
        raw_block = raw_velocity[0, kpoint]
        polar_block = polar_velocity[0, kpoint]
        per_kpoint.append(
            {
                "kpoint": kpoint,
                "raw_unitarity_max_abs": unitarity_max_abs(raw_transform),
                "polar_unitarity_max_abs": unitarity_max_abs(polar_transform),
                "singular_value_max_abs_deviation": float(
                    np.max(np.abs(singular_values - 1.0))
                ),
                "raw_wfc_reconstruction_relative_frobenius": relative_frobenius(
                    source, raw_transform @ reference
                ),
                "polar_wfc_reconstruction_relative_frobenius": relative_frobenius(
                    source, polar_transform @ reference
                ),
                "raw_vs_polar_transform_relative_frobenius": relative_frobenius(
                    raw_transform, polar_transform
                ),
                "raw_vs_polar_velocity_maximum_absolute": maximum_abs(
                    raw_block - polar_block
                ),
                "raw_vs_polar_velocity_relative_frobenius": relative_frobenius(
                    raw_block, polar_block
                ),
                "raw_vs_reference_velocity_maximum_absolute": maximum_abs(
                    raw_block - reference_block
                ),
                "raw_vs_reference_velocity_relative_frobenius": relative_frobenius(
                    raw_block, reference_block
                ),
                "polar_vs_reference_velocity_maximum_absolute": maximum_abs(
                    polar_block - reference_block
                ),
                "polar_vs_reference_velocity_relative_frobenius": relative_frobenius(
                    polar_block, reference_block
                ),
            }
        )

    def aggregate_maximum(field):
        return max(record[field] for record in per_kpoint)

    report = {
        "schema": "qsgw-velocity-polar-diagnostic-v1",
        "dataset": str(dataset),
        "dimensions": {
            "n_kpoints": n_kpoints,
            "n_spins": 1,
            "n_bands": n_bands,
            "n_aos": n_bands,
        },
        "input_eigenvalues": {
            "maximum_absolute_difference_ha": maximum_abs(
                source_band["eigenvalues"] - reference_band["eigenvalues"]
            ),
            "relative_frobenius": relative_frobenius(
                source_band["eigenvalues"], reference_band["eigenvalues"]
            ),
        },
        "transform": {
            "raw_unitarity_maximum_absolute": aggregate_maximum(
                "raw_unitarity_max_abs"
            ),
            "polar_unitarity_maximum_absolute": aggregate_maximum(
                "polar_unitarity_max_abs"
            ),
            "singular_value_maximum_absolute_deviation": aggregate_maximum(
                "singular_value_max_abs_deviation"
            ),
            "raw_wfc_reconstruction_maximum_relative_frobenius": aggregate_maximum(
                "raw_wfc_reconstruction_relative_frobenius"
            ),
            "polar_wfc_reconstruction_maximum_relative_frobenius": aggregate_maximum(
                "polar_wfc_reconstruction_relative_frobenius"
            ),
            "raw_vs_polar_maximum_relative_frobenius": aggregate_maximum(
                "raw_vs_polar_transform_relative_frobenius"
            ),
        },
        "velocity": {
            "raw_vs_polar_maximum_absolute": maximum_abs(
                raw_velocity - polar_velocity
            ),
            "raw_vs_polar_relative_frobenius": relative_frobenius(
                raw_velocity, polar_velocity
            ),
            "raw_vs_reference_maximum_absolute": maximum_abs(
                raw_velocity - reference_velocity
            ),
            "raw_vs_reference_relative_frobenius": relative_frobenius(
                raw_velocity, reference_velocity
            ),
            "polar_vs_reference_maximum_absolute": maximum_abs(
                polar_velocity - reference_velocity
            ),
            "polar_vs_reference_relative_frobenius": relative_frobenius(
                polar_velocity, reference_velocity
            ),
            "source_hermiticity": hermiticity_metrics(source_velocity),
            "raw_hermiticity": hermiticity_metrics(raw_velocity),
            "polar_hermiticity": hermiticity_metrics(polar_velocity),
        },
        "per_kpoint": per_kpoint,
    }

    Path(args.output_json).write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    fields = (
        "kpoint",
        "raw_unitarity_max_abs",
        "polar_unitarity_max_abs",
        "singular_value_max_abs_deviation",
        "raw_wfc_reconstruction_relative_frobenius",
        "polar_wfc_reconstruction_relative_frobenius",
        "raw_vs_polar_transform_relative_frobenius",
        "raw_vs_polar_velocity_maximum_absolute",
        "raw_vs_polar_velocity_relative_frobenius",
        "raw_vs_reference_velocity_maximum_absolute",
        "raw_vs_reference_velocity_relative_frobenius",
        "polar_vs_reference_velocity_maximum_absolute",
        "polar_vs_reference_velocity_relative_frobenius",
    )
    with Path(args.output_tsv).open("w", encoding="utf-8", newline="\n") as stream:
        stream.write("\t".join(fields) + "\n")
        for record in per_kpoint:
            stream.write(
                "\t".join(
                    str(record[field])
                    if field == "kpoint"
                    else f"{record[field]:.17e}"
                    for field in fields
                )
                + "\n"
            )

    print(
        json.dumps(
            {
                "raw_unitarity_max": report["transform"][
                    "raw_unitarity_maximum_absolute"
                ],
                "polar_unitarity_max": report["transform"][
                    "polar_unitarity_maximum_absolute"
                ],
                "raw_vs_polar_velocity_max": report["velocity"][
                    "raw_vs_polar_maximum_absolute"
                ],
                "raw_vs_reference_velocity_max": report["velocity"][
                    "raw_vs_reference_maximum_absolute"
                ],
                "polar_vs_reference_velocity_max": report["velocity"][
                    "polar_vs_reference_maximum_absolute"
                ],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
