#!/usr/bin/env python3
"""Compare two exact847 Scheme-A H0 checkpoint directories."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

tool_dir = Path(__file__).resolve().parent
if str(tool_dir) not in sys.path:
    sys.path.insert(0, str(tool_dir))
if not (tool_dir / "compare_legacy_band0_native_outputs_v1.py").is_file():
    sibling_tool_dir = tool_dir.parent / "fish-gate-a-symmetry-20260720"
    if not (sibling_tool_dir / "compare_legacy_band0_native_outputs_v1.py").is_file():
        raise ImportError("legacy exact847 matrix parser is unavailable")
    sys.path.insert(0, str(sibling_tool_dir))

import compare_legacy_band0_native_outputs_v1 as native
import compare_legacy_h0_candidate_trace_v2 as checkpoint


SCHEMA = "librpa-exact847-checkpoint-comparison-v1"
MAX_ABS_HA = 1.0e-6
RELATIVE_FROBENIUS = 1.0e-8
EIGENVALUE_MAX_ABS_HA = 1.0e-6


class ComparisonError(ValueError):
    pass


def read_checkpoint(
    root: Path,
    iteration: int,
    n_spins: int,
    n_kpoints: int,
    n_bands: int,
) -> np.ndarray:
    values = np.empty(
        (n_spins, n_kpoints, n_bands, n_bands), dtype=np.complex128
    )
    directory = root / f"iter_{iteration:05d}"
    for spin in range(n_spins):
        for kpoint in range(n_kpoints):
            path = directory / (
                f"H0_GW_spin_{spin + 1:02d}_k_{kpoint + 1:06d}.bin"
            )
            rows, columns, data = native.read_matz_binary(path)
            if (rows, columns) != (n_bands, n_bands):
                raise ComparisonError(
                    f"unexpected checkpoint dimensions in {path}: "
                    f"{rows}x{columns}"
                )
            values[spin, kpoint] = np.asarray(
                data, dtype=np.complex128
            ).reshape(n_bands, n_bands)
    return values


def metrics(reference: np.ndarray, observed: np.ndarray) -> dict[str, float]:
    difference = observed - reference
    scale = max(
        float(np.linalg.norm(reference.ravel())),
        float(np.linalg.norm(observed.ravel())),
        1.0e-30,
    )
    return {
        "max_abs_ha": float(np.max(np.abs(difference))),
        "relative_frobenius": float(np.linalg.norm(difference.ravel())) / scale,
    }


def analyze(reference_raw: np.ndarray, observed_raw: np.ndarray) -> dict[str, object]:
    if reference_raw.shape != observed_raw.shape:
        raise ComparisonError("checkpoint shapes differ")
    reference = checkpoint.legacy.hermitize_legacy_upper(reference_raw.copy())
    observed = checkpoint.legacy.hermitize_legacy_upper(observed_raw.copy())
    reference_eigenvalues, _ = checkpoint.legacy.eigensystem(reference_raw)
    observed_eigenvalues, _ = checkpoint.legacy.eigensystem(observed_raw)
    authoritative = metrics(reference, observed)
    eigenvalue_difference = float(
        np.max(np.abs(observed_eigenvalues - reference_eigenvalues))
    )
    parity = (
        authoritative["max_abs_ha"] <= MAX_ABS_HA
        and authoritative["relative_frobenius"] <= RELATIVE_FROBENIUS
        and eigenvalue_difference <= EIGENVALUE_MAX_ABS_HA
    )
    return {
        "schema": SCHEMA,
        "diagnostic_complete": True,
        "raw_full_matrix": metrics(reference_raw, observed_raw),
        "upper_triangle_hermitized": authoritative,
        "eigenvalue_max_abs_ha": eigenvalue_difference,
        "reference_raw_hermiticity_max_abs_ha": (
            checkpoint.legacy.hermiticity_max(reference_raw)
        ),
        "observed_raw_hermiticity_max_abs_ha": (
            checkpoint.legacy.hermiticity_max(observed_raw)
        ),
        "numerical_parity_passed": parity,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("reference_root", type=Path)
    parser.add_argument("observed_root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--iteration", type=int, default=1)
    parser.add_argument("--n-spins", type=int, default=1)
    parser.add_argument("--n-kpoints", type=int, default=8)
    parser.add_argument("--n-bands", type=int, default=44)
    args = parser.parse_args()
    try:
        reference = read_checkpoint(
            args.reference_root,
            args.iteration,
            args.n_spins,
            args.n_kpoints,
            args.n_bands,
        )
        observed = read_checkpoint(
            args.observed_root,
            args.iteration,
            args.n_spins,
            args.n_kpoints,
            args.n_bands,
        )
        report = analyze(reference, observed)
        report["inputs"] = {
            "reference_root": str(args.reference_root.resolve()),
            "observed_root": str(args.observed_root.resolve()),
            "iteration": args.iteration,
        }
    except (ComparisonError, native.ComparisonError, OSError, ValueError) as error:
        report = {
            "schema": SCHEMA,
            "diagnostic_complete": False,
            "numerical_parity_passed": False,
            "error": str(error),
        }
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))
    return 0 if report.get("diagnostic_complete") else 2


if __name__ == "__main__":
    raise SystemExit(main())
