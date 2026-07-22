#!/usr/bin/env python3
"""Compare an instrumented exact847 component dump with a current trace."""

from __future__ import annotations

import argparse
import json
import math
import re
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
import diagnose_exact847_component_parity_v1 as diagnostic


SCHEMA = "librpa-exact847-current-component-comparison-v1"
STATIC_COMPONENTS = ("h0", "vxc_dft", "exx", "vc", "raw_h")


class ComparisonError(ValueError):
    pass


def read_matrix(path: Path, n_bands: int) -> np.ndarray:
    rows, columns, values = native.read_matz_binary(path)
    if (rows, columns) != (n_bands, n_bands):
        raise ComparisonError(
            f"unexpected matrix dimensions in {path}: {rows}x{columns}"
        )
    return np.asarray(values, dtype=np.complex128).reshape(n_bands, n_bands)


def read_metadata(path: Path) -> tuple[dict[str, int | str], np.ndarray]:
    scalar: dict[str, int | str] = {}
    frequencies: dict[int, float] = {}
    for line_number, raw_line in enumerate(path.read_text().splitlines(), 1):
        fields = raw_line.split()
        if not fields:
            continue
        if fields[0] == "schema" and len(fields) == 2:
            scalar["schema"] = fields[1]
        elif fields[0] == "frequency_ha" and len(fields) == 3:
            frequencies[int(fields[1])] = float(fields[2])
        elif len(fields) == 2:
            scalar[fields[0]] = int(fields[1])
        else:
            raise ComparisonError(f"{path}:{line_number}: malformed metadata")
    if scalar.get("schema") != "librpa-exact847-component-dump-v1":
        raise ComparisonError("legacy component metadata schema mismatch")
    n_frequencies = int(scalar.get("n_frequencies", -1))
    if sorted(frequencies) != list(range(n_frequencies)):
        raise ComparisonError("legacy component frequencies are incomplete")
    return scalar, np.asarray(
        [frequencies[index] for index in range(n_frequencies)], dtype=float
    )


def read_legacy_dump(
    directory: Path,
    n_spins: int,
    n_kpoints: int,
    n_bands: int,
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    metadata, frequencies = read_metadata(directory / "metadata.txt")
    expected = {
        "iteration": 1,
        "n_spins": n_spins,
        "n_kpoints": n_kpoints,
        "n_bands": n_bands,
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ComparisonError(
                f"legacy component metadata mismatch for {key}: "
                f"{metadata.get(key)!r} != {value!r}"
            )

    static = {
        component: np.empty(
            (n_spins, n_kpoints, n_bands, n_bands), dtype=np.complex128
        )
        for component in STATIC_COMPONENTS
    }
    sigma = np.empty(
        (len(frequencies), n_spins, n_kpoints, n_bands, n_bands),
        dtype=np.complex128,
    )
    for spin in range(n_spins):
        for kpoint in range(n_kpoints):
            suffix = f"spin_{spin + 1:02d}_k_{kpoint + 1:06d}.bin"
            for component in STATIC_COMPONENTS:
                static[component][spin, kpoint] = read_matrix(
                    directory / f"{component}_{suffix}", n_bands
                )
            for frequency_index in range(len(frequencies)):
                sigma[frequency_index, spin, kpoint] = read_matrix(
                    directory
                    / f"sigma_c_iw_{frequency_index:03d}_{suffix}",
                    n_bands,
                )
    return static, frequencies, sigma


def read_candidate_sigma_text(
    text: str,
    label: str,
    iteration: int,
    n_frequencies: int,
    n_spins: int,
    n_kpoints: int,
    n_bands: int,
) -> tuple[np.ndarray, np.ndarray]:
    values = np.empty(
        (n_frequencies, n_spins, n_kpoints, n_bands, n_bands),
        dtype=np.complex128,
    )
    frequencies = np.full(n_frequencies, np.nan, dtype=float)
    seen: set[tuple[int, int, int, int, int]] = set()
    for line_number, raw_line in enumerate(text.splitlines(), 1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) != 11:
            raise ComparisonError(f"{label}:{line_number}: expected 11 columns")
        if fields[2] != "sigma_c_iw":
            continue
        try:
            row_iteration = int(fields[0])
            channel = int(fields[1])
        except ValueError as error:
            raise ComparisonError(f"{label}:{line_number}: invalid index") from error
        if row_iteration != iteration or channel != 0:
            continue
        try:
            spin = int(fields[3])
            kpoint = int(fields[4])
            frequency_index = int(fields[5])
            frequency = float(fields[6])
            row = int(fields[7])
            column = int(fields[8])
            value = complex(float(fields[9]), float(fields[10]))
        except ValueError as error:
            raise ComparisonError(
                f"{label}:{line_number}: malformed Sigma row"
            ) from error
        if not (
            0 <= spin < n_spins
            and 0 <= kpoint < n_kpoints
            and 0 <= frequency_index < n_frequencies
            and 0 <= row < n_bands
            and 0 <= column < n_bands
        ):
            raise ComparisonError(f"{label}:{line_number}: Sigma index out of range")
        if not all(
            math.isfinite(item)
            for item in (frequency, value.real, value.imag)
        ):
            raise ComparisonError(f"{label}:{line_number}: non-finite Sigma row")
        key = (frequency_index, spin, kpoint, row, column)
        if key in seen:
            raise ComparisonError(f"{label}:{line_number}: duplicate Sigma row")
        seen.add(key)
        if math.isnan(frequencies[frequency_index]):
            frequencies[frequency_index] = frequency
        elif frequencies[frequency_index] != frequency:
            raise ComparisonError(
                f"{label}:{line_number}: inconsistent Sigma frequency"
            )
        values[key] = value
    expected = n_frequencies * n_spins * n_kpoints * n_bands * n_bands
    if len(seen) != expected:
        raise ComparisonError(
            f"candidate Sigma has {len(seen)} rows; expected {expected}"
        )
    if np.any(~np.isfinite(frequencies)):
        raise ComparisonError("candidate Sigma frequencies are incomplete")
    return frequencies, values


def upper(values: np.ndarray) -> np.ndarray:
    return checkpoint.legacy.hermitize_legacy_upper(values.copy())


def metrics(reference: np.ndarray, observed: np.ndarray) -> dict[str, object]:
    difference = observed - reference
    scale = max(
        float(np.linalg.norm(reference.ravel())),
        float(np.linalg.norm(observed.ravel())),
        1.0e-30,
    )
    flat_index = int(np.argmax(np.abs(difference)))
    location = np.unravel_index(flat_index, difference.shape)
    value = difference[location]
    return {
        "max_abs_ha": float(abs(value)),
        "relative_frobenius": float(np.linalg.norm(difference.ravel())) / scale,
        "maximum_difference_index": [int(index) for index in location],
        "maximum_difference_real_ha": float(value.real),
        "maximum_difference_imag_ha": float(value.imag),
    }


def analyze(
    legacy_static: dict[str, np.ndarray],
    candidate_static: dict[str, np.ndarray],
    legacy_frequencies: np.ndarray,
    candidate_frequencies: np.ndarray,
    legacy_sigma: np.ndarray,
    candidate_sigma: np.ndarray,
) -> dict[str, object]:
    if set(legacy_static) != set(STATIC_COMPONENTS):
        raise ComparisonError("legacy static component set mismatch")
    missing = set(STATIC_COMPONENTS) - set(candidate_static)
    if missing:
        raise ComparisonError(
            "candidate static components are missing: " + ", ".join(sorted(missing))
        )
    if legacy_frequencies.shape != candidate_frequencies.shape:
        raise ComparisonError("frequency-grid shape mismatch")
    frequency_difference = float(
        np.max(np.abs(candidate_frequencies - legacy_frequencies))
    )
    static_report = {}
    for component in STATIC_COMPONENTS:
        static_report[component] = {
            "raw": metrics(legacy_static[component], candidate_static[component]),
            "upper_triangle_hermitized": metrics(
                upper(legacy_static[component]), upper(candidate_static[component])
            ),
        }
    frequency_reports = [
        {
            "frequency_index": index,
            "frequency_ha": float(legacy_frequencies[index]),
            **metrics(legacy_sigma[index], candidate_sigma[index]),
        }
        for index in range(len(legacy_frequencies))
    ]
    sigma_metrics = metrics(legacy_sigma, candidate_sigma)
    return {
        "schema": SCHEMA,
        "diagnostic_complete": True,
        "frequency_grid_max_abs_ha": frequency_difference,
        "static_components": static_report,
        "sigma_c_iw": {
            "overall": sigma_metrics,
            "by_frequency": frequency_reports,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("legacy_dump", type=Path)
    parser.add_argument("candidate_trace", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--iteration", type=int, default=1)
    parser.add_argument("--n-frequencies", type=int, default=16)
    parser.add_argument("--n-spins", type=int, default=1)
    parser.add_argument("--n-kpoints", type=int, default=8)
    parser.add_argument("--n-bands", type=int, default=44)
    args = parser.parse_args()
    try:
        legacy_static, legacy_frequencies, legacy_sigma = read_legacy_dump(
            args.legacy_dump, args.n_spins, args.n_kpoints, args.n_bands
        )
        trace_text = args.candidate_trace.read_text(encoding="utf-8")
        candidate_all = diagnostic.parse_static_components_text(
            trace_text,
            str(args.candidate_trace),
            args.n_spins,
            args.n_kpoints,
            args.n_bands,
        )
        candidate_static = {
            component: candidate_all[component]
            for component in STATIC_COMPONENTS
        }
        candidate_frequencies, candidate_sigma = read_candidate_sigma_text(
            trace_text,
            str(args.candidate_trace),
            args.iteration,
            args.n_frequencies,
            args.n_spins,
            args.n_kpoints,
            args.n_bands,
        )
        report = analyze(
            legacy_static,
            candidate_static,
            legacy_frequencies,
            candidate_frequencies,
            legacy_sigma,
            candidate_sigma,
        )
        report["inputs"] = {
            "legacy_dump": str(args.legacy_dump.resolve()),
            "candidate_trace": str(args.candidate_trace.resolve()),
        }
    except (ComparisonError, diagnostic.DiagnosticError, OSError, ValueError) as error:
        report = {
            "schema": SCHEMA,
            "diagnostic_complete": False,
            "error": str(error),
        }
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))
    return 0 if report.get("diagnostic_complete") else 2


if __name__ == "__main__":
    raise SystemExit(main())
