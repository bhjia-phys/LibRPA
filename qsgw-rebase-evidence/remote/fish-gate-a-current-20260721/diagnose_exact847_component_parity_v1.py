#!/usr/bin/env python3
"""Localize exact847/current QSGW one-update Hamiltonian differences."""

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

import compare_legacy_h0_candidate_trace_v2 as checkpoint


SCHEMA = "librpa-exact847-component-parity-diagnostic-v1"
HA2EV = 27.211386245988
MATRIX_COMPONENTS = {
    "h0": 0,
    "vxc_dft": 0,
    "exx": 1,
    "vc": 1,
    "raw_h": 1,
    "mixed_h": 1,
}


class DiagnosticError(ValueError):
    pass


def matrix_metrics(reference: np.ndarray, observed: np.ndarray) -> dict[str, float]:
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


def maximum_location(values: np.ndarray) -> dict[str, object]:
    flat_index = int(np.argmax(np.abs(values)))
    spin, kpoint, row, column = np.unravel_index(flat_index, values.shape)
    value = values[spin, kpoint, row, column]
    return {
        "spin": int(spin),
        "kpoint": int(kpoint),
        "row": int(row),
        "column": int(column),
        "real_ha": float(value.real),
        "imag_ha": float(value.imag),
        "abs_ha": float(abs(value)),
    }


def parse_static_components_text(
    text: str,
    label: str,
    n_spins: int,
    n_kpoints: int,
    n_bands: int,
) -> dict[str, np.ndarray]:
    if "# qsgw_contract_version 6" not in text.splitlines():
        raise DiagnosticError(f"{label}: missing contract-v6 header")
    result = {
        component: np.empty(
            (n_spins, n_kpoints, n_bands, n_bands), dtype=np.complex128
        )
        for component in MATRIX_COMPONENTS
    }
    seen: dict[str, set[tuple[int, int, int, int]]] = {
        component: set() for component in MATRIX_COMPONENTS
    }
    for line_number, raw_line in enumerate(text.splitlines(), 1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) != 11:
            raise DiagnosticError(f"{label}:{line_number}: expected 11 columns")
        try:
            iteration = int(fields[0])
            channel = int(fields[1])
        except ValueError as error:
            raise DiagnosticError(
                f"{label}:{line_number}: invalid trace index"
            ) from error
        component = fields[2]
        if component not in MATRIX_COMPONENTS:
            continue
        if iteration != MATRIX_COMPONENTS[component] or channel != 0:
            continue
        try:
            spin = int(fields[3])
            kpoint = int(fields[4])
            frequency_index = int(fields[5])
            row = int(fields[7])
            column = int(fields[8])
            value = complex(float(fields[9]), float(fields[10]))
        except ValueError as error:
            raise DiagnosticError(
                f"{label}:{line_number}: invalid static matrix row"
            ) from error
        if frequency_index != -1:
            raise DiagnosticError(
                f"{label}:{line_number}: selected component is not static"
            )
        if not (math.isfinite(value.real) and math.isfinite(value.imag)):
            raise DiagnosticError(f"{label}:{line_number}: non-finite value")
        key = (spin, kpoint, row, column)
        if key in seen[component]:
            raise DiagnosticError(
                f"{label}:{line_number}: duplicate {component} row {key}"
            )
        if not (
            0 <= spin < n_spins
            and 0 <= kpoint < n_kpoints
            and 0 <= row < n_bands
            and 0 <= column < n_bands
        ):
            raise DiagnosticError(
                f"{label}:{line_number}: {component} index is out of range"
            )
        seen[component].add(key)
        result[component][spin, kpoint, row, column] = value

    expected = n_spins * n_kpoints * n_bands * n_bands
    for component, keys in seen.items():
        if len(keys) != expected:
            raise DiagnosticError(
                f"{label}: {component} has {len(keys)} rows; expected {expected}"
            )
    return result


def parse_static_components(
    path: Path,
    n_spins: int,
    n_kpoints: int,
    n_bands: int,
) -> dict[str, np.ndarray]:
    return parse_static_components_text(
        path.read_text(encoding="utf-8"),
        str(path),
        n_spins,
        n_kpoints,
        n_bands,
    )


def parse_legacy_state_table_text(
    text: str,
    n_spins: int,
    n_kpoints: int,
    n_bands: int,
) -> dict[str, np.ndarray]:
    marker = "Final Quasi-Particle Energy after QSGW Iterations"
    sections = text.split(marker)
    if len(sections) < 2:
        raise DiagnosticError("legacy stdout has no QSGW state table")

    spin_kpoint = re.compile(r"spin\s+(\d+),\s+k-point\s+(\d+):")
    expected = n_spins * n_kpoints * n_bands
    for section in sections[1:]:
        values: dict[tuple[int, int, int], tuple[float, ...]] = {}
        current: tuple[int, int] | None = None
        for raw_line in section.splitlines():
            match = spin_kpoint.search(raw_line)
            if match:
                current = (int(match.group(1)) - 1, int(match.group(2)) - 1)
                continue
            fields = raw_line.split()
            if current is None or len(fields) < 7 or not fields[0].isdigit():
                continue
            try:
                state = int(fields[0]) - 1
                row = tuple(float(value) for value in fields[1:7])
            except ValueError:
                continue
            spin, kpoint = current
            if (
                0 <= spin < n_spins
                and 0 <= kpoint < n_kpoints
                and 0 <= state < n_bands
            ):
                values[(spin, kpoint, state)] = row
        if len(values) != expected:
            continue
        names = ("e_mf_ev", "vxc_ev", "exx_energy_ev", "exx_matrix_ev", "resigc_ev", "imsigc_ev")
        result = {
            name: np.empty((n_spins, n_kpoints, n_bands), dtype=np.float64)
            for name in names
        }
        for key, row in values.items():
            for name, value in zip(names, row):
                result[name][key] = value
        return result
    raise DiagnosticError("legacy stdout has no complete QSGW state table")


def diagonal(values: np.ndarray) -> np.ndarray:
    return np.diagonal(values, axis1=-2, axis2=-1)


def scalar_metrics(reference: np.ndarray, observed: np.ndarray) -> dict[str, object]:
    difference = observed - reference
    flat_index = int(np.argmax(np.abs(difference)))
    spin, kpoint, band = np.unravel_index(flat_index, difference.shape)
    value = difference[spin, kpoint, band]
    return {
        "max_abs_ha": float(np.max(np.abs(difference))),
        "rms_ha": float(np.sqrt(np.mean(np.abs(difference) ** 2))),
        "maximum_difference": {
            "spin": int(spin),
            "kpoint": int(kpoint),
            "band": int(band),
            "real_ha": float(value.real),
            "imag_ha": float(value.imag),
            "abs_ha": float(abs(value)),
        },
    }


def analyze(
    legacy_raw: np.ndarray,
    components: dict[str, np.ndarray],
    legacy_table: dict[str, np.ndarray],
) -> dict[str, object]:
    legacy_h = checkpoint.legacy.hermitize_legacy_upper(legacy_raw.copy())
    base = components["h0"] - components["vxc_dft"] + components["exx"]
    base_upper = checkpoint.legacy.hermitize_legacy_upper(base.copy())
    candidate_raw = components["raw_h"]
    candidate_vc = components["vc"]
    candidate_vc_upper = checkpoint.legacy.hermitize_legacy_upper(
        candidate_vc.copy()
    )
    implied_legacy_vc = legacy_h - base_upper
    assembled_upper = checkpoint.legacy.hermitize_legacy_upper(
        (base + candidate_vc).copy()
    )
    raw_closure = assembled_upper - candidate_raw
    target = legacy_h - base_upper
    denominator = float(
        np.vdot(candidate_vc_upper.ravel(), candidate_vc_upper.ravel()).real
    )
    best_scale = (
        float(np.vdot(candidate_vc_upper.ravel(), target.ravel()).real)
        / denominator
        if denominator > 0.0
        else 0.0
    )
    best_scaled = base_upper + best_scale * candidate_vc_upper

    legacy_eigenvalues, _ = checkpoint.legacy.eigensystem(legacy_raw)
    candidate_eigenvalues, _ = checkpoint.legacy.eigensystem(candidate_raw)
    legacy_stdout_eigenvalues = legacy_table["e_mf_ev"] / HA2EV
    legacy_vxc = legacy_table["vxc_ev"] / HA2EV
    legacy_exx = legacy_table["exx_matrix_ev"] / HA2EV

    raw_difference = candidate_raw - legacy_h
    return {
        "schema": SCHEMA,
        "diagnostic_complete": True,
        "candidate_raw_closure": {
            "semantics": "upper_triangle_hermitized",
            **matrix_metrics(candidate_raw, assembled_upper),
            "maximum_residual": maximum_location(raw_closure),
        },
        "legacy_checkpoint": {
            "raw_hermiticity_max_abs_ha": checkpoint.legacy.hermiticity_max(
                legacy_raw
            ),
            "stdout_eigenvalue_max_abs_ha": float(
                np.max(np.abs(legacy_eigenvalues - legacy_stdout_eigenvalues))
            ),
        },
        "legacy_diagonal_observers": {
            "vxc": scalar_metrics(legacy_vxc, diagonal(components["vxc_dft"])),
            "exx": scalar_metrics(legacy_exx, diagonal(components["exx"])),
            "legacy_reported_resigc_max_abs_ev": float(
                np.max(np.abs(legacy_table["resigc_ev"]))
            ),
            "legacy_reported_imsigc_max_abs_ev": float(
                np.max(np.abs(legacy_table["imsigc_ev"]))
            ),
        },
        "candidate_vs_legacy": {
            "raw_h": {
                **matrix_metrics(legacy_h, candidate_raw),
                "maximum_difference": maximum_location(raw_difference),
                "eigenvalue_max_abs_ha": float(
                    np.max(np.abs(candidate_eigenvalues - legacy_eigenvalues))
                ),
            },
            "base_without_vc": matrix_metrics(legacy_h, base_upper),
            "candidate_vc_vs_legacy_implied_vc": matrix_metrics(
                implied_legacy_vc, candidate_vc_upper
            ),
            "best_real_vc_scale": best_scale,
            "best_scaled_vc_hamiltonian": matrix_metrics(legacy_h, best_scaled),
        },
        "component_magnitudes": {
            name: {
                "frobenius_ha": float(np.linalg.norm(values.ravel())),
                "max_abs_ha": float(np.max(np.abs(values))),
                "maximum": maximum_location(values),
            }
            for name, values in {
                **components,
                "candidate_vc_upper": candidate_vc_upper,
                "legacy_implied_vc": implied_legacy_vc,
            }.items()
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("legacy_checkpoint_root", type=Path)
    parser.add_argument("legacy_stdout", type=Path)
    parser.add_argument("candidate_matrix_trace", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--iteration", type=int, default=1)
    parser.add_argument("--n-spins", type=int, default=1)
    parser.add_argument("--n-kpoints", type=int, default=8)
    parser.add_argument("--n-bands", type=int, default=44)
    args = parser.parse_args()
    try:
        legacy_raw = checkpoint.legacy.read_legacy_h0(
            args.legacy_checkpoint_root,
            args.iteration,
            args.n_spins,
            args.n_kpoints,
            args.n_bands,
        )
        components = parse_static_components(
            args.candidate_matrix_trace,
            args.n_spins,
            args.n_kpoints,
            args.n_bands,
        )
        legacy_table = parse_legacy_state_table_text(
            args.legacy_stdout.read_text(encoding="utf-8", errors="replace"),
            args.n_spins,
            args.n_kpoints,
            args.n_bands,
        )
        report = analyze(legacy_raw, components, legacy_table)
        report["iteration"] = args.iteration
        report["inputs"] = {
            "legacy_checkpoint_root": str(args.legacy_checkpoint_root.resolve()),
            "legacy_stdout": str(args.legacy_stdout.resolve()),
            "candidate_matrix_trace": str(args.candidate_matrix_trace.resolve()),
        }
    except (DiagnosticError, OSError, ValueError, KeyError) as error:
        report = {"schema": SCHEMA, "diagnostic_complete": False, "error": str(error)}
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))
    return 0 if report.get("diagnostic_complete") else 2


if __name__ == "__main__":
    raise SystemExit(main())
