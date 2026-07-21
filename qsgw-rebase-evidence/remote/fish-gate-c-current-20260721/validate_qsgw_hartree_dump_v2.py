#!/usr/bin/env python3
"""Validate current QSGW Hartree dumps without assuming full-grid order.

The C++ observer records the exact full-k grid, real-space translations, and
atom-pair BvK remap.  This tool uses those records to validate the inverse
Fourier transform and to project the dumped periodic Hartree operator onto
every fixed-basis trace channel (SCF grid and, when enabled, band path).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np


def _bootstrap_import_paths() -> None:
    here = Path(__file__).resolve()
    for parent in here.parents:
        comparison = parent / "regression_tests" / "backend" / "comparisons"
        historical = (
            parent
            / "qsgw-rebase-evidence"
            / "remote"
            / "dongfang-gates-20260715-7d69a18c"
        )
        if comparison.is_dir():
            sys.path.insert(0, str(comparison))
        if historical.is_dir():
            sys.path.insert(0, str(historical))


_bootstrap_import_paths()

import cmp_qsgw  # noqa: E402
import recompute_qsgw_hartree_delta_v1 as legacy  # noqa: E402


SCHEMA = "librpa-qsgw-hartree-dump-v2-validation-v1"
DUMP_SCHEMA = "qsgw_hartree_pipeline_dump_v1"


class HartreeDumpValidationError(ValueError):
    pass


def _parse_int(value: str, label: str) -> int:
    try:
        return int(value)
    except ValueError as error:
        raise HartreeDumpValidationError(f"{label} is not an integer") from error


def _parse_float(value: str, label: str) -> float:
    try:
        result = float(value)
    except ValueError as error:
        raise HartreeDumpValidationError(f"{label} is not a float") from error
    if not math.isfinite(result):
        raise HartreeDumpValidationError(f"{label} is not finite")
    return result


def _read_manifest_entries(call_dir: Path) -> dict[str, str]:
    path = call_dir / "manifest.txt"
    if not path.is_file():
        raise HartreeDumpValidationError(f"missing Hartree dump manifest: {path}")
    entries: dict[str, str] = {}
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        if "=" not in line:
            raise HartreeDumpValidationError(
                f"manifest line {line_number} is not key=value"
            )
        key, value = line.split("=", 1)
        if key in entries:
            raise HartreeDumpValidationError(f"manifest repeats key {key!r}")
        entries[key] = value
    return entries


def load_manifest_v2(call_dir: Path) -> dict:
    try:
        result = dict(legacy.load_manifest(call_dir))
    except Exception as error:
        raise HartreeDumpValidationError(str(error)) from error
    entries = _read_manifest_entries(call_dir)
    expected = {
        "schema": DUMP_SCHEMA,
        "full_kpoints_file": "full_kpoints.txt",
        "full_kpoints_columns": "index kx ky kz",
        "translations_file": "translations.txt",
        "translations_columns": "index R_x R_y R_z",
        "bvk_remap_file": "bvk_remap.txt",
        "bvk_remap_columns": (
            "atom_i atom_j source_R_x source_R_y source_R_z target_index "
            "target_count target_R_x target_R_y target_R_z"
        ),
    }
    for key, value in expected.items():
        if entries.get(key) != value:
            raise HartreeDumpValidationError(
                f"manifest {key}={entries.get(key)!r}, expected {value!r}"
            )
    if "bvk_remap_source_count" not in entries:
        raise HartreeDumpValidationError("manifest misses bvk_remap_source_count")
    source_count = _parse_int(
        entries["bvk_remap_source_count"], "bvk_remap_source_count"
    )
    if source_count < 0:
        raise HartreeDumpValidationError("bvk_remap_source_count is negative")
    result["bvk_remap_source_count"] = source_count
    return result


def _read_table(path: Path, header: tuple[str, ...], label: str) -> list[list[str]]:
    if not path.is_file():
        raise HartreeDumpValidationError(f"missing {label}: {path}")
    rows: list[list[str]] = []
    saw_header = False
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            fields = tuple(stripped[1:].strip().split())
            if fields != header or saw_header:
                raise HartreeDumpValidationError(
                    f"{label}:{line_number}: unexpected or duplicate header"
                )
            saw_header = True
            continue
        fields = stripped.split()
        if len(fields) != len(header):
            raise HartreeDumpValidationError(
                f"{label}:{line_number}: expected {len(header)} columns"
            )
        rows.append(fields)
    if not saw_header:
        raise HartreeDumpValidationError(f"{label} misses its exact header")
    return rows


def load_full_kpoints(call_dir: Path, count: int) -> list[tuple[float, float, float]]:
    rows = _read_table(
        call_dir / "full_kpoints.txt", ("index", "kx", "ky", "kz"), "full_kpoints"
    )
    if len(rows) != count:
        raise HartreeDumpValidationError("full_kpoints row count differs from manifest")
    result: list[tuple[float, float, float]] = []
    for expected_index, fields in enumerate(rows):
        index = _parse_int(fields[0], "full_kpoints index")
        if index != expected_index:
            raise HartreeDumpValidationError("full_kpoints indices are not contiguous")
        result.append(
            tuple(
                _parse_float(value, "full_kpoints coordinate")
                for value in fields[1:]
            )
        )
    return result


def load_translations(
    call_dir: Path, count: int, period: tuple[int, int, int]
) -> list[tuple[int, int, int]]:
    rows = _read_table(
        call_dir / "translations.txt",
        ("index", "R_x", "R_y", "R_z"),
        "translations",
    )
    if len(rows) != count or math.prod(period) != count:
        raise HartreeDumpValidationError(
            "translation count differs from manifest or BvK period"
        )
    result: list[tuple[int, int, int]] = []
    for expected_index, fields in enumerate(rows):
        index = _parse_int(fields[0], "translations index")
        if index != expected_index:
            raise HartreeDumpValidationError("translation indices are not contiguous")
        result.append(
            tuple(_parse_int(value, "translation component") for value in fields[1:])
        )
    if len(set(result)) != len(result):
        raise HartreeDumpValidationError("translations contain duplicates")
    modulo = {
        tuple(value[axis] % period[axis] for axis in range(3)) for value in result
    }
    if len(modulo) != count:
        raise HartreeDumpValidationError(
            "translations do not cover the BvK cell modulo the period"
        )
    return result


def load_bvk_remap(
    call_dir: Path,
    source_count: int,
    atom_ao_sizes: dict[int, int],
    translations: list[tuple[int, int, int]],
) -> dict[tuple[int, int, tuple[int, int, int]], list[tuple[int, int, int]]]:
    header = (
        "atom_i", "atom_j", "source_R_x", "source_R_y", "source_R_z",
        "target_index", "target_count", "target_R_x", "target_R_y", "target_R_z",
    )
    rows = _read_table(call_dir / "bvk_remap.txt", header, "bvk_remap")
    grouped: dict[
        tuple[int, int, tuple[int, int, int]], dict[int, tuple[int, tuple[int, int, int]]]
    ] = {}
    translation_set = set(translations)
    for fields in rows:
        atom_i = _parse_int(fields[0], "bvk_remap atom_i")
        atom_j = _parse_int(fields[1], "bvk_remap atom_j")
        source = tuple(
            _parse_int(value, "bvk_remap source translation") for value in fields[2:5]
        )
        target_index = _parse_int(fields[5], "bvk_remap target_index")
        target_count = _parse_int(fields[6], "bvk_remap target_count")
        target = tuple(
            _parse_int(value, "bvk_remap target translation") for value in fields[7:10]
        )
        if atom_i not in atom_ao_sizes or atom_j not in atom_ao_sizes:
            raise HartreeDumpValidationError("bvk_remap atom index is out of range")
        if source not in translation_set:
            raise HartreeDumpValidationError("bvk_remap source is not in translations")
        if target_count <= 0 or not 0 <= target_index < target_count:
            raise HartreeDumpValidationError("bvk_remap target metadata is invalid")
        key = (atom_i, atom_j, source)
        targets = grouped.setdefault(key, {})
        if target_index in targets:
            raise HartreeDumpValidationError("bvk_remap repeats a target index")
        targets[target_index] = (target_count, target)
    if len(grouped) != source_count:
        raise HartreeDumpValidationError(
            "bvk_remap source count differs from the manifest"
        )
    result: dict[
        tuple[int, int, tuple[int, int, int]], list[tuple[int, int, int]]
    ] = {}
    for key, indexed in grouped.items():
        counts = {item[0] for item in indexed.values()}
        if len(counts) != 1:
            raise HartreeDumpValidationError("bvk_remap target_count is inconsistent")
        count = counts.pop()
        if set(indexed) != set(range(count)):
            raise HartreeDumpValidationError("bvk_remap target indices are incomplete")
        targets = [indexed[index][1] for index in range(count)]
        if len(set(targets)) != len(targets):
            raise HartreeDumpValidationError("bvk_remap repeats a target cell")
        result[key] = targets
    return result


def validate_k_r_duality(
    kpoints: list[tuple[float, float, float]],
    translations: list[tuple[int, int, int]],
    period: tuple[int, int, int],
) -> tuple[float, float]:
    k_array = np.asarray(kpoints, dtype=float)
    r_array = np.asarray(translations, dtype=float)
    reference = k_array[0]
    commensurability = 0.0
    indices: set[tuple[int, int, int]] = set()
    for kpoint in k_array:
        index = []
        for axis, size in enumerate(period):
            scaled = size * (kpoint[axis] - reference[axis])
            nearest = round(float(scaled))
            commensurability = max(commensurability, abs(float(scaled) - nearest))
            index.append(nearest % size)
        indices.add(tuple(index))
    if len(indices) != len(kpoints):
        raise HartreeDumpValidationError("full_kpoints contain periodic duplicates")
    phases = np.exp(2.0j * np.pi * (k_array @ r_array.T))
    gram = phases.conj().T @ phases / len(kpoints)
    orthogonality = float(np.max(np.abs(gram - np.eye(len(kpoints)))))
    return commensurability, orthogonality


def _max_abs(value: np.ndarray) -> float:
    return float(np.max(np.abs(value))) if value.size else 0.0


def _relative_difference(actual: np.ndarray, expected: np.ndarray) -> float:
    numerator = float(np.linalg.norm(actual - expected))
    denominator = float(np.linalg.norm(expected))
    return numerator / denominator if denominator else numerator


def _matrix(blocks: dict, key: tuple) -> np.ndarray:
    if key not in blocks:
        raise HartreeDumpValidationError(f"trace misses matrix block {key}")
    return np.asarray(blocks[key][1], dtype=np.complex128)


def _reference_spinors(
    blocks: dict, channel: int, spin: int, kpoint: int
) -> list[np.ndarray]:
    components = sorted(
        {
            key[2]
            for key in blocks
            if key[0] == 0
            and key[1] == channel
            and key[3] == spin
            and key[4] == kpoint
            and key[5] == -1
            and key[2].startswith("wfc_spinor")
        },
        key=lambda value: int(value[len("wfc_spinor"):]),
    )
    expected = [f"wfc_spinor{index}" for index in range(len(components))]
    if not components or components != expected:
        raise HartreeDumpValidationError("trace has incomplete reference spinors")
    return [
        _matrix(blocks, (0, channel, component, spin, kpoint, -1))
        for component in components
    ]


def _trace_coordinate(
    eigen_rows: dict, channel: int, spin: int, kpoint: int
) -> tuple[float, float, float]:
    coordinates = {
        item[0]
        for key, item in eigen_rows.items()
        if key[0] == 0 and key[1:4] == (channel, spin, kpoint)
    }
    if len(coordinates) != 1:
        raise HartreeDumpValidationError("trace has inconsistent k-point coordinates")
    return next(iter(coordinates))


def _assemble_full_operator_k(
    blocks: dict[tuple[int, int], np.ndarray],
    atom_ao_sizes: dict[int, int],
    kpoint: int,
) -> np.ndarray:
    offsets: dict[int, int] = {}
    total = 0
    for atom in sorted(atom_ao_sizes):
        offsets[atom] = total
        total += atom_ao_sizes[atom]
    result = np.zeros((total, total), dtype=np.complex128)
    for (atom_i, atom_j), values in blocks.items():
        block = values[kpoint]
        row = offsets[atom_i]
        column = offsets[atom_j]
        result[row:row + block.shape[0], column:column + block.shape[1]] = block
    return result


def _project(
    hartree_r: dict,
    atom_ao_sizes: dict[int, int],
    spinors: list[np.ndarray],
    kpoint: tuple[float, float, float],
) -> np.ndarray:
    operator_k = legacy.assemble_operator_k(hartree_r, atom_ao_sizes, kpoint)
    projected = np.zeros(
        (spinors[0].shape[0], spinors[0].shape[0]), dtype=np.complex128
    )
    for wavefunctions in spinors:
        if wavefunctions.shape[1] != operator_k.shape[0]:
            raise HartreeDumpValidationError(
                "reference wavefunction AO dimension differs from Hartree operator"
            )
        projected += wavefunctions.conj() @ operator_k @ wavefunctions.T
    return 0.5 * (projected + projected.conj().T)


def _call_index(call_dir: Path) -> int:
    name = call_dir.name
    if not name.startswith("call_") or not name[5:].isdigit():
        raise HartreeDumpValidationError(f"invalid dump call directory {name!r}")
    return int(name[5:])


def _validate_call(
    call_dir: Path,
    blocks: dict,
    eigen_rows: dict,
    contract: dict,
    args: argparse.Namespace,
) -> dict:
    iteration = _call_index(call_dir)
    manifest = load_manifest_v2(call_dir)
    if manifest["normalization"] != contract["hartree_normalization"]:
        raise HartreeDumpValidationError(
            "dump normalization differs from the trace contract"
        )
    atom_ao_sizes = manifest["atom_ao_sizes"]
    n_aos = sum(atom_ao_sizes.values())
    kpoint_count = manifest["kpoint_count"]
    kpoints = load_full_kpoints(call_dir, kpoint_count)
    translations = load_translations(
        call_dir, manifest["translation_count"], manifest["period"]
    )
    remap = load_bvk_remap(
        call_dir,
        manifest["bvk_remap_source_count"],
        atom_ao_sizes,
        translations,
    )
    commensurability, orthogonality = validate_k_r_duality(
        kpoints, translations, manifest["period"]
    )

    density = legacy.load_density_delta_k(call_dir, kpoint_count, n_aos)
    hartree_k = legacy.load_hartree_k(call_dir, atom_ao_sizes, kpoint_count)
    hartree_r = legacy.load_hartree_r(call_dir, atom_ao_sizes)
    recomputed_r = legacy.inverse_fourier_hartree_operator(
        hartree_k, kpoints, translations, bvk_remap=remap
    )
    fourier_max_abs, fourier_relative = legacy._block_metrics(
        recomputed_r, hartree_r
    )

    density_hermiticity = max(
        _max_abs(matrix - matrix.conj().T) for matrix in density
    )
    density_charge = complex(sum(np.trace(matrix) for matrix in density))
    density_max_abs = _max_abs(density)
    hartree_k_hermiticity = 0.0
    hartree_k_max_abs = 0.0
    for kpoint in range(kpoint_count):
        matrix = _assemble_full_operator_k(hartree_k, atom_ao_sizes, kpoint)
        hartree_k_hermiticity = max(
            hartree_k_hermiticity, _max_abs(matrix - matrix.conj().T)
        )
        hartree_k_max_abs = max(hartree_k_max_abs, _max_abs(matrix))

    expected_channels = {0}
    if contract["band"] == "fixed_reference_operator_fourier_live":
        expected_channels.add(1)
    observed_layout = {
        (key[1], key[3], key[4])
        for key in blocks
        if key[0] == iteration and key[2] == "delta_vh" and key[5] == -1
    }
    observed_channels = {item[0] for item in observed_layout}
    if observed_channels != expected_channels:
        raise HartreeDumpValidationError(
            "delta_vh trace channels differ from the QSGW contract"
        )

    projection_max_abs = 0.0
    projection_relative = 0.0
    projection_hermiticity = 0.0
    delta_vh_max_abs = 0.0
    projection_blocks = 0
    for channel, spin, kpoint in sorted(observed_layout):
        expected = _matrix(
            blocks, (iteration, channel, "delta_vh", spin, kpoint, -1)
        )
        spinors = _reference_spinors(blocks, channel, spin, kpoint)
        coordinate = _trace_coordinate(eigen_rows, channel, spin, kpoint)
        actual = _project(hartree_r, atom_ao_sizes, spinors, coordinate)
        if actual.shape != expected.shape:
            raise HartreeDumpValidationError("projected delta_vh shape differs")
        projection_max_abs = max(
            projection_max_abs, _max_abs(actual - expected)
        )
        projection_relative = max(
            projection_relative, _relative_difference(actual, expected)
        )
        projection_hermiticity = max(
            projection_hermiticity,
            _max_abs(actual - actual.conj().T),
            _max_abs(expected - expected.conj().T),
        )
        delta_vh_max_abs = max(delta_vh_max_abs, _max_abs(expected))
        projection_blocks += 1

    duality_passed = (
        commensurability <= args.grid_tolerance
        and orthogonality <= args.grid_tolerance
    )
    fourier_passed = (
        fourier_max_abs <= args.matrix_max_abs_tolerance_ha
        and fourier_relative <= args.matrix_relative_tolerance
    )
    invariants_passed = (
        density_hermiticity <= args.hermiticity_tolerance
        and abs(density_charge) <= args.charge_tolerance
        and hartree_k_hermiticity <= args.hermiticity_tolerance
    )
    projection_passed = (
        projection_max_abs <= args.matrix_max_abs_tolerance_ha
        and projection_relative <= args.matrix_relative_tolerance
        and projection_hermiticity <= args.hermiticity_tolerance
    )
    response_passed = True
    if iteration == 1:
        response_passed = max(
            density_max_abs, hartree_k_max_abs, delta_vh_max_abs
        ) <= args.zero_tolerance
    elif args.require_nonzero_after > 0 and iteration >= args.require_nonzero_after:
        response_passed = min(
            density_max_abs, hartree_k_max_abs, delta_vh_max_abs
        ) >= args.nonzero_tolerance

    return {
        "iteration": iteration,
        "passed": (
            duality_passed
            and fourier_passed
            and invariants_passed
            and projection_passed
            and response_passed
        ),
        "grid": {
            "commensurability_max": commensurability,
            "fourier_orthogonality_max": orthogonality,
            "passed": duality_passed,
        },
        "bvk_remap": {
            "source_count": len(remap),
            "target_count": sum(len(targets) for targets in remap.values()),
        },
        "inverse_fourier": {
            "max_abs_ha": fourier_max_abs,
            "relative_frobenius": fourier_relative,
            "passed": fourier_passed,
        },
        "density": {
            "max_abs": density_max_abs,
            "hermiticity_max": density_hermiticity,
            "total_charge_real": density_charge.real,
            "total_charge_imag": density_charge.imag,
        },
        "hartree_k": {
            "max_abs_ha": hartree_k_max_abs,
            "hermiticity_max": hartree_k_hermiticity,
        },
        "fixed_basis_projection": {
            "block_count": projection_blocks,
            "channels": sorted(observed_channels),
            "delta_vh_max_abs_ha": delta_vh_max_abs,
            "max_abs_ha": projection_max_abs,
            "relative_frobenius": projection_relative,
            "hermiticity_max": projection_hermiticity,
            "passed": projection_passed,
        },
        "invariants_passed": invariants_passed,
        "response_passed": response_passed,
    }


def run_checks(args: argparse.Namespace) -> dict:
    dump_root = Path(args.dump_root)
    matrix_path = Path(args.matrix_trace)
    eigenvalue_path = Path(args.eigenvalue_trace)
    matrix_text = matrix_path.read_text(encoding="utf-8")
    eigenvalue_text = eigenvalue_path.read_text(encoding="utf-8")
    contract = cmp_qsgw._require_same_contract(
        matrix_text, eigenvalue_text, "Hartree matrix/eigenvalue trace"
    )
    if contract["qsgw_contract_version"] != 6:
        raise HartreeDumpValidationError("Hartree observer requires contract v6")
    if contract["hartree"] != "delta_density":
        raise HartreeDumpValidationError("trace does not enable Hartree delta_density")
    if contract["headwing"] != "disabled_stage1":
        raise HartreeDumpValidationError("Gate C requires head-wing disabled")
    if args.expected_symmetry and contract["symmetry"] != args.expected_symmetry:
        raise HartreeDumpValidationError("trace symmetry contract differs")

    blocks = cmp_qsgw._parse_matrix_trace(matrix_text, "Hartree matrix trace")
    eigen_rows = cmp_qsgw._parse_eigenvalue_trace(
        eigenvalue_text, "Hartree eigenvalue trace"
    )
    cmp_qsgw._validate_matrix_trajectory(blocks, contract, "Hartree matrix trace")
    cmp_qsgw._validate_eigenvalue_trajectory(
        eigen_rows, contract, "Hartree eigenvalue trace"
    )
    last_iteration = max(key[0] for key in blocks)
    if last_iteration != args.expected_iterations:
        raise HartreeDumpValidationError(
            f"trace ends at iteration {last_iteration}, expected {args.expected_iterations}"
        )

    call_dirs = sorted(path for path in dump_root.glob("call_*") if path.is_dir())
    expected_names = [
        f"call_{index:03d}" for index in range(1, args.expected_iterations + 1)
    ]
    if [path.name for path in call_dirs] != expected_names:
        raise HartreeDumpValidationError(
            "Hartree dump calls are not exactly the completed iterations"
        )
    calls = [
        _validate_call(path, blocks, eigen_rows, contract, args)
        for path in call_dirs
    ]
    return {
        "schema": SCHEMA,
        "passed": all(call["passed"] for call in calls),
        "acceptance_scope": (
            "current_dump_fourier_remap_density_and_fixed_basis_projection"
        ),
        "legacy_same_dataset_acceptance": False,
        "contract": {
            "version": contract["qsgw_contract_version"],
            "symmetry": contract["symmetry"],
            "hartree_coulomb": contract["hartree_coulomb"],
            "hartree_normalization": contract["hartree_normalization"],
            "band": contract["band"],
        },
        "calls": calls,
        "tolerances": {
            "matrix_max_abs_ha": args.matrix_max_abs_tolerance_ha,
            "matrix_relative_frobenius": args.matrix_relative_tolerance,
            "hermiticity": args.hermiticity_tolerance,
            "charge": args.charge_tolerance,
            "grid": args.grid_tolerance,
            "iteration1_zero": args.zero_tolerance,
            "nonzero": args.nonzero_tolerance,
        },
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("dump_root")
    parser.add_argument("matrix_trace")
    parser.add_argument("eigenvalue_trace")
    parser.add_argument("output")
    parser.add_argument("--expected-iterations", type=int, default=2)
    parser.add_argument(
        "--expected-symmetry", default="exx_on_gw_on_rpa_on"
    )
    parser.add_argument("--matrix-max-abs-tolerance-ha", type=float, default=1.0e-8)
    parser.add_argument("--matrix-relative-tolerance", type=float, default=1.0e-8)
    parser.add_argument("--hermiticity-tolerance", type=float, default=1.0e-10)
    parser.add_argument("--charge-tolerance", type=float, default=1.0e-10)
    parser.add_argument("--grid-tolerance", type=float, default=1.0e-10)
    parser.add_argument("--zero-tolerance", type=float, default=1.0e-10)
    parser.add_argument("--nonzero-tolerance", type=float, default=1.0e-12)
    parser.add_argument("--require-nonzero-after", type=int, default=2)
    args = parser.parse_args(argv)
    positive = (
        "matrix_max_abs_tolerance_ha",
        "matrix_relative_tolerance",
        "hermiticity_tolerance",
        "charge_tolerance",
        "grid_tolerance",
        "zero_tolerance",
        "nonzero_tolerance",
    )
    if args.expected_iterations < 1 or args.require_nonzero_after < 0:
        parser.error("iteration controls are invalid")
    if any(
        not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0.0
        for name in positive
    ):
        parser.error("all tolerances must be finite and positive")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output = Path(args.output)
    try:
        report = run_checks(args)
    except Exception as error:  # fail closed on malformed evidence
        report = {"schema": SCHEMA, "passed": False, "error": str(error)}
        exit_code = 1
    else:
        exit_code = 0 if report["passed"] else 2
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
