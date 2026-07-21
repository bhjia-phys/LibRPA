#!/usr/bin/env python3
"""Independently recompute the QSGW Hartree C-V-density contraction.

The program under test dumps only the full-grid density increment and the
resulting Hartree operator.  This observer rebuilds C(k) and V(q=0) from the
frozen ABACUS producer files, contracts them in NumPy, and compares the result
with the dump.  It therefore does not use the C++ Hartree intermediate values
as inputs to the numerical recomputation.
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
        historical = (
            parent
            / "qsgw-rebase-evidence"
            / "remote"
            / "dongfang-gates-20260715-7d69a18c"
        )
        current = (
            parent
            / "qsgw-rebase-evidence"
            / "remote"
            / "fish-gate-c-current-20260721"
        )
        if historical.is_dir():
            sys.path.insert(0, str(historical))
        if current.is_dir():
            sys.path.insert(0, str(current))


_bootstrap_import_paths()

import recompute_qsgw_hartree_delta_v1 as legacy  # noqa: E402
import validate_qsgw_hartree_dump_v2 as dump_v2  # noqa: E402


SCHEMA = "librpa-qsgw-hartree-independent-contraction-v1"


class HartreeContractionValidationError(ValueError):
    pass


def _find_unique_gamma(
    kpoints: list[tuple[float, float, float]], tolerance: float
) -> int:
    matches = [
        index
        for index, point in enumerate(kpoints)
        if max(abs(value - round(value)) for value in point) <= tolerance
    ]
    if len(matches) != 1:
        raise HartreeContractionValidationError(
            f"full-k dump contains {len(matches)} periodic Gamma points, expected one"
        )
    return matches[0]


def _block_metrics(actual: dict, expected: dict) -> tuple[float, float]:
    if set(actual) != set(expected):
        mismatch = sorted(set(actual) ^ set(expected))
        raise HartreeContractionValidationError(
            f"Hartree atom-pair keys differ: {mismatch[:8]}"
        )
    max_abs = 0.0
    difference_sq = 0.0
    reference_sq = 0.0
    for key in sorted(actual):
        if actual[key].shape != expected[key].shape:
            raise HartreeContractionValidationError(
                f"Hartree block shape differs at {key}: "
                f"{actual[key].shape} != {expected[key].shape}"
            )
        difference = actual[key] - expected[key]
        if difference.size:
            max_abs = max(max_abs, float(np.max(np.abs(difference))))
        difference_sq += float(np.sum(np.abs(difference) ** 2))
        reference_sq += float(np.sum(np.abs(expected[key]) ** 2))
    return max_abs, math.sqrt(difference_sq / max(1.0, reference_sq))


def _optional_lower_atom_entries(atom_aux_sizes: dict[int, int]) -> int:
    return sum(
        atom_aux_sizes[atom_i] * atom_aux_sizes[atom_j]
        for atom_i in atom_aux_sizes
        for atom_j in atom_aux_sizes
        if atom_i > atom_j
    )


def run_checks(args: argparse.Namespace) -> dict:
    input_dir = Path(args.input_dir)
    call_dir = Path(args.dump_call)
    if not input_dir.is_dir():
        raise HartreeContractionValidationError(
            f"missing frozen input directory: {input_dir}"
        )
    if not call_dir.is_dir():
        raise HartreeContractionValidationError(
            f"missing Hartree dump call: {call_dir}"
        )

    try:
        manifest = dump_v2.load_manifest_v2(call_dir)
        normalization = manifest["normalization"]
        if normalization != args.expected_normalization:
            raise HartreeContractionValidationError(
                f"dump normalization {normalization!r} differs from expected "
                f"{args.expected_normalization!r}"
            )

        kpoint_count = manifest["kpoint_count"]
        atom_ao_sizes = manifest["atom_ao_sizes"]
        n_aos = sum(atom_ao_sizes.values())
        kpoints = dump_v2.load_full_kpoints(call_dir, kpoint_count)
        gamma_index = _find_unique_gamma(kpoints, args.gamma_tolerance)

        density = legacy.load_density_delta_k(call_dir, kpoint_count, n_aos)
        dumped_hartree = legacy.load_hartree_k(
            call_dir, atom_ao_sizes, kpoint_count
        )
        cs = legacy.read_cs_text(input_dir, prefix=args.cs_prefix)
        atom_aux_sizes = legacy.infer_atom_aux_sizes(cs, atom_ao_sizes)
        c_k = legacy.build_c_k(cs, atom_ao_sizes, atom_aux_sizes, kpoints)
        coulomb_full, zero_filled = legacy.read_coulomb_gamma_full(
            input_dir,
            gamma_q_num=gamma_index + 1,
            prefix=args.coulomb_prefix,
        )
        optional_lower = _optional_lower_atom_entries(atom_aux_sizes)
        if zero_filled > optional_lower:
            raise HartreeContractionValidationError(
                "Gamma Coulomb data omits entries outside the optional lower "
                f"atom triangle: zero_filled={zero_filled}, "
                f"optional_lower={optional_lower}"
            )
        v_q0 = legacy.build_v_q0(coulomb_full, atom_aux_sizes)
        density_blocks = legacy.split_weighted_density_by_atom(
            density, atom_ao_sizes
        )
        recomputed_hartree = legacy.contract_hartree_full_grid(
            c_k,
            v_q0,
            density_blocks,
            atom_ao_sizes,
            atom_aux_sizes,
            normalization,
        )
    except HartreeContractionValidationError:
        raise
    except Exception as error:
        raise HartreeContractionValidationError(str(error)) from error

    max_abs, relative_frobenius = _block_metrics(
        recomputed_hartree, dumped_hartree
    )
    passed = (
        max_abs <= args.matrix_max_abs_tolerance_ha
        and relative_frobenius <= args.matrix_relative_tolerance
    )
    return {
        "schema": SCHEMA,
        "passed": passed,
        "acceptance_scope": (
            "independent_CVCdagger_contraction_from_frozen_Cs_Coulomb_"
            "and_dump_density"
        ),
        "legacy_same_dataset_acceptance": False,
        "normalization": normalization,
        "full_kpoint_count": kpoint_count,
        "gamma": {
            "zero_based_index": gamma_index,
            "one_based_q_num": gamma_index + 1,
            "coordinates": list(kpoints[gamma_index]),
        },
        "basis": {
            "atom_ao_sizes": atom_ao_sizes,
            "atom_aux_sizes": atom_aux_sizes,
        },
        "input_coverage": {
            "cs_prefix": args.cs_prefix,
            "coulomb_prefix": args.coulomb_prefix,
            "coulomb_zero_filled_entries": zero_filled,
            "optional_lower_atom_triangle_entries": optional_lower,
        },
        "comparison": {
            "max_abs_ha": max_abs,
            "relative_frobenius": relative_frobenius,
            "atom_pair_count": len(recomputed_hartree),
            "passed": passed,
        },
        "tolerances": {
            "matrix_max_abs_ha": args.matrix_max_abs_tolerance_ha,
            "matrix_relative_frobenius": args.matrix_relative_tolerance,
            "periodic_gamma": args.gamma_tolerance,
        },
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--dump-call", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--expected-normalization",
        choices=legacy.NORMALIZATIONS,
        required=True,
    )
    parser.add_argument("--cs-prefix", default="Cs_data_")
    parser.add_argument("--coulomb-prefix", default="coulomb_mat_")
    parser.add_argument(
        "--matrix-max-abs-tolerance-ha", type=float, default=1.0e-8
    )
    parser.add_argument("--matrix-relative-tolerance", type=float, default=1.0e-8)
    parser.add_argument("--gamma-tolerance", type=float, default=1.0e-10)
    args = parser.parse_args(argv)
    for name in (
        "matrix_max_abs_tolerance_ha",
        "matrix_relative_tolerance",
        "gamma_tolerance",
    ):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0.0:
            parser.error(f"{name} must be finite and positive")
    if not args.cs_prefix or not args.coulomb_prefix:
        parser.error("reader prefixes must be nonempty")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output = Path(args.output)
    try:
        report = run_checks(args)
        exit_code = 0 if report["passed"] else 1
    except Exception as error:  # fail closed on malformed evidence
        report = {"schema": SCHEMA, "passed": False, "error": str(error)}
        exit_code = 1
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
