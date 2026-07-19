#!/usr/bin/env python3
"""Validate a controlled QSGW head-only/head-plus-wing trace pair."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import numpy as np

from compare_qsgw_component_traces_v3 import _matrix_groups, parse_rows
from compare_qsgw_head_mode_pair import _difference, _header, _selected


SHA256 = re.compile(r"[0-9a-f]{64}")
COMMIT = re.compile(r"[0-9a-f]{40}")
GRAM_VALUE = re.compile(
    r"\(\s*([-+0-9.Ee]+)\s*,\s*([-+0-9.Ee]+)\s*\)"
)


def _validate_pair(pair: dict[str, object]) -> tuple[dict, dict, dict]:
    if pair.get("schema") != "qsgw-wing-mode-controlled-pair-v1":
        raise ValueError("unsupported controlled-pair schema")
    if pair.get("changed_factor") != "parameters.head_mode":
        raise ValueError("changed_factor must be parameters.head_mode")
    if pair.get("factor_role") != "head_mode":
        raise ValueError("factor_role must be head_mode")

    common = pair.get("common")
    side_a = pair.get("side_a")
    side_b = pair.get("side_b")
    variants = pair.get("variants")
    if not all(isinstance(item, dict) for item in (common, side_a, side_b, variants)):
        raise ValueError("controlled-pair records must be objects")
    assert isinstance(common, dict)
    assert isinstance(side_a, dict)
    assert isinstance(side_b, dict)
    assert isinstance(variants, dict)

    def mode(side: dict, expected: str) -> None:
        if set(side) != {"parameters"} or not isinstance(side["parameters"], dict):
            raise ValueError("pair sides may contain only head_mode parameters")
        parameters = side["parameters"]
        if set(parameters) != {"head_mode"}:
            raise ValueError("pair sides may contain only head_mode")
        if parameters["head_mode"] != expected:
            raise ValueError(f"expected head_mode {expected}")

    mode(side_a, "head_only")
    mode(side_b, "head_plus_wing")

    required_common = {
        "source_commit",
        "executable_sha256",
        "dataset_manifest_sha256",
        "mpi_ranks",
        "omp_threads",
        "deterministic_reduction",
    }
    if set(common) != required_common:
        raise ValueError("controlled-pair common provenance is incomplete")
    if not COMMIT.fullmatch(str(common["source_commit"])):
        raise ValueError("invalid common source commit")
    for key in ("executable_sha256", "dataset_manifest_sha256"):
        if not SHA256.fullmatch(str(common[key])):
            raise ValueError(f"invalid common {key}")
    if not isinstance(common["mpi_ranks"], int) or common["mpi_ranks"] <= 0:
        raise ValueError("invalid common MPI rank count")
    if not isinstance(common["omp_threads"], int) or common["omp_threads"] <= 0:
        raise ValueError("invalid common OMP thread count")
    if common["deterministic_reduction"] is not True:
        raise ValueError("deterministic reduction must be enabled")

    if set(variants) != {"head_only", "head_plus_wing"}:
        raise ValueError("wing-mode variants are incomplete")
    required_variant = {
        "replace_w_head",
        "option_dielect_func",
        "qsgw_input_contract",
        "qsgw_input_contract_sha256",
        "librpa_input_sha256",
    }
    head_only = variants["head_only"]
    head_plus_wing = variants["head_plus_wing"]
    if not isinstance(head_only, dict) or not isinstance(head_plus_wing, dict):
        raise ValueError("wing-mode variants must be objects")
    expected = {"head_only": 4, "head_plus_wing": 3}
    for name, variant in (
        ("head_only", head_only),
        ("head_plus_wing", head_plus_wing),
    ):
        if set(variant) != required_variant:
            raise ValueError("wing-mode variant expansion is incomplete")
        if (
            variant["replace_w_head"] is not True
            or variant["option_dielect_func"] != expected[name]
            or variant["qsgw_input_contract"]
            != "qsgw_input.head-only.contract"
        ):
            raise ValueError(f"invalid {name} wing-mode expansion")
        for key in ("qsgw_input_contract_sha256", "librpa_input_sha256"):
            if not SHA256.fullmatch(str(variant[key])):
                raise ValueError(f"invalid {name} {key}")
    if (
        head_only["qsgw_input_contract_sha256"]
        != head_plus_wing["qsgw_input_contract_sha256"]
    ):
        raise ValueError("wing-mode variants must share one input contract")
    return common, head_only, head_plus_wing


def _stdout_mode(text: str, label: str, expected_option: int) -> int:
    replace_values = re.findall(
        r"(?m)^replace_w_head\s*=\s*(true|false)\s*$", text
    )
    option_values = re.findall(
        r"(?m)^option_dielect_func\s*=\s*([0-9]+)\s*$", text
    )
    if replace_values != ["true"]:
        raise ValueError(f"{label} replace_w_head is not uniquely true")
    if option_values != [str(expected_option)]:
        raise ValueError(f"{label} option_dielect_func does not match the pair")
    return text.count("* Success: calculate wing term.")


def _wing_gram(text: str) -> tuple[np.ndarray, float, float]:
    marker = "Wing_mu Gram (iomega=0, rows alpha, columns beta):"
    lines = text.splitlines()
    indices = [index for index, line in enumerate(lines) if line.strip() == marker]
    if len(indices) != 1:
        raise ValueError("head-plus-wing stdout must contain one Wing_mu Gram")
    start = indices[0] + 1
    if start + 3 > len(lines):
        raise ValueError("Wing_mu Gram is truncated")
    gram = np.empty((3, 3), dtype=np.complex128)
    for row in range(3):
        values = GRAM_VALUE.findall(lines[start + row])
        if len(values) != 3:
            raise ValueError("Wing_mu Gram row has an invalid layout")
        for column, (real, imag) in enumerate(values):
            gram[row, column] = complex(float(real), float(imag))
    if not np.all(np.isfinite(gram)):
        raise ValueError("Wing_mu Gram contains non-finite values")
    hermiticity = float(np.max(np.abs(gram - gram.conj().T)))
    hermitian = 0.5 * (gram + gram.conj().T)
    minimum_eigenvalue = float(np.min(np.linalg.eigvalsh(hermitian)))
    return gram, hermiticity, minimum_eigenvalue


def compare(
    head_only_text: str,
    head_plus_wing_text: str,
    head_only_stdout: str,
    head_plus_wing_stdout: str,
    pair: dict[str, object],
    *,
    initial_tolerance: float,
    invariant_tolerance: float,
    wing_effect_minimum: float,
) -> dict[str, object]:
    for label, value in (
        ("initial tolerance", initial_tolerance),
        ("invariant tolerance", invariant_tolerance),
        ("wing-effect minimum", wing_effect_minimum),
    ):
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"invalid {label}")

    _common, head_variant, wing_variant = _validate_pair(pair)
    head_header = _header(head_only_text, "head-only trace")
    wing_header = _header(head_plus_wing_text, "head-plus-wing trace")
    for label, header in (
        ("head-only", head_header),
        ("head-plus-wing", wing_header),
    ):
        if (
            header["velocity"] != "fixed_basis_rotation"
            or header["headwing"] != "scf_grid_analytic_live"
        ):
            raise ValueError(f"{label} trace is not same-grid live head/wing")
    if head_header != wing_header:
        raise ValueError("head-only and head-plus-wing trace headers differ")
    if Path(head_header["qsgw_input_contract"]).name != head_variant[
        "qsgw_input_contract"
    ]:
        raise ValueError("trace contract path does not match the pair record")
    if (
        head_header["qsgw_input_contract_sha256"]
        != head_variant["qsgw_input_contract_sha256"]
        or head_header["qsgw_input_contract_sha256"]
        != wing_variant["qsgw_input_contract_sha256"]
    ):
        raise ValueError("trace contract SHA256 does not match the pair record")

    head_success_count = _stdout_mode(head_only_stdout, "head-only", 4)
    wing_success_count = _stdout_mode(
        head_plus_wing_stdout, "head-plus-wing", 3
    )
    if head_success_count != 0:
        raise ValueError("head-only stdout unexpectedly calculated a wing")
    if "Wing_mu diagnostics (iomega=0):" not in head_plus_wing_stdout:
        raise ValueError("head-plus-wing stdout lacks Wing_mu diagnostics")
    _gram, gram_hermiticity, gram_minimum = _wing_gram(
        head_plus_wing_stdout
    )

    head_rows = parse_rows(head_only_text, "head-only trace", {0, 1}, 0)
    wing_rows = parse_rows(
        head_plus_wing_text, "head-plus-wing trace", {0, 1}, 0
    )
    head_matrices = _matrix_groups(head_rows)
    wing_matrices = _matrix_groups(wing_rows)

    initial_left = _selected(
        head_matrices,
        iteration=0,
        components=("h0", "wfc_spinor"),
        prefix=True,
    )
    initial_right = _selected(
        wing_matrices,
        iteration=0,
        components=("h0", "wfc_spinor"),
        prefix=True,
    )
    initial_max, initial_relative, initial_blocks = _difference(
        initial_left, initial_right, "initial state"
    )
    if not any(key[0] == "h0" for key in initial_left) or not any(
        key[0].startswith("wfc_spinor") for key in initial_left
    ):
        raise ValueError("initial-state observer requires h0 and WFC blocks")

    exx_max, exx_relative, exx_blocks = _difference(
        _selected(head_matrices, iteration=1, components=("exx",)),
        _selected(wing_matrices, iteration=1, components=("exx",)),
        "iteration-1 EXX",
    )
    head_max, head_relative, head_blocks = _difference(
        _selected(head_matrices, iteration=1, components=("head_tensor",)),
        _selected(wing_matrices, iteration=1, components=("head_tensor",)),
        "iteration-1 head tensor",
    )
    sigc_max, sigc_relative, sigc_blocks = _difference(
        _selected(head_matrices, iteration=1, components=("sigma_c_iw",)),
        _selected(wing_matrices, iteration=1, components=("sigma_c_iw",)),
        "iteration-1 Sigma_c",
    )
    passed = bool(
        initial_max <= initial_tolerance
        and exx_max <= invariant_tolerance
        and head_max <= invariant_tolerance
        and gram_hermiticity <= invariant_tolerance
        and gram_minimum >= -invariant_tolerance
        and wing_success_count >= 3
        and sigc_max >= wing_effect_minimum
    )
    return {
        "passed": passed,
        "changed_factor": "parameters.head_mode",
        "factor_role": "head_mode",
        "side_a": "head_only",
        "side_b": "head_plus_wing",
        "initial_block_count": initial_blocks,
        "initial_max_abs": initial_max,
        "initial_relative_frobenius": initial_relative,
        "initial_tolerance": initial_tolerance,
        "iteration1_exx_block_count": exx_blocks,
        "iteration1_exx_max_abs": exx_max,
        "iteration1_exx_relative_frobenius": exx_relative,
        "head_tensor_block_count": head_blocks,
        "head_tensor_max_abs": head_max,
        "head_tensor_relative_frobenius": head_relative,
        "iteration1_sigc_block_count": sigc_blocks,
        "iteration1_sigc_max_abs_change": sigc_max,
        "iteration1_sigc_relative_frobenius_change": sigc_relative,
        "invariant_tolerance": invariant_tolerance,
        "wing_effect_minimum": wing_effect_minimum,
        "head_success_count": head_success_count,
        "wing_success_count": wing_success_count,
        "wing_gram_hermiticity": gram_hermiticity,
        "wing_gram_min_eigenvalue": gram_minimum,
        "input_contract_sha256": head_header[
            "qsgw_input_contract_sha256"
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("head_only_trace", type=Path)
    parser.add_argument("head_plus_wing_trace", type=Path)
    parser.add_argument("head_only_stdout", type=Path)
    parser.add_argument("head_plus_wing_stdout", type=Path)
    parser.add_argument("pair_contract", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--initial-tolerance", type=float, default=1.0e-10)
    parser.add_argument("--invariant-tolerance", type=float, default=1.0e-10)
    parser.add_argument("--wing-effect-minimum", type=float, default=1.0e-12)
    args = parser.parse_args()
    try:
        report = compare(
            args.head_only_trace.read_text(encoding="utf-8"),
            args.head_plus_wing_trace.read_text(encoding="utf-8"),
            args.head_only_stdout.read_text(encoding="utf-8"),
            args.head_plus_wing_stdout.read_text(encoding="utf-8"),
            json.loads(args.pair_contract.read_text(encoding="utf-8")),
            initial_tolerance=args.initial_tolerance,
            invariant_tolerance=args.invariant_tolerance,
            wing_effect_minimum=args.wing_effect_minimum,
        )
        status = 0 if report["passed"] else 2
    except (OSError, ValueError, json.JSONDecodeError) as error:
        report = {"passed": False, "error": str(error)}
        status = 2
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="ascii",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return status


if __name__ == "__main__":
    raise SystemExit(main())
