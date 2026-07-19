#!/usr/bin/env python3
"""Validate a controlled QSGW disabled/head-only trace pair."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import numpy as np

from compare_qsgw_component_traces_v3 import _matrix_groups, parse_rows


SHA256 = re.compile(r"[0-9a-f]{64}")
COMMIT = re.compile(r"[0-9a-f]{40}")
COMMON_HEADERS = {
    "qsgw_contract_version": "5",
    "fixed_basis": "immutable_mf0",
    "live_update": "eigenvalues_wfc",
    "symmetry": "unsupported_full_bz_only",
    "hartree": "disabled_stage1",
    "band": "disabled_stage1",
    "qsgw_mixer": "none",
}


def _header(text: str, label: str) -> dict[str, str]:
    accepted = set(COMMON_HEADERS) | {
        "velocity",
        "headwing",
        "qsgw_input_contract",
        "qsgw_input_contract_sha256",
        "qsgw_mixing_beta",
    }
    values: dict[str, str] = {}
    for line in text.splitlines():
        if not line.startswith("# "):
            continue
        fields = line[2:].split(maxsplit=1)
        if len(fields) == 2 and fields[0] in accepted:
            if fields[0] in values:
                raise ValueError(f"duplicate {fields[0]} header in {label}")
            values[fields[0]] = fields[1]
    missing = sorted(accepted - set(values))
    if missing:
        raise ValueError(f"missing headers in {label}: {missing}")
    for key, expected in COMMON_HEADERS.items():
        if values[key] != expected:
            raise ValueError(
                f"unexpected {key}={values[key]!r} in {label}"
            )
    if not SHA256.fullmatch(values["qsgw_input_contract_sha256"]):
        raise ValueError(f"invalid input-contract SHA256 in {label}")
    return values


def _validate_pair(pair: dict[str, object]) -> tuple[dict, dict, dict]:
    if pair.get("schema") != "qsgw-head-mode-controlled-pair-v1":
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

    mode(side_a, "disabled")
    mode(side_b, "head_only")

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

    if set(variants) != {"disabled", "head_only"}:
        raise ValueError("head-mode variants are incomplete")
    required_variant = {
        "replace_w_head",
        "option_dielect_func",
        "qsgw_input_contract",
        "qsgw_input_contract_sha256",
        "librpa_input_sha256",
    }
    disabled = variants["disabled"]
    head_only = variants["head_only"]
    if not isinstance(disabled, dict) or not isinstance(head_only, dict):
        raise ValueError("head-mode variants must be objects")
    if set(disabled) != required_variant or set(head_only) != required_variant:
        raise ValueError("head-mode variant expansion is incomplete")
    expected = {
        "disabled": (False, 0, "qsgw_input.disabled.contract"),
        "head_only": (True, 4, "qsgw_input.head-only.contract"),
    }
    for name, variant in (("disabled", disabled), ("head_only", head_only)):
        replace, option, contract = expected[name]
        if (
            variant["replace_w_head"] is not replace
            or variant["option_dielect_func"] != option
            or variant["qsgw_input_contract"] != contract
        ):
            raise ValueError(f"invalid {name} head-mode expansion")
        for key in ("qsgw_input_contract_sha256", "librpa_input_sha256"):
            if not SHA256.fullmatch(str(variant[key])):
                raise ValueError(f"invalid {name} {key}")
    return common, disabled, head_only


def _selected(
    matrices: dict[tuple[int, str, int, int, int], np.ndarray],
    *,
    iteration: int,
    components: tuple[str, ...],
    prefix: bool = False,
) -> dict[tuple[str, int, int, int], np.ndarray]:
    result = {}
    for (row_iteration, component, spin, kpoint, frequency), matrix in matrices.items():
        matches = (
            any(component.startswith(item) for item in components)
            if prefix
            else component in components
        )
        if row_iteration == iteration and matches:
            result[(component, spin, kpoint, frequency)] = matrix
    return result


def _difference(
    left: dict[tuple[str, int, int, int], np.ndarray],
    right: dict[tuple[str, int, int, int], np.ndarray],
    label: str,
) -> tuple[float, float, int]:
    if not left or set(left) != set(right):
        raise ValueError(f"{label} matrix layout differs or is empty")
    maximum = 0.0
    difference_squared = 0.0
    scale_squared = 0.0
    for key in sorted(left):
        if left[key].shape != right[key].shape:
            raise ValueError(f"{label} matrix shape differs for {key}")
        delta = left[key] - right[key]
        maximum = max(maximum, float(np.max(np.abs(delta))))
        difference_squared += float(np.linalg.norm(delta)) ** 2
        scale_squared += max(
            float(np.linalg.norm(left[key])),
            float(np.linalg.norm(right[key])),
        ) ** 2
    relative = math.sqrt(difference_squared) / max(
        math.sqrt(scale_squared), 1.0e-300
    )
    return maximum, relative, len(left)


def compare(
    disabled_text: str,
    head_only_text: str,
    pair: dict[str, object],
    *,
    initial_tolerance: float,
    exx_tolerance: float,
    head_effect_minimum: float,
) -> dict[str, object]:
    for label, value in (
        ("initial tolerance", initial_tolerance),
        ("EXX tolerance", exx_tolerance),
        ("head-effect minimum", head_effect_minimum),
    ):
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"invalid {label}")
    _common, disabled_variant, head_variant = _validate_pair(pair)
    disabled_header = _header(disabled_text, "disabled trace")
    head_header = _header(head_only_text, "head-only trace")
    if disabled_header["velocity"] != "disabled_stage1" or disabled_header["headwing"] != "disabled_stage1":
        raise ValueError("side A is not a disabled-head trace")
    if head_header["velocity"] != "fixed_basis_rotation" or head_header["headwing"] != "scf_grid_analytic_live":
        raise ValueError("side B is not a same-grid head-only trace")
    for key in set(COMMON_HEADERS) | {"qsgw_mixing_beta"}:
        if disabled_header[key] != head_header[key]:
            raise ValueError(f"shared trace header {key} differs")
    if disabled_header["qsgw_input_contract_sha256"] != disabled_variant["qsgw_input_contract_sha256"]:
        raise ValueError("disabled trace contract does not match the pair record")
    if head_header["qsgw_input_contract_sha256"] != head_variant["qsgw_input_contract_sha256"]:
        raise ValueError("head-only trace contract does not match the pair record")
    if Path(disabled_header["qsgw_input_contract"]).name != disabled_variant["qsgw_input_contract"]:
        raise ValueError("disabled trace contract path does not match the pair record")
    if Path(head_header["qsgw_input_contract"]).name != head_variant["qsgw_input_contract"]:
        raise ValueError("head-only trace contract path does not match the pair record")

    disabled_rows = parse_rows(disabled_text, "disabled trace", {0, 1}, 0)
    head_rows = parse_rows(head_only_text, "head-only trace", {0, 1}, 0)
    disabled_matrices = _matrix_groups(disabled_rows)
    head_matrices = _matrix_groups(head_rows)

    initial_left = _selected(
        disabled_matrices,
        iteration=0,
        components=("h0", "wfc_spinor"),
        prefix=True,
    )
    initial_right = _selected(
        head_matrices,
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
        _selected(disabled_matrices, iteration=1, components=("exx",)),
        _selected(head_matrices, iteration=1, components=("exx",)),
        "iteration-1 EXX",
    )
    sigc_max, sigc_relative, sigc_blocks = _difference(
        _selected(
            disabled_matrices,
            iteration=1,
            components=("sigma_c_iw",),
        ),
        _selected(
            head_matrices,
            iteration=1,
            components=("sigma_c_iw",),
        ),
        "iteration-1 Sigma_c",
    )
    passed = bool(
        initial_max <= initial_tolerance
        and exx_max <= exx_tolerance
        and sigc_max >= head_effect_minimum
    )
    return {
        "passed": passed,
        "changed_factor": "parameters.head_mode",
        "factor_role": "head_mode",
        "side_a": "disabled",
        "side_b": "head_only",
        "initial_block_count": initial_blocks,
        "initial_max_abs": initial_max,
        "initial_relative_frobenius": initial_relative,
        "initial_tolerance": initial_tolerance,
        "iteration1_exx_block_count": exx_blocks,
        "iteration1_exx_max_abs": exx_max,
        "iteration1_exx_relative_frobenius": exx_relative,
        "exx_tolerance": exx_tolerance,
        "iteration1_sigc_block_count": sigc_blocks,
        "iteration1_sigc_max_abs_change": sigc_max,
        "iteration1_sigc_relative_frobenius_change": sigc_relative,
        "head_effect_minimum": head_effect_minimum,
        "disabled_contract_sha256": disabled_header[
            "qsgw_input_contract_sha256"
        ],
        "head_only_contract_sha256": head_header[
            "qsgw_input_contract_sha256"
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("disabled_trace", type=Path)
    parser.add_argument("head_only_trace", type=Path)
    parser.add_argument("pair_contract", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--initial-tolerance", type=float, default=1.0e-10)
    parser.add_argument("--exx-tolerance", type=float, default=1.0e-10)
    parser.add_argument("--head-effect-minimum", type=float, default=1.0e-12)
    args = parser.parse_args()
    try:
        report = compare(
            args.disabled_trace.read_text(encoding="utf-8"),
            args.head_only_trace.read_text(encoding="utf-8"),
            json.loads(args.pair_contract.read_text(encoding="utf-8")),
            initial_tolerance=args.initial_tolerance,
            exx_tolerance=args.exx_tolerance,
            head_effect_minimum=args.head_effect_minimum,
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
