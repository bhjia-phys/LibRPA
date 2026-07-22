#!/usr/bin/env python3
"""Compare a frozen legacy-v4 QSGW trace with a current-v6 trace.

The historical numerical aligner understands legacy v4 versus current v5.
This adapter first validates the actual v4/v6 contracts, then rewrites only
contract header lines so the frozen aligner can compare the unchanged numeric
rows. The normalization is explicit in the JSON report.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import re
from pathlib import Path
from types import ModuleType


def _load_module(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot load Python module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _iterations(spec: str) -> list[int]:
    if ":" in spec:
        start_text, stop_text = spec.split(":", 1)
        start = int(start_text)
        stop = int(stop_text)
        values = list(range(start, stop + 1))
    else:
        values = sorted({int(value) for value in spec.split(",")})
    if len(values) < 2 or values != list(range(values[-1] + 1)):
        raise ValueError("iterations must be continuous from zero")
    return values


def _equal(actual: object, expected: object) -> bool:
    if isinstance(expected, float):
        try:
            return math.isclose(
                float(actual), expected, rel_tol=0.0, abs_tol=1.0e-15
            )
        except (TypeError, ValueError):
            return False
    return actual == expected


def _require_values(
    values: dict[str, object], expected: dict[str, object], label: str
) -> None:
    differences = {
        key: {"expected": required, "actual": values.get(key)}
        for key, required in expected.items()
        if not _equal(values.get(key), required)
    }
    if differences:
        raise ValueError(f"{label}: contract mismatch {differences}")


def _validate_actual_contracts(
    *,
    base_module: ModuleType,
    current_module: ModuleType,
    legacy_text: str,
    current_texts: tuple[tuple[str, str], ...],
    final_iteration: int,
    expected_mode: str,
    expected_legacy_beta: float,
    expected_current_beta: float,
    allow_legacy_iteration_prefix: bool = False,
) -> dict[str, object]:
    legacy = base_module.parse_contract(legacy_text, "legacy v4 trace")
    declared_min_iteration = int(legacy["qsgw_min_iter"])
    declared_max_iteration = int(legacy["qsgw_max_iter"])
    if declared_min_iteration != declared_max_iteration:
        raise ValueError(
            "legacy v4 trace: qsgw_min_iter and qsgw_max_iter differ"
        )
    if allow_legacy_iteration_prefix:
        if declared_max_iteration < final_iteration:
            raise ValueError(
                "legacy v4 trace: declared iteration bound is shorter than "
                "the selected prefix"
            )
    elif declared_max_iteration != final_iteration:
        raise ValueError(
            "legacy v4 trace: declared iteration bound differs from the "
            "selected final iteration"
        )

    declared_n_params = int(legacy["n_params_anacon"])
    effective_n_params = (
        int(legacy["nfreq"])
        if declared_n_params == -1
        else declared_n_params
    )
    if effective_n_params != int(legacy["nfreq"]):
        raise ValueError(
            "legacy v4 trace: analytic continuation does not use all "
            "available frequency points"
        )
    _require_values(
        legacy,
        {
            "qsgw_contract_version": "4",
            "task": "qsgw",
            "fixed_basis": "immutable_reference",
            "qsgw_mixer": "linear",
            "qsgw_mixing_beta": expected_legacy_beta,
            "starting_vxc": "dft_only",
            "vxc_basis": "fixed_state",
            "qsgw_update_hartree": "0",
            "use_symmetry_gw": "1",
            "use_symmetry_exx": "1",
            "replace_w_head": "0",
            "option_dielect_func": "0",
            "nfreq": "6",
            "use_shrink_abfs": "1",
            "use_fullcoul_exx": "0",
            "use_fullcoul_eps": "1",
            "use_fullcoul_wc": "0",
            "constants_choice": "internal",
            "ac_policy": "direct_pade",
        },
        "legacy v4 trace",
    )

    parsed_current = [
        (label, current_module._parse_contract(text, label))
        for label, text in current_texts
    ]
    reference_label, reference = parsed_current[0]
    for label, values in parsed_current[1:]:
        if values != reference:
            raise ValueError(
                f"{label}: v6 contract differs from {reference_label}"
            )
    _require_values(
        reference,
        {
            "qsgw_contract_version": 6,
            "fixed_basis": "immutable_mf0",
            "live_update": "eigenvalues_wfc",
            "velocity": "disabled_stage1",
            "headwing": "disabled_stage1",
            "symmetry": "exx_on_gw_on_rpa_on",
            "hartree": "disabled_stage1",
            "band": "disabled_stage1",
            "h_qsgw_cut": "disabled_non_band",
            "qsgw_mixer": expected_mode,
            "qsgw_mixing_beta": expected_current_beta,
        },
        "current v6 trace",
    )
    return {
        "passed": True,
        "legacy_version": 4,
        "current_version": 6,
        "legacy_role": "compatibility_harness_not_raw_source",
        "fixed_basis_mapping": "immutable_reference_to_immutable_mf0",
        "symmetry": "legacy_exx_on_gw_on_to_current_exx_on_gw_on_rpa_on",
        "headwing": "off",
        "hartree": "off",
        "band": "off",
        "mixer": expected_mode,
        "legacy_effective_beta": expected_legacy_beta,
        "current_configured_beta": expected_current_beta,
        "final_iteration": final_iteration,
        "legacy_declared_final_iteration": declared_max_iteration,
        "legacy_iteration_selection": (
            "prefix"
            if declared_max_iteration > final_iteration
            else "exact"
        ),
        "legacy_declared_n_params_anacon": declared_n_params,
        "legacy_effective_n_params_anacon": effective_n_params,
        "current_input_contract": reference["qsgw_input_contract"],
        "current_input_contract_sha256": reference[
            "qsgw_input_contract_sha256"
        ],
    }


def _rewrite_headers(text: str, replacements: dict[str, str]) -> str:
    remaining = set(replacements)
    output: list[str] = []
    for line in text.splitlines():
        match = re.match(r"^(#\s*)(\S+)(?:\s+)(.*)$", line)
        if match and match.group(2) in replacements:
            key = match.group(2)
            output.append(f"# {key} {replacements[key]}")
            remaining.discard(key)
        else:
            output.append(line)
    if remaining:
        raise ValueError(f"cannot normalize missing headers {sorted(remaining)}")
    return "\n".join(output) + ("\n" if text.endswith("\n") else "")


def _normalized_for_frozen_aligner(
    legacy_text: str, current_texts: tuple[str, str, str]
) -> tuple[str, tuple[str, str, str]]:
    normalized_legacy = _rewrite_headers(
        legacy_text,
        {
            "qsgw_mixing_beta": "1",
            "use_symmetry_gw": "0",
            "use_symmetry_exx": "0",
            "n_params_anacon": "6",
            "use_shrink_abfs": "0",
        },
    )
    normalized_current = tuple(
        _rewrite_headers(
            text,
            {
                "qsgw_contract_version": "5",
                "symmetry": "unsupported_full_bz_only",
                "qsgw_mixer": "none",
            },
        )
        for text in current_texts
    )
    return normalized_legacy, normalized_current


def _normalized_for_v5_self_validators(text: str) -> str:
    return _rewrite_headers(
        text,
        {
            "qsgw_contract_version": "5",
            "symmetry": "input_kstar_live",
        },
    )


def compare(args: argparse.Namespace) -> dict[str, object]:
    base_module = _load_module(args.base_comparator, "qsgw_legacy_aligner")
    current_module = _load_module(
        args.current_contract_parser, "qsgw_current_contract_parser"
    )
    legacy_text = args.legacy_matrix.read_text(encoding="utf-8")
    current_matrix_text = args.current_matrix.read_text(encoding="utf-8")
    current_eigenvalue_text = args.current_eigenvalues.read_text(
        encoding="utf-8"
    )
    current_iteration_text = args.current_iterations.read_text(encoding="utf-8")
    iterations = _iterations(args.iterations)
    contract = _validate_actual_contracts(
        base_module=base_module,
        current_module=current_module,
        legacy_text=legacy_text,
        current_texts=(
            ("current matrix trace", current_matrix_text),
            ("current eigenvalue trace", current_eigenvalue_text),
            ("current iteration trace", current_iteration_text),
        ),
        final_iteration=iterations[-1],
        expected_mode=args.expected_mode,
        expected_legacy_beta=args.expected_legacy_beta,
        expected_current_beta=args.expected_current_beta,
        allow_legacy_iteration_prefix=args.allow_legacy_iteration_prefix,
    )
    normalized_legacy, normalized_current = _normalized_for_frozen_aligner(
        legacy_text,
        (current_matrix_text, current_eigenvalue_text, current_iteration_text),
    )
    numeric = base_module.compare_legacy_v4_current_v5(
        old_matrix_text=normalized_legacy,
        current_matrix_text=normalized_current[0],
        current_eigenvalue_text=normalized_current[1],
        current_iteration_text=normalized_current[2],
        iterations=iterations,
        channel=0,
        frequency_tolerance_ha=args.frequency_tolerance,
        matrix_max_abs_tolerance_ha=args.matrix_max_abs_tolerance_ha,
        matrix_relative_tolerance=args.matrix_relative_tolerance,
        eigenvalue_tolerance_ha=args.eigenvalue_tolerance_ha,
        gap_tolerance_ev=args.gap_tolerance_ev,
        degeneracy_tolerance_ha=args.degeneracy_tolerance_ha,
        state_tolerance=args.state_tolerance,
        expected_legacy_use_fullcoul_exx=False,
        allow_iteration_prefix=args.allow_legacy_iteration_prefix,
    )
    if "contract" in numeric:
        numeric["normalized_contract_check"] = numeric.pop("contract")
    normalized_output_paths = (
        args.normalized_current_matrix,
        args.normalized_current_eigenvalues,
        args.normalized_current_iterations,
    )
    if any(path is not None for path in normalized_output_paths):
        if not all(path is not None for path in normalized_output_paths):
            raise ValueError("all normalized current output paths are required together")
        normalized_self_traces = tuple(
            _normalized_for_v5_self_validators(text)
            for text in (
                current_matrix_text,
                current_eigenvalue_text,
                current_iteration_text,
            )
        )
        for path, text in zip(normalized_output_paths, normalized_self_traces):
            assert path is not None
            path.write_text(text, encoding="utf-8")
        normalized_outputs = {
            "scope": "contract_headers_only_numeric_rows_unchanged",
            "qsgw_contract_version": "5",
            "symmetry": "input_kstar_live",
            "matrix": str(args.normalized_current_matrix.resolve()),
            "matrix_sha256": _sha256(args.normalized_current_matrix),
            "eigenvalues": str(args.normalized_current_eigenvalues.resolve()),
            "eigenvalues_sha256": _sha256(args.normalized_current_eigenvalues),
            "iterations": str(args.normalized_current_iterations.resolve()),
            "iterations_sha256": _sha256(args.normalized_current_iterations),
        }
    else:
        normalized_outputs = None
    return {
        "passed": bool(contract["passed"] and numeric["passed"]),
        "contract_mode": "legacy_v4_to_current_v6",
        "actual_contract": contract,
        "numeric_alignment": numeric,
        "header_normalization": {
            "scope": "contract_headers_only_numeric_rows_unchanged",
            "legacy": {
                "qsgw_mixing_beta": "1",
                "use_symmetry_gw": "0",
                "use_symmetry_exx": "0",
                "n_params_anacon": "6",
                "use_shrink_abfs": "0",
            },
            "current": {
                "qsgw_contract_version": "5",
                "symmetry": "unsupported_full_bz_only",
                "qsgw_mixer": "none",
            },
            "reason": "reuse frozen v4-v5 numerical row alignment after independent v4-v6 contract validation",
        },
        "v5_self_validator_inputs": normalized_outputs,
        "tool_provenance": {
            "base_comparator": str(args.base_comparator.resolve()),
            "base_comparator_sha256": _sha256(args.base_comparator),
            "current_contract_parser": str(
                args.current_contract_parser.resolve()
            ),
            "current_contract_parser_sha256": _sha256(
                args.current_contract_parser
            ),
        },
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("legacy_matrix", type=Path)
    parser.add_argument("current_matrix", type=Path)
    parser.add_argument("current_eigenvalues", type=Path)
    parser.add_argument("current_iterations", type=Path)
    parser.add_argument("output_json", type=Path)
    parser.add_argument("--base-comparator", required=True, type=Path)
    parser.add_argument("--current-contract-parser", required=True, type=Path)
    parser.add_argument("--normalized-current-matrix", type=Path)
    parser.add_argument("--normalized-current-eigenvalues", type=Path)
    parser.add_argument("--normalized-current-iterations", type=Path)
    parser.add_argument("--iterations", required=True)
    parser.add_argument("--expected-mode", choices=("none", "linear"), required=True)
    parser.add_argument("--expected-legacy-beta", type=float, required=True)
    parser.add_argument("--expected-current-beta", type=float, required=True)
    parser.add_argument("--allow-legacy-iteration-prefix", action="store_true")
    parser.add_argument("--frequency-tolerance", type=float, default=1.0e-10)
    parser.add_argument(
        "--matrix-max-abs-tolerance-ha", type=float, default=1.0e-8
    )
    parser.add_argument(
        "--matrix-relative-tolerance", type=float, default=1.0e-8
    )
    parser.add_argument("--eigenvalue-tolerance-ha", type=float, default=1.0e-6)
    parser.add_argument("--gap-tolerance-ev", type=float, default=1.0e-5)
    parser.add_argument(
        "--degeneracy-tolerance-ha", type=float, default=1.0e-8
    )
    parser.add_argument("--state-tolerance", type=float, default=1.0e-10)
    return parser


def main() -> None:
    args = _parser().parse_args()
    try:
        report = compare(args)
    except Exception as error:
        report = {"passed": False, "error": str(error)}
    report["legacy_matrix"] = str(args.legacy_matrix.resolve())
    report["current_matrix"] = str(args.current_matrix.resolve())
    report["current_eigenvalues"] = str(args.current_eigenvalues.resolve())
    report["current_iterations"] = str(args.current_iterations.resolve())
    args.output_json.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    raise SystemExit(0 if report["passed"] else 2)


if __name__ == "__main__":
    main()
