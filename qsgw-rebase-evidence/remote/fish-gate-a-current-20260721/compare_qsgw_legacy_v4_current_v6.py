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

INTERNAL_HA2EV = 27.211386245988


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
    expected_legacy_symmetry: str,
    expected_current_symmetry: str,
    expected_legacy_head: str,
    expected_current_head: str,
    allow_legacy_iteration_prefix: bool = False,
    expected_legacy_use_shrink_abfs: bool = True,
) -> dict[str, object]:
    symmetry_switch = {"off": "0", "on": "1"}
    current_symmetry_contract = {
        "off": "exx_off_gw_off_rpa_off",
        "on": "exx_on_gw_on_rpa_on",
    }
    if expected_legacy_symmetry not in symmetry_switch:
        raise ValueError("expected legacy symmetry must be 'off' or 'on'")
    if expected_current_symmetry not in current_symmetry_contract:
        raise ValueError("expected current symmetry must be 'off' or 'on'")
    head_switch = {
        "off": ("0", "0"),
        "on": ("1", "4"),
    }
    current_head_contract = {
        "off": "disabled_stage1",
        "on": "scf_grid_analytic_live",
    }
    current_velocity_contract = {
        "off": "disabled_stage1",
        "on": "fixed_basis_rotation",
    }
    if expected_legacy_head not in head_switch:
        raise ValueError("expected legacy head must be 'off' or 'on'")
    if expected_current_head not in current_head_contract:
        raise ValueError("expected current head must be 'off' or 'on'")

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
            "use_symmetry_gw": symmetry_switch[expected_legacy_symmetry],
            "use_symmetry_exx": symmetry_switch[expected_legacy_symmetry],
            "replace_w_head": head_switch[expected_legacy_head][0],
            "option_dielect_func": head_switch[expected_legacy_head][1],
            "nfreq": "6",
            "use_shrink_abfs": (
                "1" if expected_legacy_use_shrink_abfs else "0"
            ),
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
            "velocity": current_velocity_contract[expected_current_head],
            "headwing": current_head_contract[expected_current_head],
            "symmetry": current_symmetry_contract[expected_current_symmetry],
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
        "legacy_symmetry": expected_legacy_symmetry,
        "current_symmetry": expected_current_symmetry,
        "symmetry_mapping": (
            f"legacy_{expected_legacy_symmetry}_to_"
            f"current_{expected_current_symmetry}"
        ),
        "legacy_head": expected_legacy_head,
        "current_head": expected_current_head,
        "head_mapping": (
            f"legacy_{expected_legacy_head}_to_"
            f"current_{expected_current_head}"
        ),
        "wing": "off",
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
        "legacy_use_shrink_abfs": expected_legacy_use_shrink_abfs,
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


def _normalize_current_v6_to_v5(
    text: str,
    *,
    headwing: str,
    velocity: str,
    symmetry: str,
    mixer: str | None = None,
) -> str:
    replacements = {
        "qsgw_contract_version": "5",
        "velocity": velocity,
        "symmetry": symmetry,
    }
    if mixer is not None:
        replacements["qsgw_mixer"] = mixer
    rewritten = _rewrite_headers(text, replacements)
    output: list[str] = []
    head_seen = False
    wing_seen = False
    for line in rewritten.splitlines():
        match = re.match(r"^(#\s*)(\S+)(?:\s+)(.*)$", line)
        if match and match.group(2) == "head":
            output.append(f"# headwing {headwing}")
            head_seen = True
        elif match and match.group(2) == "wing":
            wing_seen = True
        else:
            output.append(line)
    if not head_seen or not wing_seen:
        raise ValueError("cannot normalize v6 trace without split head/wing headers")
    return "\n".join(output) + ("\n" if text.endswith("\n") else "")


def _rescale_trace_ev_columns(
    text: str,
    *,
    source_ha2ev: float,
    columns: tuple[int, ...],
) -> str:
    if not math.isfinite(source_ha2ev) or source_ha2ev <= 0.0:
        raise ValueError("current_ha2ev must be finite and positive")
    factor = INTERNAL_HA2EV / source_ha2ev
    output: list[str] = []
    for line_number, line in enumerate(text.splitlines(), 1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            output.append(line)
            continue
        fields = stripped.split()
        try:
            for column in columns:
                fields[column] = f"{float(fields[column]) * factor:.17e}"
        except (IndexError, ValueError) as error:
            raise ValueError(
                f"cannot rescale trace eV columns at line {line_number}"
            ) from error
        output.append(" ".join(fields))
    return "\n".join(output) + ("\n" if text.endswith("\n") else "")


def _normalized_for_frozen_aligner(
    legacy_text: str,
    current_texts: tuple[str, str, str],
    current_ha2ev: float = INTERNAL_HA2EV,
) -> tuple[str, tuple[str, str, str]]:
    normalized_legacy = _rewrite_headers(
        legacy_text,
        {
            "qsgw_mixing_beta": "1",
            "use_symmetry_gw": "0",
            "use_symmetry_exx": "0",
            "replace_w_head": "0",
            "option_dielect_func": "0",
            "n_params_anacon": "6",
            "use_shrink_abfs": "0",
        },
    )
    normalized_current_headers = tuple(
        _normalize_current_v6_to_v5(
            text,
            headwing="disabled_stage1",
            velocity="disabled_stage1",
            symmetry="unsupported_full_bz_only",
            mixer="none",
        )
        for text in current_texts
    )
    normalized_current = (
        normalized_current_headers[0],
        _rescale_trace_ev_columns(
            normalized_current_headers[1],
            source_ha2ev=current_ha2ev,
            columns=(8,),
        ),
        _rescale_trace_ev_columns(
            normalized_current_headers[2],
            source_ha2ev=current_ha2ev,
            columns=(1, 4, 5),
        ),
    )
    return normalized_legacy, normalized_current


def _normalized_for_v5_self_validators(text: str) -> str:
    contract_head = re.search(r"^#\s+head\s+(\S+)\s*$", text, re.MULTILINE)
    contract_velocity = re.search(
        r"^#\s+velocity\s+(\S+)\s*$", text, re.MULTILINE
    )
    if contract_head is None or contract_velocity is None:
        raise ValueError("cannot normalize v6 self-validator trace headers")
    return _normalize_current_v6_to_v5(
        text,
        headwing=contract_head.group(1),
        velocity=contract_velocity.group(1),
        symmetry="input_kstar_live",
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
        expected_legacy_symmetry=args.expected_legacy_symmetry,
        expected_current_symmetry=args.expected_current_symmetry,
        expected_legacy_head=args.expected_legacy_head,
        expected_current_head=args.expected_current_head,
        allow_legacy_iteration_prefix=args.allow_legacy_iteration_prefix,
        expected_legacy_use_shrink_abfs=bool(
            args.expected_legacy_use_shrink_abfs
        ),
    )
    normalized_legacy, normalized_current = _normalized_for_frozen_aligner(
        legacy_text,
        (current_matrix_text, current_eigenvalue_text, current_iteration_text),
        args.current_ha2ev,
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
        normalized_self_traces = (
            normalized_self_traces[0],
            _rescale_trace_ev_columns(
                normalized_self_traces[1],
                source_ha2ev=args.current_ha2ev,
                columns=(8,),
            ),
            _rescale_trace_ev_columns(
                normalized_self_traces[2],
                source_ha2ev=args.current_ha2ev,
                columns=(1, 4, 5),
            ),
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
            "actual_symmetry": {
            "legacy": args.expected_legacy_symmetry,
            "current": args.expected_current_symmetry,
        },
        "actual_head": {
            "legacy": args.expected_legacy_head,
            "current": args.expected_current_head,
        },
            "legacy": {
                "qsgw_mixing_beta": "1",
                "use_symmetry_gw": "0",
                "use_symmetry_exx": "0",
                "replace_w_head": "0",
                "option_dielect_func": "0",
                "n_params_anacon": "6",
                "use_shrink_abfs": "0",
            },
            "current": {
                "qsgw_contract_version": "5",
                "symmetry": "unsupported_full_bz_only",
                "velocity": "disabled_stage1",
                "head": "headwing disabled_stage1",
                "wing": "removed",
                "qsgw_mixer": "none",
            },
            "reason": "reuse frozen v4-v5 numerical row alignment after independent v4-v6 contract validation",
        },
        "unit_normalization": {
            "scope": "current_eV_trace_columns_only",
            "source_ha2ev": args.current_ha2ev,
            "target_ha2ev": INTERNAL_HA2EV,
            "factor": INTERNAL_HA2EV / args.current_ha2ev,
            "matrix_rows_changed": False,
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
    parser.add_argument(
        "--expected-legacy-symmetry",
        choices=("off", "on"),
        required=True,
    )
    parser.add_argument(
        "--expected-current-symmetry",
        choices=("off", "on"),
        required=True,
    )
    parser.add_argument(
        "--expected-legacy-head",
        choices=("off", "on"),
        default="off",
    )
    parser.add_argument(
        "--expected-current-head",
        choices=("off", "on"),
        default="off",
    )
    parser.add_argument(
        "--expected-legacy-use-shrink-abfs",
        choices=(0, 1),
        type=int,
        default=1,
    )
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
    parser.add_argument(
        "--current-ha2ev",
        type=float,
        default=INTERNAL_HA2EV,
    )
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
