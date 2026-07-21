#!/usr/bin/env python3
"""Detect legacy Hartree reader side effects while delta-Vh is zero."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path


def _bootstrap_import_paths() -> None:
    here = Path(__file__).resolve()
    for parent in here.parents:
        historical = (
            parent
            / "qsgw-rebase-evidence"
            / "remote"
            / "dongfang-gates-20260715-7d69a18c"
        )
        if historical.is_dir():
            sys.path.insert(0, str(historical))


_bootstrap_import_paths()

import compare_qsgw_component_traces_v3 as base  # noqa: E402


SCHEMA = "librpa-qsgw-legacy-hartree-null-delta-v1"


class LegacyHartreeNullDeltaError(ValueError):
    pass


def _validate_contracts(enabled_text: str, disabled_text: str) -> dict:
    enabled = base.parse_contract(enabled_text, "legacy Hartree-on trace")
    disabled = base.parse_contract(disabled_text, "legacy Hartree-off trace")
    required = {
        "qsgw_contract_version": "4",
        "task": "qsgw",
        "fixed_basis": "immutable_reference",
        "qsgw_mixer": "linear",
        "qsgw_mixing_beta": "0.2",
        "qsgw_min_iter": "1",
        "qsgw_max_iter": "1",
        "use_symmetry_gw": "0",
        "use_symmetry_exx": "0",
        "replace_w_head": "0",
        "option_dielect_func": "0",
        "use_shrink_abfs": "0",
        "qsgw_hartree_coulomb": "truncated",
        "qsgw_hartree_normalization": "legacy_extra_inverse_nk",
    }
    for label, contract, hartree in (
        ("Hartree-on", enabled, "1"),
        ("Hartree-off", disabled, "0"),
    ):
        expected = dict(required)
        expected["qsgw_update_hartree"] = hartree
        differences = {
            key: {"expected": value, "actual": contract.get(key)}
            for key, value in expected.items()
            if contract.get(key) != value
        }
        if differences:
            raise LegacyHartreeNullDeltaError(
                f"legacy {label} contract mismatch: {differences}"
            )
    allowed = {"qsgw_update_hartree"}
    differences = {
        key: {"enabled": enabled.get(key), "disabled": disabled.get(key)}
        for key in sorted(set(enabled) | set(disabled))
        if key not in allowed and enabled.get(key) != disabled.get(key)
    }
    if differences:
        raise LegacyHartreeNullDeltaError(
            f"legacy on/off contracts differ outside Hartree enablement: {differences}"
        )
    return enabled


def _parse_summary(path: Path, label: str) -> dict[int, tuple[float, float, float]]:
    if not path.is_file():
        raise LegacyHartreeNullDeltaError(f"missing {label}: {path}")
    result: dict[int, tuple[float, float, float]] = {}
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        fields = stripped.split()
        if len(fields) != 4:
            raise LegacyHartreeNullDeltaError(
                f"{label}:{line_number}: expected iteration HOMO LUMO Fermi"
            )
        iteration = int(fields[0])
        values = tuple(float(value) for value in fields[1:])
        if iteration in result or any(not math.isfinite(value) for value in values):
            raise LegacyHartreeNullDeltaError(
                f"{label}:{line_number}: duplicate or non-finite row"
            )
        result[iteration] = values
    if set(result) != {0, 1}:
        raise LegacyHartreeNullDeltaError(
            f"{label}: expected exactly iterations 0 and 1"
        )
    return result


def _component_metrics(
    enabled_rows: dict, disabled_rows: dict, frequency_tolerance: float
) -> dict[str, dict]:
    if set(enabled_rows) != set(disabled_rows):
        missing = sorted(set(disabled_rows) - set(enabled_rows))[:5]
        extra = sorted(set(enabled_rows) - set(disabled_rows))[:5]
        raise LegacyHartreeNullDeltaError(
            f"legacy on/off non-Hartree row layouts differ: missing={missing}, extra={extra}"
        )
    accumulators: dict[str, dict[str, float | int]] = {}
    for key in sorted(enabled_rows):
        enabled_frequency, enabled_value = enabled_rows[key]
        disabled_frequency, disabled_value = disabled_rows[key]
        if abs(enabled_frequency - disabled_frequency) > frequency_tolerance:
            raise LegacyHartreeNullDeltaError(
                f"legacy on/off frequency differs at {key}"
            )
        component = key[2]
        entry = accumulators.setdefault(
            component,
            {"max_abs_ha": 0.0, "difference_sq": 0.0, "reference_sq": 0.0, "rows": 0},
        )
        difference = abs(enabled_value - disabled_value)
        entry["max_abs_ha"] = max(float(entry["max_abs_ha"]), difference)
        entry["difference_sq"] = float(entry["difference_sq"]) + difference**2
        entry["reference_sq"] = float(entry["reference_sq"]) + abs(disabled_value) ** 2
        entry["rows"] = int(entry["rows"]) + 1
    result: dict[str, dict] = {}
    for component, entry in sorted(accumulators.items()):
        result[component] = {
            "max_abs_ha": entry["max_abs_ha"],
            "relative_frobenius": math.sqrt(
                float(entry["difference_sq"])
                / max(1.0, float(entry["reference_sq"]))
            ),
            "row_count": entry["rows"],
        }
    return result


def run_checks(args: argparse.Namespace) -> dict:
    enabled_text = Path(args.enabled_matrix).read_text(encoding="utf-8")
    disabled_text = Path(args.disabled_matrix).read_text(encoding="utf-8")
    contract = _validate_contracts(enabled_text, disabled_text)
    selected = {0, 1}
    enabled_rows = base.parse_rows(
        enabled_text, "legacy Hartree-on trace", selected, 0
    )
    disabled_rows = base.parse_rows(
        disabled_text, "legacy Hartree-off trace", selected, 0
    )
    base.validate_iteration_coverage(
        enabled_rows, "legacy Hartree-on trace", selected, contract
    )
    disabled_contract = base.parse_contract(
        disabled_text, "legacy Hartree-off trace"
    )
    base.validate_iteration_coverage(
        disabled_rows, "legacy Hartree-off trace", selected, disabled_contract
    )

    delta_rows = {
        key: value for key, value in enabled_rows.items() if key[2] == "delta_vh"
    }
    if not delta_rows or {key[0] for key in delta_rows} != {1}:
        raise LegacyHartreeNullDeltaError(
            "legacy Hartree-on trace must contain iteration-1 delta_vh"
        )
    delta_max = max(abs(value) for _frequency, value in delta_rows.values())
    comparable_enabled = {
        key: value for key, value in enabled_rows.items() if key[2] != "delta_vh"
    }
    comparable_disabled = {
        key: value for key, value in disabled_rows.items() if key[2] != "delta_vh"
    }
    components = _component_metrics(
        comparable_enabled, comparable_disabled, args.frequency_tolerance_ha
    )
    component_passed = all(
        metrics["max_abs_ha"] <= args.matrix_max_abs_tolerance_ha
        and metrics["relative_frobenius"] <= args.matrix_relative_tolerance
        for metrics in components.values()
    )

    enabled_summary = _parse_summary(
        Path(args.enabled_summary), "legacy Hartree-on summary"
    )
    disabled_summary = _parse_summary(
        Path(args.disabled_summary), "legacy Hartree-off summary"
    )
    summary_max = max(
        abs(enabled_summary[iteration][column] - disabled_summary[iteration][column])
        for iteration in (0, 1)
        for column in range(3)
    )
    passed = (
        delta_max <= args.zero_tolerance_ha
        and component_passed
        and summary_max <= args.summary_energy_tolerance_ev
    )
    return {
        "schema": SCHEMA,
        "passed": passed,
        "acceptance_scope": "legacy_isolated_reader_iteration1_null_delta_side_effect_guard",
        "delta_vh": {
            "max_abs_ha": delta_max,
            "row_count": len(delta_rows),
            "passed": delta_max <= args.zero_tolerance_ha,
        },
        "components": {
            component: {
                **metrics,
                "passed": (
                    metrics["max_abs_ha"] <= args.matrix_max_abs_tolerance_ha
                    and metrics["relative_frobenius"]
                    <= args.matrix_relative_tolerance
                ),
            }
            for component, metrics in components.items()
        },
        "summary": {
            "max_abs_ev": summary_max,
            "passed": summary_max <= args.summary_energy_tolerance_ev,
        },
        "tolerances": {
            "zero_delta_ha": args.zero_tolerance_ha,
            "matrix_max_abs_ha": args.matrix_max_abs_tolerance_ha,
            "matrix_relative_frobenius": args.matrix_relative_tolerance,
            "frequency_ha": args.frequency_tolerance_ha,
            "summary_energy_ev": args.summary_energy_tolerance_ev,
        },
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("enabled_matrix")
    parser.add_argument("disabled_matrix")
    parser.add_argument("enabled_summary")
    parser.add_argument("disabled_summary")
    parser.add_argument("output")
    parser.add_argument("--zero-tolerance-ha", type=float, default=1.0e-10)
    parser.add_argument(
        "--matrix-max-abs-tolerance-ha", type=float, default=1.0e-8
    )
    parser.add_argument("--matrix-relative-tolerance", type=float, default=1.0e-8)
    parser.add_argument("--frequency-tolerance-ha", type=float, default=1.0e-10)
    parser.add_argument("--summary-energy-tolerance-ev", type=float, default=1.0e-5)
    args = parser.parse_args(argv)
    for name in (
        "zero_tolerance_ha",
        "matrix_max_abs_tolerance_ha",
        "matrix_relative_tolerance",
        "frequency_tolerance_ha",
        "summary_energy_tolerance_ev",
    ):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0.0:
            parser.error(f"{name} must be finite and positive")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output = Path(args.output)
    try:
        report = run_checks(args)
        exit_code = 0 if report["passed"] else 1
    except Exception as error:
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
