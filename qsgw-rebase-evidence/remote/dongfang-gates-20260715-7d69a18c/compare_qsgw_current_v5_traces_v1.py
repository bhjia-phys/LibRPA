#!/usr/bin/env python3
"""Compare two current (contract v5) QSGW traces for run-to-run consistency.

Both inputs must be QSGW component traces carrying the contract-version-5
header. The tool aligns data rows on
(iter, channel, component, spin, kpoint, frequency_index, row, column),
checks the run-defining contract keys, per-row frequencies, and per-component
value metrics (max abs diff and relative Frobenius), and reports JSON.

Exit codes: 0 = passed, 2 = comparison failed, 1 = malformed input
(fail-closed: wrong contract version, bad columns, duplicate keys,
non-finite values, unreadable files, invalid options).
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


DEFAULT_FREQUENCY_TOLERANCE_HA = 1.0e-10
DEFAULT_MATRIX_MAX_ABS_TOLERANCE_HA = 1.0e-12
DEFAULT_MATRIX_RELATIVE_TOLERANCE = 1.0e-8

CONTRACT_VERSION_KEY = "qsgw_contract_version"
REQUIRED_CONTRACT_VERSION = 5
CONTRACT_COMPARE_KEYS = (
    "hartree",
    "qsgw_mixer",
    "qsgw_mixing_beta",
    "qsgw_input_contract_sha256",
)

RowKey = tuple[int, int, str, int, int, int, int, int]
RowValue = tuple[float, complex]


class TraceFormatError(ValueError):
    """Raised when a trace or option violates the v5 format contract."""


def parse_iterations(spec: str) -> list[int]:
    if ":" in spec:
        start_text, stop_text = spec.split(":", 1)
        start = int(start_text)
        stop = int(stop_text)
        if start < 0 or stop < start:
            raise TraceFormatError("invalid iteration range")
        return list(range(start, stop + 1))
    values = sorted({int(value) for value in spec.split(",")})
    if not values or values[0] < 0:
        raise TraceFormatError("invalid iteration list")
    return values


def parse_contract_header(text: str, label: str) -> dict[str, str]:
    header: dict[str, str] = {}
    for line_number, raw_line in enumerate(text.splitlines(), 1):
        line = raw_line.strip()
        if not line.startswith("#"):
            continue
        fields = line[1:].strip().split(None, 1)
        if len(fields) != 2:
            continue
        key, value = fields
        if key in header:
            raise TraceFormatError(
                f"{label}:{line_number}: duplicate contract key {key}"
            )
        header[key] = value.strip()
    version = header.get(CONTRACT_VERSION_KEY)
    if version is None:
        raise TraceFormatError(f"{label}: missing {CONTRACT_VERSION_KEY}")
    try:
        version_int = int(version)
    except ValueError as error:
        raise TraceFormatError(
            f"{label}: invalid {CONTRACT_VERSION_KEY} {version!r}"
        ) from error
    if version_int != REQUIRED_CONTRACT_VERSION:
        raise TraceFormatError(
            f"{label}: unsupported {CONTRACT_VERSION_KEY} {version_int}"
        )
    return header


def parse_rows(
    text: str,
    label: str,
    iterations: set[int] | None,
    channel: int,
) -> dict[RowKey, RowValue]:
    rows: dict[RowKey, RowValue] = {}
    for line_number, raw_line in enumerate(text.splitlines(), 1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) != 11:
            raise TraceFormatError(
                f"{label}:{line_number}: expected 11 columns"
            )
        try:
            iteration = int(fields[0])
            row_channel = int(fields[1])
            spin = int(fields[3])
            kpoint = int(fields[4])
            frequency_index = int(fields[5])
            frequency = float(fields[6])
            row = int(fields[7])
            column = int(fields[8])
            real = float(fields[9])
            imag = float(fields[10])
        except ValueError as error:
            raise TraceFormatError(
                f"{label}:{line_number}: unparsable trace row"
            ) from error
        if iterations is not None and iteration not in iterations:
            continue
        if row_channel != channel:
            continue
        key: RowKey = (
            iteration,
            row_channel,
            fields[2],
            spin,
            kpoint,
            frequency_index,
            row,
            column,
        )
        if key in rows:
            raise TraceFormatError(
                f"{label}:{line_number}: duplicate row key {key}"
            )
        if not (
            math.isfinite(frequency)
            and math.isfinite(real)
            and math.isfinite(imag)
        ):
            raise TraceFormatError(
                f"{label}:{line_number}: non-finite trace value"
            )
        rows[key] = (frequency, complex(real, imag))
    if not rows:
        raise TraceFormatError(f"{label}: no selected trace rows")
    return rows


def compare_contract_keys(
    old_header: dict[str, str], new_header: dict[str, str]
) -> list[dict[str, object]]:
    differences: list[dict[str, object]] = []
    for key in CONTRACT_COMPARE_KEYS:
        old_value = old_header.get(key)
        new_value = new_header.get(key)
        if old_value is None or new_value is None:
            if old_value != new_value:
                differences.append(
                    {"key": key, "old": old_value, "new": new_value}
                )
            continue
        if key == "qsgw_mixing_beta":
            try:
                equal = math.isclose(
                    float(old_value),
                    float(new_value),
                    rel_tol=0.0,
                    abs_tol=1.0e-15,
                )
            except ValueError:
                equal = old_value == new_value
        else:
            equal = old_value == new_value
        if not equal:
            differences.append({"key": key, "old": old_value, "new": new_value})
    return differences


def _component_metrics(
    old_rows: dict[RowKey, RowValue],
    new_rows: dict[RowKey, RowValue],
    common_keys: set[RowKey],
    max_abs_tolerance_ha: float,
    relative_tolerance: float,
) -> dict[str, dict[str, float | int | bool]]:
    accumulators: dict[str, dict[str, object]] = {}
    for key in sorted(common_keys):
        component = key[2]
        _old_frequency, old_value = old_rows[key]
        _new_frequency, new_value = new_rows[key]
        difference = abs(old_value - new_value)
        entry = accumulators.setdefault(
            component,
            {
                "blocks": set(),
                "max_abs_diff_ha": 0.0,
                "sum_diff2": 0.0,
                "sum_old2": 0.0,
                "sum_new2": 0.0,
            },
        )
        blocks = entry["blocks"]
        assert isinstance(blocks, set)
        blocks.add((key[0], key[1], key[3], key[4], key[5]))
        entry["max_abs_diff_ha"] = max(
            float(entry["max_abs_diff_ha"]), difference
        )
        entry["sum_diff2"] = float(entry["sum_diff2"]) + difference * difference
        entry["sum_old2"] = float(entry["sum_old2"]) + abs(old_value) ** 2
        entry["sum_new2"] = float(entry["sum_new2"]) + abs(new_value) ** 2

    metrics: dict[str, dict[str, float | int | bool]] = {}
    for component, entry in accumulators.items():
        maximum = float(entry["max_abs_diff_ha"])
        scale = max(
            math.sqrt(float(entry["sum_old2"])),
            math.sqrt(float(entry["sum_new2"])),
            1.0e-30,
        )
        relative = math.sqrt(float(entry["sum_diff2"])) / scale
        metrics[component] = {
            "max_abs_diff_ha": maximum,
            "max_relative_frobenius": relative,
            "block_count": len(entry["blocks"]),
            "passed": bool(
                maximum <= max_abs_tolerance_ha
                and relative <= relative_tolerance
            ),
        }
    return metrics


def compare_trace_text(
    old_text: str,
    new_text: str,
    iterations: list[int] | None = None,
    channel: int = 0,
    frequency_tolerance_ha: float = DEFAULT_FREQUENCY_TOLERANCE_HA,
    matrix_max_abs_tolerance_ha: float = DEFAULT_MATRIX_MAX_ABS_TOLERANCE_HA,
    matrix_relative_tolerance: float = DEFAULT_MATRIX_RELATIVE_TOLERANCE,
    exclude_components: frozenset[str] | None = None,
) -> dict[str, object]:
    for label, tolerance in (
        ("frequency", frequency_tolerance_ha),
        ("matrix max-abs", matrix_max_abs_tolerance_ha),
        ("matrix relative", matrix_relative_tolerance),
    ):
        if not math.isfinite(tolerance) or tolerance < 0.0:
            raise TraceFormatError(f"invalid {label} tolerance")

    excluded = frozenset(exclude_components or ())
    selected = None if iterations is None else set(iterations)
    old_header = parse_contract_header(old_text, "old trace")
    new_header = parse_contract_header(new_text, "new trace")
    old_rows = parse_rows(old_text, "old trace", selected, channel)
    new_rows = parse_rows(new_text, "new trace", selected, channel)

    contract_differences = compare_contract_keys(old_header, new_header)

    old_keys = set(old_rows)
    new_keys = set(new_rows)
    missing = sorted(old_keys - new_keys)[:5]
    extra = sorted(new_keys - old_keys)[:5]
    common_keys = {
        key for key in old_keys & new_keys if key[2] not in excluded
    }

    frequency_differences: list[dict[str, object]] = []
    frequency_difference_count = 0
    for key in sorted(common_keys):
        old_frequency = old_rows[key][0]
        new_frequency = new_rows[key][0]
        if abs(old_frequency - new_frequency) > frequency_tolerance_ha:
            frequency_difference_count += 1
            if len(frequency_differences) < 5:
                frequency_differences.append(
                    {
                        "key": list(key),
                        "old_frequency_ha": old_frequency,
                        "new_frequency_ha": new_frequency,
                    }
                )

    components = _component_metrics(
        old_rows,
        new_rows,
        common_keys,
        matrix_max_abs_tolerance_ha,
        matrix_relative_tolerance,
    )
    components_passed = all(
        bool(metric["passed"]) for metric in components.values()
    )
    passed = bool(
        not contract_differences
        and not missing
        and not extra
        and frequency_difference_count == 0
        and components_passed
    )
    return {
        "passed": passed,
        "contract_differences": contract_differences,
        "components": components,
        "excluded_components": sorted(excluded),
        "row_count": {"old": len(old_rows), "new": len(new_rows)},
        "key_differences": {
            "missing": [list(key) for key in missing],
            "extra": [list(key) for key in extra],
        },
        "frequency_differences": frequency_differences,
        "frequency_difference_count": frequency_difference_count,
        "iterations": iterations,
        "channel": channel,
        "tolerances": {
            "frequency_ha": frequency_tolerance_ha,
            "matrix_max_abs_ha": matrix_max_abs_tolerance_ha,
            "matrix_relative": matrix_relative_tolerance,
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Compare two current (contract v5) QSGW traces and verify that "
            "two runs of the same computation agree numerically."
        )
    )
    parser.add_argument("old_trace", type=Path)
    parser.add_argument("new_trace", type=Path)
    parser.add_argument(
        "output_json",
        type=Path,
        nargs="?",
        help="optional path to also write the JSON report",
    )
    parser.add_argument(
        "--iterations",
        default=None,
        help=(
            "inclusive range like 0:2 or comma list like 0,1; "
            "default selects all iterations"
        ),
    )
    parser.add_argument("--channel", type=int, default=0)
    parser.add_argument(
        "--frequency-tolerance",
        type=float,
        default=DEFAULT_FREQUENCY_TOLERANCE_HA,
    )
    parser.add_argument(
        "--matrix-max-abs-tolerance-ha",
        type=float,
        default=DEFAULT_MATRIX_MAX_ABS_TOLERANCE_HA,
    )
    parser.add_argument(
        "--matrix-relative-tolerance",
        type=float,
        default=DEFAULT_MATRIX_RELATIVE_TOLERANCE,
    )
    parser.add_argument(
        "--exclude-components",
        nargs="+",
        default=[],
        metavar="COMPONENT",
        help=(
            "components to skip during value/frequency comparison (e.g. "
            "gauge-dependent rotation_u wfc_spinor0); key-set equality is "
            "still enforced on the full traces"
        ),
    )
    args = parser.parse_args(argv)

    try:
        iterations = (
            None
            if args.iterations is None
            else parse_iterations(args.iterations)
        )
        report: dict[str, object] = compare_trace_text(
            args.old_trace.read_text(encoding="utf-8"),
            args.new_trace.read_text(encoding="utf-8"),
            iterations=iterations,
            channel=args.channel,
            frequency_tolerance_ha=args.frequency_tolerance,
            matrix_max_abs_tolerance_ha=args.matrix_max_abs_tolerance_ha,
            matrix_relative_tolerance=args.matrix_relative_tolerance,
            exclude_components=frozenset(args.exclude_components),
        )
    except (TraceFormatError, OSError, ValueError) as error:
        report = {"passed": False, "error": str(error)}
        exit_code = 1
    else:
        exit_code = 0 if report["passed"] else 2
    report["old_trace"] = str(args.old_trace.resolve())
    report["new_trace"] = str(args.new_trace.resolve())
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.output_json is not None:
        args.output_json.write_text(
            rendered + "\n", encoding="utf-8", newline="\n"
        )
    print(rendered)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
