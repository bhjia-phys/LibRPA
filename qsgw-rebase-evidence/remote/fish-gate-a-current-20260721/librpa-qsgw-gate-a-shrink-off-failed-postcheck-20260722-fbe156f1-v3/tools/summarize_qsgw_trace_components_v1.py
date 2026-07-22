#!/usr/bin/env python3
"""Summarize absolute component magnitudes in a QSGW matrix trace."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def summarize(path: Path, selected_iterations: set[int]) -> dict[str, object]:
    groups: dict[tuple[int, int, str], dict[str, object]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            fields = line.split()
            if len(fields) != 11:
                raise ValueError(f"{path}:{line_number}: expected 11 columns")
            iteration = int(fields[0])
            if iteration not in selected_iterations:
                continue
            channel = int(fields[1])
            component = fields[2]
            value = complex(float(fields[9]), float(fields[10]))
            magnitude = abs(value)
            if not math.isfinite(magnitude):
                raise ValueError(f"{path}:{line_number}: non-finite value")
            key = (iteration, channel, component)
            group = groups.setdefault(
                key,
                {
                    "iteration": iteration,
                    "channel": channel,
                    "component": component,
                    "count": 0,
                    "max_abs": 0.0,
                    "sum_abs_square": 0.0,
                    "max_location": None,
                },
            )
            group["count"] = int(group["count"]) + 1
            group["sum_abs_square"] = (
                float(group["sum_abs_square"]) + magnitude * magnitude
            )
            if magnitude > float(group["max_abs"]):
                group["max_abs"] = magnitude
                group["max_location"] = {
                    "spin": int(fields[3]),
                    "kpoint": int(fields[4]),
                    "frequency_index": int(fields[5]),
                    "frequency_ha": float(fields[6]),
                    "row": int(fields[7]),
                    "column": int(fields[8]),
                    "real": value.real,
                    "imag": value.imag,
                    "line": line_number,
                }
    if not groups:
        raise ValueError("no selected trace rows")
    output_groups: list[dict[str, object]] = []
    for key in sorted(groups):
        group = groups[key]
        group["frobenius"] = math.sqrt(float(group.pop("sum_abs_square")))
        output_groups.append(group)
    return {
        "schema": "librpa-qsgw-trace-component-summary-v1",
        "source": str(path.resolve()),
        "selected_iterations": sorted(selected_iterations),
        "groups": output_groups,
    }


def _iterations(spec: str) -> set[int]:
    if ":" in spec:
        first, last = (int(value) for value in spec.split(":", 1))
        if last < first:
            raise ValueError("iteration range is reversed")
        return set(range(first, last + 1))
    values = {int(value) for value in spec.split(",")}
    if not values:
        raise ValueError("no iterations selected")
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("trace", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--iterations", required=True)
    args = parser.parse_args()
    try:
        report = summarize(args.trace, _iterations(args.iterations))
        report["passed"] = True
    except Exception as error:
        report = {"passed": False, "error": str(error)}
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    raise SystemExit(0 if report["passed"] else 2)


if __name__ == "__main__":
    main()
