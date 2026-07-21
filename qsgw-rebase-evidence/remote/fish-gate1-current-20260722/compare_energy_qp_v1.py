#!/usr/bin/env python3
"""Compare LibRPA energy_qp tables without requiring text identity."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path


FLOAT = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][+-]?\d+)?"
KPOINT_RE = re.compile(
    rf"^\s*K_point\s+(\d+)\s*:\s*({FLOAT})\s+({FLOAT})\s+({FLOAT})\s*$"
)
STATE_RE = re.compile(
    rf"^\s*(\d+)\s+({FLOAT})\s+({FLOAT})\s+({FLOAT})\s*$"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def finite(token: str, context: str) -> float:
    value = float(token)
    if not math.isfinite(value):
        raise ValueError(f"{context}: non-finite value {token!r}")
    return value


def parse_energy_qp(path: Path) -> dict[str, object]:
    kpoints: dict[int, tuple[float, float, float]] = {}
    states: dict[tuple[int, int], tuple[float, float, float]] = {}
    current_kpoint: int | None = None
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        kpoint_match = KPOINT_RE.fullmatch(line)
        if kpoint_match:
            current_kpoint = int(kpoint_match.group(1))
            if current_kpoint in kpoints:
                raise ValueError(f"{path}: duplicate k-point {current_kpoint}")
            kpoints[current_kpoint] = tuple(
                finite(kpoint_match.group(index), f"{path}:{line_number}")
                for index in range(2, 5)
            )
            continue
        state_match = STATE_RE.fullmatch(line)
        if state_match:
            if current_kpoint is None:
                raise ValueError(f"{path}:{line_number}: state row before a k-point")
            state = int(state_match.group(1))
            key = (current_kpoint, state)
            if key in states:
                raise ValueError(f"{path}: duplicate state key {key}")
            states[key] = tuple(
                finite(state_match.group(index), f"{path}:{line_number}")
                for index in range(2, 5)
            )
            continue
        stripped = line.strip()
        if stripped and not stripped.startswith("state") and set(stripped) != {"-"}:
            raise ValueError(f"{path}:{line_number}: unrecognized line {stripped!r}")
    if not kpoints:
        raise ValueError(f"{path}: no k-point headers")
    if not states:
        raise ValueError(f"{path}: no state rows")
    return {"kpoints": kpoints, "states": states}


def maximum_difference(
    reference: dict[tuple[int, int], tuple[float, float, float]],
    candidate: dict[tuple[int, int], tuple[float, float, float]],
    column: int,
) -> tuple[float, tuple[int, int], int]:
    differences = {
        key: abs(reference[key][column] - candidate[key][column]) for key in reference
    }
    location = max(differences, key=differences.__getitem__)
    return (
        differences[location],
        location,
        sum(difference != 0.0 for difference in differences.values()),
    )


def compare_files(
    reference_path: Path, candidate_path: Path, qp_tolerance: float
) -> dict[str, object]:
    if qp_tolerance < 0.0 or not math.isfinite(qp_tolerance):
        raise ValueError("QP tolerance must be finite and non-negative")
    reference = parse_energy_qp(reference_path)
    candidate = parse_energy_qp(candidate_path)
    reference_kpoints = reference["kpoints"]
    candidate_kpoints = candidate["kpoints"]
    reference_states = reference["states"]
    candidate_states = candidate["states"]
    assert isinstance(reference_kpoints, dict)
    assert isinstance(candidate_kpoints, dict)
    assert isinstance(reference_states, dict)
    assert isinstance(candidate_states, dict)
    if reference_kpoints.keys() != candidate_kpoints.keys():
        raise ValueError("k-point keys differ")
    if reference_states.keys() != candidate_states.keys():
        raise ValueError("state keys differ")

    coordinate_max_abs = max(
        abs(reference_kpoints[ik][axis] - candidate_kpoints[ik][axis])
        for ik in reference_kpoints
        for axis in range(3)
    )
    occupation_max, occupation_location, occupation_nonzero = maximum_difference(
        reference_states, candidate_states, 0
    )
    ks_max, ks_location, ks_nonzero = maximum_difference(
        reference_states, candidate_states, 1
    )
    qp_max, qp_location, qp_nonzero = maximum_difference(
        reference_states, candidate_states, 2
    )
    checks = {
        "kpoint_coordinates_exact": coordinate_max_abs == 0.0,
        "occupations_exact": occupation_max == 0.0,
        "ks_energies_exact": ks_max == 0.0,
        "qp_energies_within_tolerance": qp_max <= qp_tolerance,
    }
    return {
        "schema": "energy-qp-comparison-v1",
        "passed": all(checks.values()),
        "reference_path": str(reference_path),
        "candidate_path": str(candidate_path),
        "reference_sha256": sha256(reference_path),
        "candidate_sha256": sha256(candidate_path),
        "kpoint_count": len(reference_kpoints),
        "state_count": len(reference_states),
        "kpoint_coordinate_max_abs_difference": coordinate_max_abs,
        "occupation_max_abs_difference": occupation_max,
        "occupation_max_abs_location": list(occupation_location),
        "occupation_nonzero_difference_count": occupation_nonzero,
        "ks_energy_max_abs_difference_ha": ks_max,
        "ks_energy_max_abs_location": list(ks_location),
        "ks_energy_nonzero_difference_count": ks_nonzero,
        "qp_energy_max_abs_difference_ha": qp_max,
        "qp_energy_max_abs_location": list(qp_location),
        "qp_energy_nonzero_difference_count": qp_nonzero,
        "thresholds": {"qp_energy_max_abs_tolerance_ha": qp_tolerance},
        "checks": checks,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("reference_path", type=Path)
    parser.add_argument("candidate_path", type=Path)
    parser.add_argument("report_path", type=Path)
    parser.add_argument("--max-abs-tolerance-ha", type=float, required=True)
    args = parser.parse_args()
    report = compare_files(
        args.reference_path.resolve(),
        args.candidate_path.resolve(),
        args.max_abs_tolerance_ha,
    )
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    args.report_path.write_text(rendered, encoding="utf-8", newline="\n")
    print(rendered, end="")
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
