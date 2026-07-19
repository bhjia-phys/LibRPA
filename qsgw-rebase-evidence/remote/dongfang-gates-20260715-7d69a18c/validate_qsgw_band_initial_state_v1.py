#!/usr/bin/env python3
"""Validate the iteration-zero qsgw_band eigenvalue-channel initial state.

The new QSGW band trace (channel 1) must reproduce, at iteration zero, the
Kohn-Sham eigenvalues shipped as ``band_KS_eigenvalue_k_NNNNN.txt`` inputs
(one file per band k-point, NNNNN starting at 00001). The observer checks
that the trace (spin, kpoint, band) key set matches the input files exactly
(zero-based trace indices map to one-based file spin/band indices), that
every eigenvalue agrees within a Hartree tolerance, and that occupations
are consistent: when an occupation trace is supplied its channel-1
iteration-0 values are compared against the input weights key-for-key,
otherwise the per-k-point weight totals are checked for mutual
self-consistency.

Trace data rows carry exactly nine fields
``iter channel spin kpoint f1 f2 f3 band eigenvalue_eV`` (comment lines
start with ``#``; f1-f3 are floats, typically kx ky kz, and are not
checked). Input eigenvalue rows carry five fields
``spin band weight eigenvalue_ha eigenvalue_eV`` with one-based spin/band.

Exit status: 0 when the report passes, 2 when validation fails, and 1 on
malformed inputs (missing files/columns, duplicate keys, non-finite
values); every exit writes the JSON report.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


HA_TO_EV = 27.211386245988

BandKey = tuple[int, int, int]  # (spin, kpoint, band), zero-based


def parse_iterations(specification: str) -> list[int]:
    if ":" in specification:
        start_text, stop_text = specification.split(":", 1)
        start, stop = int(start_text), int(stop_text)
        if start < 0 or stop < start:
            raise ValueError("invalid iteration range")
        return list(range(start, stop + 1))
    result = [int(value) for value in specification.split(",")]
    if not result or any(value < 0 for value in result):
        raise ValueError("invalid iteration list")
    return result


def parse_trace_scalar_column(
    text: str, label: str, iterations: set[int], channel: int
) -> dict[BandKey, float]:
    """Collect the last column of selected trace rows keyed (spin, kpoint, band)."""
    result: dict[BandKey, float] = {}
    for line_number, line in enumerate(text.splitlines(), 1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        fields = line.split()
        if len(fields) != 9:
            raise ValueError(f"{label}:{line_number}: expected 9 fields")
        try:
            iteration = int(fields[0])
            observed_channel = int(fields[1])
        except ValueError as exc:
            raise ValueError(
                f"{label}:{line_number}: invalid iter/channel field"
            ) from exc
        if iteration not in iterations or observed_channel != channel:
            continue
        try:
            spin, kpoint, band = (
                int(fields[2]),
                int(fields[3]),
                int(fields[7]),
            )
            values = [float(fields[index]) for index in (4, 5, 6, 8)]
        except ValueError as exc:
            raise ValueError(
                f"{label}:{line_number}: invalid numeric field"
            ) from exc
        if spin < 0 or kpoint < 0 or band < 0:
            raise ValueError(
                f"{label}:{line_number}: negative spin/kpoint/band index"
            )
        if not all(math.isfinite(value) for value in values):
            raise ValueError(f"{label}:{line_number}: non-finite value")
        key = (spin, kpoint, band)
        if key in result:
            raise ValueError(f"{label}:{line_number}: duplicate row {key}")
        result[key] = values[3]
    return result


def parse_band_eigenvalue_file(
    text: str, label: str
) -> dict[tuple[int, int], tuple[float, float]]:
    """Parse one band_KS_eigenvalue file into {(spin, band): (weight, ha)} (one-based)."""
    result: dict[tuple[int, int], tuple[float, float]] = {}
    for line_number, line in enumerate(text.splitlines(), 1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        fields = line.split()
        if len(fields) != 5:
            raise ValueError(f"{label}:{line_number}: expected 5 fields")
        try:
            spin, band = int(fields[0]), int(fields[1])
            weight = float(fields[2])
            eigenvalue_ha = float(fields[3])
            eigenvalue_ev = float(fields[4])
        except ValueError as exc:
            raise ValueError(
                f"{label}:{line_number}: invalid numeric field"
            ) from exc
        if spin < 1 or band < 1:
            raise ValueError(
                f"{label}:{line_number}: spin/band must be one-based"
            )
        if not all(
            math.isfinite(value)
            for value in (weight, eigenvalue_ha, eigenvalue_ev)
        ):
            raise ValueError(f"{label}:{line_number}: non-finite value")
        key = (spin, band)
        if key in result:
            raise ValueError(f"{label}:{line_number}: duplicate row {key}")
        result[key] = (weight, eigenvalue_ha)
    if not result:
        raise ValueError(f"{label}: no band rows")
    return result


def load_band_eigenvalue_dir(
    directory: Path, n_band_kpoints: int
) -> tuple[dict[BandKey, tuple[float, float]], int, int]:
    """Load all k-point files into zero-based {(spin, kpoint, band): (weight, ha)}."""
    if n_band_kpoints <= 0:
        raise ValueError("n_band_kpoints must be positive")
    combined: dict[BandKey, tuple[float, float]] = {}
    layout: set[tuple[int, int]] | None = None
    for kpoint in range(n_band_kpoints):
        path = directory / f"band_KS_eigenvalue_k_{kpoint + 1:05d}.txt"
        if not path.is_file():
            raise ValueError(f"missing band eigenvalue file: {path}")
        rows = parse_band_eigenvalue_file(
            path.read_text(encoding="utf-8"),
            f"band eigenvalue file {path.name}",
        )
        if layout is None:
            layout = set(rows)
        elif set(rows) != layout:
            raise ValueError(
                f"band eigenvalue file {path.name} (spin, band) layout "
                "differs from the first k-point file"
            )
        for (spin, band), value in rows.items():
            combined[(spin - 1, kpoint, band - 1)] = value
    assert layout is not None
    n_spins = max(spin for spin, _ in layout)
    n_bands = max(band for _, band in layout)
    expected = {
        (spin, band)
        for spin in range(1, n_spins + 1)
        for band in range(1, n_bands + 1)
    }
    if layout != expected:
        raise ValueError(
            "band eigenvalue files do not form a rectangular spin x band layout"
        )
    return combined, n_spins, n_bands


def _sorted_key_lists(keys) -> list[list[int]]:
    return [list(key) for key in sorted(keys)]


def validate_band_initial_state(
    *,
    trace_text: str,
    band_dir: Path,
    n_band_kpoints: int,
    iterations: list[int],
    eigenvalue_tolerance_ha: float = 1.0e-12,
    occupation_tolerance: float = 1.0e-12,
    occupation_trace_text: str | None = None,
    input_occupation_convention: str = "kweighted",
) -> dict[str, object]:
    if iterations != [0]:
        raise ValueError(
            "band initial-state validation requires --iterations 0"
        )
    if input_occupation_convention not in {"kweighted", "per-state"}:
        raise ValueError(
            "invalid input occupation convention "
            f"{input_occupation_convention!r}"
        )
    for name, value in (
        ("eigenvalue", eigenvalue_tolerance_ha),
        ("occupation", occupation_tolerance),
    ):
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"invalid {name} tolerance")

    # ABACUS band-mode files are k-weighted (occupation * k_weight);
    # per-state files (like band_out) must be divided by the k-point count
    # before comparison against the trace's mean-field weights.
    occupation_scale = (
        1.0
        if input_occupation_convention == "kweighted"
        else 1.0 / n_band_kpoints
    )

    trace_ev = parse_trace_scalar_column(
        trace_text, "eigenvalue trace", {0}, 1
    )
    band_side, n_spins, n_bands = load_band_eigenvalue_dir(
        Path(band_dir), n_band_kpoints
    )

    trace_keys = set(trace_ev)
    band_keys = set(band_side)
    common = sorted(trace_keys & band_keys)
    key_mismatch = None
    if trace_keys != band_keys:
        key_mismatch = {
            "missing_in_trace": _sorted_key_lists(band_keys - trace_keys)[:8],
            "missing_in_band": _sorted_key_lists(trace_keys - band_keys)[:8],
            "missing_in_trace_count": len(band_keys - trace_keys),
            "missing_in_band_count": len(trace_keys - band_keys),
        }

    eigenvalue_max = 0.0
    diffs: list[tuple[float, str, BandKey]] = []
    for key in common:
        eigenvalue_ha = band_side[key][1]
        diff = abs(trace_ev[key] / HA_TO_EV - eigenvalue_ha)
        eigenvalue_max = max(eigenvalue_max, diff)
        diffs.append((diff, "eigenvalue", key))

    occupation_max = 0.0
    occupation_check = "band_weight_total_self_consistency"
    occupation_key_mismatch = None
    if occupation_trace_text is not None:
        occupation_check = "trace_occupation"
        occupation = parse_trace_scalar_column(
            occupation_trace_text, "occupation trace", {0}, 1
        )
        occupation_keys = set(occupation)
        if occupation_keys != band_keys:
            occupation_key_mismatch = {
                "missing_in_occupation_trace": _sorted_key_lists(
                    band_keys - occupation_keys
                )[:8],
                "unexpected_in_occupation_trace": _sorted_key_lists(
                    occupation_keys - band_keys
                )[:8],
            }
        for key in sorted(occupation_keys & band_keys):
            diff = abs(occupation[key] - band_side[key][0] * occupation_scale)
            occupation_max = max(occupation_max, diff)
            diffs.append((diff, "occupation", key))
    else:
        totals: dict[int, float] = {}
        for (_spin, kpoint, _band), (weight, _ha) in band_side.items():
            totals[kpoint] = totals.get(kpoint, 0.0) + weight
        reference = totals[0]
        occupation_max = max(
            abs(total - reference) for total in totals.values()
        )

    worst = [
        {
            "check": kind,
            "spin": key[0],
            "kpoint": key[1],
            "band": key[2],
            "abs_diff": diff,
        }
        for diff, kind, key in sorted(diffs, reverse=True)[:5]
    ]

    passed = (
        key_mismatch is None
        and occupation_key_mismatch is None
        and eigenvalue_max <= eigenvalue_tolerance_ha
        and occupation_max <= occupation_tolerance
    )
    report: dict[str, object] = {
        "passed": passed,
        "task": "qsgw_band",
        "channel": 1,
        "iterations": [0],
        "n_band_kpoints": n_band_kpoints,
        "n_spins": n_spins,
        "n_bands": n_bands,
        "key_count": len(common),
        "expected_key_count": n_band_kpoints * n_bands * n_spins,
        "eigenvalue_max_abs_diff_ha": eigenvalue_max,
        "occupation_max_abs_diff": occupation_max,
        "occupation_check": occupation_check,
        "input_occupation_convention": input_occupation_convention,
        "tolerances": {
            "eigenvalue_ha": eigenvalue_tolerance_ha,
            "occupation": occupation_tolerance,
        },
        "ha_to_ev": HA_TO_EV,
        "worst": worst,
    }
    if key_mismatch is not None:
        report["key_mismatch"] = key_mismatch
    if occupation_key_mismatch is not None:
        report["occupation_key_mismatch"] = occupation_key_mismatch
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate the qsgw_band iteration-zero band-channel "
        "initial state against the band_KS_eigenvalue inputs."
    )
    parser.add_argument("eigenvalue_trace", type=Path)
    parser.add_argument("band_eigenvalue_dir", type=Path)
    parser.add_argument("n_band_kpoints", type=int)
    parser.add_argument("output_json", type=Path)
    parser.add_argument("--iterations", default="0")
    parser.add_argument("--eigenvalue-tolerance-ha", type=float, default=1.0e-12)
    parser.add_argument("--occupation-tolerance", type=float, default=1.0e-12)
    parser.add_argument(
        "--occupation-trace",
        type=Path,
        default=None,
        help="optional trace whose channel-1 iteration-0 last column "
        "carries occupations to compare against the input weights",
    )
    parser.add_argument(
        "--input-occupation-convention",
        choices=("kweighted", "per-state"),
        default="kweighted",
        help="occupation convention of the band_KS_eigenvalue inputs: "
        "kweighted (ABACUS band-mode, occupation*k_weight, default) or "
        "per-state (band_out style, divided by n_band_kpoints before "
        "comparison)",
    )
    args = parser.parse_args()
    try:
        report = validate_band_initial_state(
            trace_text=args.eigenvalue_trace.read_text(encoding="utf-8"),
            band_dir=args.band_eigenvalue_dir,
            n_band_kpoints=args.n_band_kpoints,
            iterations=parse_iterations(args.iterations),
            eigenvalue_tolerance_ha=args.eigenvalue_tolerance_ha,
            occupation_tolerance=args.occupation_tolerance,
            occupation_trace_text=(
                args.occupation_trace.read_text(encoding="utf-8")
                if args.occupation_trace is not None
                else None
            ),
            input_occupation_convention=args.input_occupation_convention,
        )
    except (OSError, ValueError) as error:
        report = {"passed": False, "error": str(error)}
        args.output_json.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(report, indent=2, sort_keys=True))
        raise SystemExit(1)
    args.output_json.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    raise SystemExit(0 if report["passed"] else 2)


if __name__ == "__main__":
    main()
