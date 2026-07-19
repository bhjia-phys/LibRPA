#!/usr/bin/env python3
"""Normalize ABACUS band_KS_eigenvalue occupation weights to per-state.

ABACUS band-mode band_KS_eigenvalue files store k-weighted occupations
(occupation * k_weight, e.g. 2/201 for a doubly occupied band on a
201-point path). The shared LibRPA band reader
(src/api/input.cpp: librpa_set_band_occ_eigval) instead expects per-state
occupations (like ABACUS band_out for the SCF grid, e.g. 2.0) and applies
an additional 1/n_kpts_band factor. Normalizing the weight column to
per-state occupations makes the reader produce the physically correct
k-weighted mean-field occupation. The transform multiplies column 3 by
n_band_kpoints; all other bytes are preserved.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


class NormalizationError(ValueError):
    """Raised when a band eigenvalue file violates the expected format."""


def normalize_file(path: Path, n_band_kpoints: int) -> dict[str, object]:
    if n_band_kpoints <= 0:
        raise NormalizationError("n_band_kpoints must be positive")
    lines = path.read_text(encoding="ascii").splitlines(keepends=True)
    output: list[str] = []
    rows = 0
    max_weight = 0.0
    for line_number, line in enumerate(lines, 1):
        stripped = line.strip()
        if not stripped:
            output.append(line)
            continue
        fields = stripped.split()
        if len(fields) != 5:
            raise NormalizationError(
                f"{path}:{line_number}: expected 5 columns, got {len(fields)}"
            )
        try:
            spin = int(fields[0])
            band = int(fields[1])
            weight = float(fields[2])
            eigenvalue_ha = float(fields[3])
            eigenvalue_ev = float(fields[4])
        except ValueError as error:
            raise NormalizationError(
                f"{path}:{line_number}: non-numeric field"
            ) from error
        if spin <= 0 or band <= 0:
            raise NormalizationError(
                f"{path}:{line_number}: non-positive spin/band index"
            )
        if weight < 0.0:
            raise NormalizationError(
                f"{path}:{line_number}: negative occupation weight"
            )
        per_state = weight * n_band_kpoints
        max_weight = max(max_weight, per_state)
        newline = (
            f"{spin:8d}{band:8d}{per_state:24.16E}"
            f"{eigenvalue_ha:24.16E}{eigenvalue_ev:24.16E}\n"
        )
        output.append(newline)
        rows += 1
    if rows == 0:
        raise NormalizationError(f"{path}: no occupation rows")
    changed = path.read_text(encoding="ascii") != "".join(output)
    return {
        "rows": rows,
        "max_per_state_weight": max_weight,
        "changed": changed,
        "content": "".join(output),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("file", type=Path)
    parser.add_argument("n_band_kpoints", type=int)
    parser.add_argument("--check", action="store_true",
                        help="only report, do not write")
    args = parser.parse_args()
    try:
        result = normalize_file(args.file, args.n_band_kpoints)
    except NormalizationError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(
        f"rows={result['rows']} "
        f"max_per_state_weight={result['max_per_state_weight']:.6e} "
        f"changed={result['changed']}"
    )
    if not args.check:
        args.file.write_text(result["content"], encoding="ascii", newline="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
