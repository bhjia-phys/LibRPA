#!/usr/bin/env python3
"""Convert ABACUS band_vxck triangular Vxc matrices to LibRPA band_vxc_k.

ABACUS band_vxck*_nao.txt files store the upper triangle of the Vxc
matrix in Rydberg (row i holds columns i..n). The upstream LibRPA
g0w0_band reader (driver/read_data.cpp: read_vxc_band) instead expects
band_vxc_k_NNNNN.txt files with one ``spin band value`` row per band,
holding the per-band diagonal Vxc in Hartree. Conversion: diagonal real
part divided by 2 (Ry -> Ha). Verified against the May 2026 reference
pair (diag Row1 -8.00126971e-01 Ry -> -4.0006348548059606E-01 Ha).
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


class ConvertError(ValueError):
    """Raised when a band_vxck file violates the expected layout."""


VALUE = re.compile(
    r"\(\s*(-?\d+(?:\.\d+)?[eE][+-]\d+)\s*,\s*(-?\d+(?:\.\d+)?[eE][+-]\d+)\s*\)"
)


def parse_band_vxck(path: Path) -> tuple[list[float], int, int]:
    rows: list[list[float]] = []
    current: list[float] | None = None
    n_rows = n_columns = 0
    for line_number, line in enumerate(
        path.read_text(encoding="ascii").splitlines(), 1
    ):
        stripped = line.strip()
        if stripped.startswith("#"):
            fields = stripped[1:].split()
            if fields[:1] == ["rows"]:
                n_rows = int(fields[1])
            elif fields[:1] == ["columns"]:
                n_columns = int(fields[1])
            continue
        if stripped.lower().startswith("row"):
            current = []
            rows.append(current)
            continue
        if current is None:
            raise ConvertError(f"{path}:{line_number}: values before Row marker")
        for match in VALUE.finditer(line):
            current.append(complex(float(match.group(1)), float(match.group(2))))
    if n_rows <= 0 or n_columns <= 0:
        raise ConvertError(f"{path}: missing rows/columns header")
    if len(rows) != n_rows:
        raise ConvertError(
            f"{path}: expected {n_rows} Row blocks, got {len(rows)}"
        )
    diagonal: list[float] = []
    for index, block in enumerate(rows):
        expected = n_columns - index
        if len(block) != expected:
            raise ConvertError(
                f"{path}: Row {index + 1} holds {len(block)} values, "
                f"expected {expected} (triangular layout)"
            )
        value = block[0]
        if abs(value.imag) > 1.0e-12:
            raise ConvertError(
                f"{path}: Row {index + 1} diagonal has non-negligible "
                f"imaginary part {value.imag:.3e}"
            )
        diagonal.append(value.real)
    return diagonal, n_rows, n_columns


def convert_file(source: Path, target: Path, n_spins: int = 1) -> dict[str, object]:
    diagonal, n_rows, n_columns = parse_band_vxck(source)
    if n_rows != n_columns:
        raise ConvertError(f"{source}: non-square matrix {n_rows}x{n_columns}")
    if n_spins <= 0:
        raise ConvertError("n_spins must be positive")
    lines: list[str] = []
    for spin in range(1, n_spins + 1):
        for band, value in enumerate(diagonal, 1):
            lines.append(f"{spin:8d}{band:8d}{value / 2.0:24.16E}\n")
    content = "".join(lines)
    changed = not target.exists() or target.read_text(encoding="ascii") != content
    return {
        "n_bands": len(diagonal),
        "rows_written": n_spins * len(diagonal),
        "content": content,
        "changed": changed,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("target", type=Path)
    parser.add_argument("--n-spins", type=int, default=1)
    parser.add_argument("--check", action="store_true",
                        help="only report, do not write")
    args = parser.parse_args()
    try:
        result = convert_file(args.source, args.target, args.n_spins)
    except ConvertError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(
        f"n_bands={result['n_bands']} rows_written={result['rows_written']} "
        f"changed={result['changed']}"
    )
    if not args.check:
        args.target.write_text(result["content"], encoding="ascii", newline="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
