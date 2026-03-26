#!/usr/bin/env python3
"""Prepare band_* topology inputs from a regular-grid LibRPA/ABACUS bundle.

This utility is intended for topology smoke validation when a case already has
regular-grid files such as:
  - band_out
  - KS_eigenvector_*.dat
  - vxc_out
  - stru_out

It writes a reduced `band_*` bundle that `topo_gw_band` can read directly.
"""

from __future__ import annotations

import argparse
import math
import struct
from array import array
from pathlib import Path

HA2EV = 27.211386245988


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", required=True, help="Source regular-grid run directory")
    parser.add_argument("--dst", required=True, help="Destination directory for generated band_* files")
    parser.add_argument(
        "--indices",
        required=True,
        help="Comma-separated 1-based regular-grid k-point indices to keep, in target mesh order",
    )
    return parser.parse_args()


def gaussian_solve_3x3(a: list[list[float]], b: list[float]) -> list[float]:
    aug = [row[:] + [rhs] for row, rhs in zip(a, b)]
    for col in range(3):
        pivot = max(range(col, 3), key=lambda row: abs(aug[row][col]))
        if abs(aug[pivot][col]) < 1.0e-14:
            raise ValueError("Singular 3x3 system while converting k-point coordinates")
        aug[col], aug[pivot] = aug[pivot], aug[col]
        pivot_val = aug[col][col]
        for j in range(col, 4):
            aug[col][j] /= pivot_val
        for row in range(3):
            if row == col:
                continue
            factor = aug[row][col]
            for j in range(col, 4):
                aug[row][j] -= factor * aug[col][j]
    return [aug[i][3] for i in range(3)]


def fold_frac(x: float) -> float:
    x = math.fmod(x, 1.0)
    if x < 0:
        x += 1.0
    if abs(x) < 1.0e-6 or abs(x - 1.0) < 1.0e-6:
        return 0.0
    if abs(x - 0.5) < 1.0e-6:
        return 0.5
    if abs(x - 0.25) < 1.0e-6:
        return 0.25
    if abs(x - 0.75) < 1.0e-6:
        return 0.75
    return round(x, 12)


def parse_stru_out(path: Path) -> tuple[list[list[float]], list[tuple[float, float, float]]]:
    lines = path.read_text().splitlines()
    recip = [list(map(float, lines[i].split())) for i in range(3, 6)]
    natom = int(lines[6].split()[0])
    nk = list(map(int, lines[7 + natom].split()))
    nkpts = nk[0] * nk[1] * nk[2]
    kcart = [tuple(map(float, lines[8 + natom + i].split()[:3])) for i in range(nkpts)]
    return recip, kcart


def cart_to_frac(recip_rows: list[list[float]], kcart: tuple[float, float, float]) -> tuple[float, float, float]:
    a = [
        [recip_rows[0][0], recip_rows[1][0], recip_rows[2][0]],
        [recip_rows[0][1], recip_rows[1][1], recip_rows[2][1]],
        [recip_rows[0][2], recip_rows[1][2], recip_rows[2][2]],
    ]
    frac = gaussian_solve_3x3(a, list(kcart))
    return tuple(fold_frac(x) for x in frac)


def parse_band_out(path: Path) -> tuple[int, int, int, int, float, list[list[list[tuple[float, float]]]]]:
    lines = path.read_text().splitlines()
    nkpts = int(lines[0].strip())
    nspins = int(lines[1].strip())
    nbands = int(lines[2].strip())
    nbasis = int(lines[3].strip())
    efermi = float(lines[4].strip())
    cursor = 5
    data: list[list[list[tuple[float, float]]]] = []
    for _ik in range(nkpts):
        spin_blocks: list[list[tuple[float, float]]] = []
        for _ispin in range(nspins):
            header = lines[cursor].split()
            cursor += 1
            if len(header) < 2:
                raise ValueError(f"Malformed band_out block header near line {cursor}")
            band_block: list[tuple[float, float]] = []
            for _ib in range(nbands):
                parts = lines[cursor].split()
                cursor += 1
                if len(parts) < 4:
                    raise ValueError(f"Malformed band_out state line near line {cursor}")
                occ = float(parts[1])
                eig_ha = float(parts[2])
                band_block.append((occ, eig_ha))
            spin_blocks.append(band_block)
        data.append(spin_blocks)
    return nkpts, nspins, nbands, nbasis, efermi, data


def parse_vxc_out(path: Path, nkpts: int, nspins: int, nbands: int) -> list[list[list[float]]]:
    lines = path.read_text().splitlines()
    if int(lines[0].strip()) != nkpts or int(lines[1].strip()) != nspins or int(lines[2].strip()) != nbands:
        raise ValueError("vxc_out dimensions do not match band_out")
    cursor = 3
    out = []
    for _ik in range(nkpts):
        spin_blocks = []
        for _ispin in range(nspins):
            band_block = []
            for _ib in range(nbands):
                parts = lines[cursor].split()
                cursor += 1
                if not parts:
                    raise ValueError(f"Malformed vxc_out line near line {cursor}")
                band_block.append(float(parts[0]))
            spin_blocks.append(band_block)
        out.append(spin_blocks)
    return out


def parse_regular_wfc(path: Path, expected_pairs: int) -> list[float]:
    tokens = path.read_text().split()
    if not tokens:
        raise ValueError(f"Empty wavefunction file: {path}")
    values = [float(x) for x in tokens[1:]]
    if len(values) != expected_pairs * 2:
        raise ValueError(
            f"Unexpected wavefunction length in {path}: got {len(values)} doubles, "
            f"expected {expected_pairs * 2}"
        )
    return values


def write_band_kpath_info(dst: Path, nbasis: int, nbands: int, nspins: int, kfrac: list[tuple[float, float, float]]) -> None:
    with (dst / "band_kpath_info").open("w", encoding="utf-8") as handle:
        handle.write(f"{nbasis:6d} {nbands:5d} {nspins:5d} {len(kfrac):5d}\n")
        for kpt in kfrac:
            handle.write(f"{kpt[0]:18.12f} {kpt[1]:17.12f} {kpt[2]:17.12f}\n")


def write_band_eigenvalues(dst: Path, new_index: int, band_block: list[list[tuple[float, float]]]) -> None:
    with (dst / f"band_KS_eigenvalue_k_{new_index:05d}.txt").open("w", encoding="utf-8") as handle:
        for ispin, spin_block in enumerate(band_block, start=1):
            for iband, (occ, eig_ha) in enumerate(spin_block, start=1):
                handle.write(
                    f"{ispin:8d} {iband:7d} {occ:27.16E} {eig_ha:27.16E} {(eig_ha * HA2EV):27.16E}\n"
                )


def write_band_vxc(dst: Path, new_index: int, vxc_block: list[list[float]]) -> None:
    with (dst / f"band_vxc_k_{new_index:05d}.txt").open("w", encoding="utf-8") as handle:
        for ispin, spin_block in enumerate(vxc_block, start=1):
            for iband, value in enumerate(spin_block, start=1):
                handle.write(f"{ispin:8d} {iband:7d} {value:27.16E}\n")


def write_band_eigenvectors(dst: Path, new_index: int, raw_values: list[float]) -> None:
    with (dst / f"band_KS_eigenvector_k_{new_index:05d}.txt").open("wb") as handle:
        array("d", raw_values).tofile(handle)


def main() -> None:
    args = parse_args()
    src = Path(args.src).resolve()
    dst = Path(args.dst).resolve()
    dst.mkdir(parents=True, exist_ok=True)

    selected = [int(part.strip()) for part in args.indices.split(",") if part.strip()]
    if not selected:
        raise ValueError("No k-point indices were provided")

    recip, kcart = parse_stru_out(src / "stru_out")
    nkpts, nspins, nbands, nbasis, _efermi, band_data = parse_band_out(src / "band_out")
    vxc_data = parse_vxc_out(src / "vxc_out", nkpts, nspins, nbands)
    expected_pairs = nspins * nbands * nbasis

    selected_kfrac: list[tuple[float, float, float]] = []
    for new_index, old_index in enumerate(selected, start=1):
        if old_index < 1 or old_index > nkpts:
            raise ValueError(f"k-point index {old_index} is out of range 1..{nkpts}")

        old_zero = old_index - 1
        selected_kfrac.append(cart_to_frac(recip, kcart[old_zero]))
        write_band_eigenvalues(dst, new_index, band_data[old_zero])
        write_band_vxc(dst, new_index, vxc_data[old_zero])
        raw_values = parse_regular_wfc(src / f"KS_eigenvector_{old_zero}.dat", expected_pairs)
        write_band_eigenvectors(dst, new_index, raw_values)

    write_band_kpath_info(dst, nbasis, nbands, nspins, selected_kfrac)

    print(f"Prepared {len(selected)} band k-points under {dst}")
    for new_index, (old_index, frac) in enumerate(zip(selected, selected_kfrac), start=1):
        print(f"  new {new_index:02d} <- regular k {old_index:02d}: frac = {frac}")


if __name__ == "__main__":
    main()
