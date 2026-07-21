#!/usr/bin/env python3
"""Audit weighted occupations after mixing a legacy QSGW checkpoint with KS H0."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
from pathlib import Path

import numpy as np


SCHEMA = "librpa-qsgw-mixing-frontier-audit-v1"
HA2EV = 27.211386245988


class AuditError(ValueError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_band_out(path: Path) -> tuple[np.ndarray, np.ndarray, float]:
    tokens = path.read_text().split()
    if len(tokens) < 5:
        raise AuditError(f"short band_out: {path}")
    offset = 0

    def take() -> str:
        nonlocal offset
        if offset >= len(tokens):
            raise AuditError(f"truncated band_out: {path}")
        value = tokens[offset]
        offset += 1
        return value

    n_kpoints = int(take())
    n_spins = int(take())
    n_bands = int(take())
    n_states = int(take())
    efermi_ha = float(take())
    if min(n_kpoints, n_spins, n_bands, n_states) <= 0:
        raise AuditError("band_out dimensions must be positive")
    if n_states != n_bands:
        raise AuditError("band_out state and band counts differ")

    occupations = np.empty((n_spins, n_kpoints, n_bands), dtype=np.float64)
    eigenvalues = np.empty_like(occupations)
    for kpoint in range(n_kpoints):
        for spin in range(n_spins):
            file_kpoint = int(take())
            file_spin = int(take())
            if file_kpoint != kpoint + 1 or file_spin != spin + 1:
                raise AuditError(
                    "band_out spin/k-point blocks are not in canonical order"
                )
            for band in range(n_bands):
                file_band = int(take())
                occupation = float(take())
                energy_ha = float(take())
                energy_ev = float(take())
                if file_band != band + 1:
                    raise AuditError("band_out bands are not in canonical order")
                if not all(
                    math.isfinite(value)
                    for value in (occupation, energy_ha, energy_ev)
                ):
                    raise AuditError("band_out contains non-finite values")
                if abs(energy_ev - energy_ha * HA2EV) > 1.0e-5:
                    raise AuditError("band_out Hartree/eV columns are inconsistent")
                occupations[spin, kpoint, band] = occupation / n_kpoints
                eigenvalues[spin, kpoint, band] = energy_ha
    if offset != len(tokens):
        raise AuditError(f"trailing band_out tokens: {len(tokens) - offset}")
    return eigenvalues, occupations, efermi_ha


def parse_bz_weights(path: Path, expected_kpoints: int) -> np.ndarray:
    lines = [line.split() for line in path.read_text().splitlines() if line.split()]
    if len(lines) < 2:
        raise AuditError(f"short bz_sampling_out: {path}")
    if len(lines[0]) != 3 or len(lines[1]) != 2:
        raise AuditError("unexpected bz_sampling_out header")
    n_kpoints = int(lines[1][0])
    if n_kpoints != expected_kpoints or len(lines) != n_kpoints + 2:
        raise AuditError("bz_sampling_out k-point count mismatch")
    weights = np.empty(n_kpoints, dtype=np.float64)
    for index, fields in enumerate(lines[2:]):
        if len(fields) < 2 or int(fields[0]) != index + 1:
            raise AuditError("invalid bz_sampling_out k-point row")
        weights[index] = float(fields[1])
    if not np.all(np.isfinite(weights)) or np.any(weights < 0.0):
        raise AuditError("invalid BZ weights")
    if abs(float(np.sum(weights)) - 1.0) > 1.0e-12:
        raise AuditError("BZ weights do not sum to one")
    return weights


def read_matz_binary(path: Path) -> np.ndarray:
    data = path.read_bytes()
    if len(data) < 8:
        raise AuditError(f"short Matz file: {path}")
    rows, columns = struct.unpack_from("<ii", data, 0)
    expected = 8 + rows * columns * 16
    if rows <= 0 or columns <= 0 or len(data) != expected:
        raise AuditError(f"invalid Matz payload: {path}")
    values = np.frombuffer(data, dtype="<c16", offset=8).copy()
    return values.reshape((rows, columns))


def read_legacy_hamiltonian(
    checkpoint_root: Path,
    iteration: int,
    n_spins: int,
    n_kpoints: int,
    n_bands: int,
) -> np.ndarray:
    result = np.empty(
        (n_spins, n_kpoints, n_bands, n_bands), dtype=np.complex128
    )
    directory = checkpoint_root / f"iter_{iteration:05d}"
    for spin in range(n_spins):
        for kpoint in range(n_kpoints):
            path = directory / (
                f"H0_GW_spin_{spin + 1:02d}_k_{kpoint + 1:06d}.bin"
            )
            matrix = read_matz_binary(path)
            if matrix.shape != (n_bands, n_bands):
                raise AuditError(f"unexpected H0 shape in {path}: {matrix.shape}")
            upper = np.triu(matrix)
            hermitian = upper + np.triu(matrix, 1).conj().T
            diagonal = np.diag_indices(n_bands)
            hermitian[diagonal] = hermitian[diagonal].real
            result[spin, kpoint] = hermitian
    return result


def weighted_fill(
    eigenvalues: np.ndarray,
    kpoint_weights: np.ndarray,
    total_electrons: float,
    degeneracy_tolerance_ha: float = 1.0e-10,
    electron_tolerance: float = 1.0e-12,
) -> dict[str, object]:
    n_spins, n_kpoints, n_bands = eigenvalues.shape
    capacity_factor = 2.0 / n_spins
    states: list[tuple[float, float, int, int, int]] = []
    for spin in range(n_spins):
        for kpoint in range(n_kpoints):
            capacity = capacity_factor * float(kpoint_weights[kpoint])
            for band in range(n_bands):
                states.append(
                    (float(eigenvalues[spin, kpoint, band]), capacity,
                     spin, kpoint, band)
                )
    states.sort(key=lambda state: (state[0], state[2], state[3], state[4]))

    occupations = np.zeros_like(eigenvalues)
    remaining = total_electrons
    partially_occupied: list[dict[str, object]] = []
    begin = 0
    while begin < len(states):
        end = begin + 1
        while (
            end < len(states)
            and abs(states[end][0] - states[begin][0]) <= degeneracy_tolerance_ha
        ):
            end += 1
        group_capacity = sum(state[1] for state in states[begin:end])
        fraction = (
            min(max(remaining / group_capacity, 0.0), 1.0)
            if group_capacity > 0.0
            else 0.0
        )
        for energy, capacity, spin, kpoint, band in states[begin:end]:
            occupations[spin, kpoint, band] = fraction * capacity
        if electron_tolerance < fraction < 1.0 - electron_tolerance:
            partially_occupied.append(
                {
                    "energy_ha": states[begin][0],
                    "energy_ev": states[begin][0] * HA2EV,
                    "fraction": fraction,
                    "group_capacity": group_capacity,
                    "states": [
                        {
                            "spin": state[2],
                            "kpoint": state[3],
                            "band": state[4],
                            "capacity": state[1],
                        }
                        for state in states[begin:end]
                    ],
                }
            )
        remaining -= fraction * group_capacity
        if remaining < electron_tolerance:
            remaining = 0.0
        begin = end
    if remaining > electron_tolerance:
        raise AuditError("global filling did not reach the target charge")

    occupied = []
    unoccupied = []
    for energy, capacity, spin, kpoint, band in states:
        occupation = float(occupations[spin, kpoint, band])
        if capacity <= electron_tolerance:
            continue
        if occupation > electron_tolerance:
            occupied.append(energy)
        if occupation < capacity - electron_tolerance:
            unoccupied.append(energy)
    if not occupied or not unoccupied:
        raise AuditError("occupation frontier is not bracketed")
    vbm = max(occupied)
    cbm = min(unoccupied)
    gap = max(0.0, cbm - vbm)
    electron_count = float(np.sum(occupations))
    return {
        "electron_count": electron_count,
        "electron_count_error": electron_count - total_electrons,
        "vbm_ha": vbm,
        "cbm_ha": cbm,
        "gap_ha": gap,
        "gap_ev": gap * HA2EV,
        "metallic": bool(partially_occupied or gap <= degeneracy_tolerance_ha),
        "partial_groups": partially_occupied,
        "min_occupied_to_unoccupied_ha": max(0.0, cbm - vbm),
    }


def audit(
    band_out: Path,
    bz_sampling: Path,
    checkpoint_root: Path,
    iteration: int,
    betas: list[float],
) -> dict[str, object]:
    reference_eigenvalues, reference_occupations, reference_efermi = (
        parse_band_out(band_out)
    )
    n_spins, n_kpoints, n_bands = reference_eigenvalues.shape
    kpoint_weights = parse_bz_weights(bz_sampling, n_kpoints)
    total_electrons = float(np.sum(reference_occupations))
    raw_hamiltonian = read_legacy_hamiltonian(
        checkpoint_root, iteration, n_spins, n_kpoints, n_bands
    )
    reference_hamiltonian = np.zeros_like(raw_hamiltonian)
    diagonal = np.arange(n_bands)
    reference_hamiltonian[..., diagonal, diagonal] = reference_eigenvalues

    reports: dict[str, object] = {}
    for beta in betas:
        if not math.isfinite(beta) or not 0.0 <= beta <= 1.0:
            raise AuditError("mixing beta must be in [0, 1]")
        mixed = (1.0 - beta) * reference_hamiltonian + beta * raw_hamiltonian
        eigenvalues = np.linalg.eigvalsh(mixed, UPLO="U")
        report = weighted_fill(eigenvalues, kpoint_weights, total_electrons)
        report.update(
            {
                "beta": beta,
                "fixed_four_band_gap_ev": (
                    float(np.min(eigenvalues[..., 4]))
                    - float(np.max(eigenvalues[..., 3]))
                )
                * HA2EV,
                "hermiticity_max_abs_ha": float(
                    np.max(np.abs(mixed - np.swapaxes(mixed.conj(), -1, -2)))
                ),
            }
        )
        reports[format(beta, ".17g")] = report

    return {
        "schema": SCHEMA,
        "inputs": {
            "band_out": str(band_out.resolve(strict=True)),
            "band_out_sha256": sha256_file(band_out),
            "bz_sampling_out": str(bz_sampling.resolve(strict=True)),
            "bz_sampling_out_sha256": sha256_file(bz_sampling),
            "checkpoint_root": str(checkpoint_root.resolve(strict=True)),
            "iteration": iteration,
        },
        "dimensions": {
            "n_spins": n_spins,
            "n_kpoints": n_kpoints,
            "n_bands": n_bands,
        },
        "reference_efermi_ha": reference_efermi,
        "target_electrons": total_electrons,
        "kpoint_weights": kpoint_weights.tolist(),
        "reports": reports,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("band_out", type=Path)
    parser.add_argument("bz_sampling_out", type=Path)
    parser.add_argument("checkpoint_root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--iteration", type=int, default=1)
    parser.add_argument("--betas", default="0,0.2,1")
    args = parser.parse_args()
    betas = [float(value) for value in args.betas.split(",")]
    result = audit(
        args.band_out,
        args.bz_sampling_out,
        args.checkpoint_root,
        args.iteration,
        betas,
    )
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
