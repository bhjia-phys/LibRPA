#!/usr/bin/env python3
"""Independent recomputation of the LibRPA QSGW Hartree delta-V pipeline (v1).

This tool re-derives one QSGW Hartree density-increment update in pure
Python/NumPy, without using any C++ reader or kernel, and compares against
the env-gated C++ dump (schema ``qsgw_hartree_pipeline_dump_v1``) and the
frozen QSGW matrix trace. Every convention below was extracted from the C++
sources (file:line given per section):

- contraction        src/qsgw/hartree_kernel.cpp:156-290
- density split      src/qsgw/hartree_workflow.cpp:402-457
- c_k assembly       src/qsgw/hartree_workflow.cpp:208-298 (phase sign +1.0,
                     row transpose to (aux, orbital))
- v_q0 assembly      src/qsgw/hartree_workflow.cpp:300-400 (q=0 selection,
                     Hermitian completion, diagonal 0.5*(V+V^dag))
- Fourier phase      src/qsgw/hartree_workflow.cpp:23-32 (exp(i*sign*2*pi*k.R))
- inverse Fourier    src/qsgw/hartree_workflow.cpp:459-545 (1/N_k, sign -1.0,
                     BvK remap scatter with 1/|remapped| weight)
- projection         src/qsgw/hartree_density.cpp:511-601 (R->k with sign
                     +1.0, plain W^dag O W with NO overlap matrix, then
                     Hermitian projection 0.5*(P+P^dag))
- Cs text format     driver/reader_lri.cpp:794-868
- Coulomb text       driver/reader_coulomb.cpp:71-174,1006-1062 (upper
                     triangle only; q key is pbc.klist[iq] per
                     src/api/input.cpp:938 with klist=klist_full staged in
                     driver/tasks/qsgw.cpp:861)
- bz_sampling_out    driver/read_data.cpp:1535-1658
- band_out           driver/read_data.cpp:411-474 (occupation / n_kpoints)
- KS_eigenvector     driver/reader_eigenvec.cpp:111-173 (legacy text order
                     BasisSpinorBandSpin: iw -> isoc -> ib -> is)
- trace rows         src/qsgw/iteration_trace.cpp:89-124; occupation is a
                     1 x n_bands matrix (row=0, column=band, value already
                     carries the band_out 1/n_kpoints factor) :349-376;
                     channel Grid=0 (iteration_trace.h:17-22)
- R grid             src/core/pbc.cpp:540-552
- full-BZ k order    src/core/pbc.cpp:212-238 (kfrac_list_full = kfrac_list)
- dump schema        src/qsgw/hartree_dump.cpp:69-203

Limitations (also reported in the JSON notes):
- BvK remap is assumed identity (no remap table is read); the dump hartree_r
  is post-remap, so a non-identity C++ remap shows up as a check-B/D failure.
- Only n_spins == 1 and n_spinor == 1 mean fields are supported.
- Only legacy *text* Cs / Coulomb / KS_eigenvector files are read (binary
  reader-v1 formats are rejected fail-closed).
"""

from __future__ import annotations

import argparse
import cmath
import json
import math
import sys
from pathlib import Path

import numpy as np

SCHEMA = "qsgw_hartree_pipeline_dump_v1"
NORMALIZATION_LEGACY = "legacy_extra_inverse_nk"
NORMALIZATION_WEIGHTED = "weighted_occupations"
NORMALIZATIONS = (NORMALIZATION_LEGACY, NORMALIZATION_WEIGHTED)
GRID_CHANNEL = 0  # IterationChannel::Grid (src/qsgw/iteration_trace.h:19)
TWO_PI = 2.0 * math.pi
GAMMA_TOLERANCE = 1.0e-10  # is_periodic_zero tolerance, hartree_workflow.h:46
BZ_WEIGHT_SUM_TOL = 1.0e-6  # kBzSamplingWeightSumTol, driver/read_data.cpp:67


class RecomputeError(Exception):
    """File/format/convention violation; maps to CLI exit code 1."""


# ---------------------------------------------------------------------------
# Phase and grids (src/qsgw/hartree_workflow.cpp:23-32, src/core/pbc.cpp:540-552)
# ---------------------------------------------------------------------------


def fourier_phase(
    kpoint: tuple[float, float, float],
    translation: tuple[int, int, int],
    sign: float,
) -> complex:
    angle = sign * TWO_PI * (
        kpoint[0] * translation[0]
        + kpoint[1] * translation[1]
        + kpoint[2] * translation[2]
    )
    return cmath.exp(1.0j * angle)


def construct_r_grid(period: tuple[int, int, int]) -> list[tuple[int, int, int]]:
    if len(period) != 3 or any(p <= 0 for p in period):
        raise RecomputeError(f"invalid BvK period: {period}")
    grid = []
    for x in range(-(period[0] // 2), (period[0] - 1) // 2 + 1):
        for y in range(-(period[1] // 2), (period[1] - 1) // 2 + 1):
            for z in range(-(period[2] // 2), (period[2] - 1) // 2 + 1):
                grid.append((x, y, z))
    return grid


# ---------------------------------------------------------------------------
# Small parsing helpers (fail-closed)
# ---------------------------------------------------------------------------


def _tokens(path: Path) -> list[str]:
    if not path.is_file():
        raise RecomputeError(f"missing file: {path}")
    try:
        text = path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError) as exc:
        raise RecomputeError(f"cannot read {path}: {exc}") from exc
    return text.split()


def _parse_int(token: str, label: str) -> int:
    try:
        return int(token)
    except ValueError:
        raise RecomputeError(f"invalid integer {token!r} in {label}") from None


def _parse_float(token: str, label: str) -> float:
    try:
        value = float(token)
    except ValueError:
        raise RecomputeError(f"invalid float {token!r} in {label}") from None
    if not math.isfinite(value):
        raise RecomputeError(f"non-finite value {token!r} in {label}")
    return value


def _read_table(
    path: Path,
    n_columns: int,
    expected_header: tuple[str, ...],
    label: str,
) -> list[list[str]]:
    """Read a dump table: one '#' header line, then fixed-width data rows."""
    if not path.is_file():
        raise RecomputeError(f"missing file: {path}")
    rows: list[list[str]] = []
    header_seen = False
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("#"):
                tokens = stripped[1:].split()
                if not header_seen:
                    header_seen = True
                    if tuple(tokens) != expected_header:
                        raise RecomputeError(
                            f"{label}: unexpected header {tokens!r} in {path}, "
                            f"expected {list(expected_header)!r}"
                        )
                continue
            tokens = stripped.split()
            if len(tokens) != n_columns:
                raise RecomputeError(
                    f"{label}: row {line_number} in {path} has {len(tokens)} "
                    f"columns, expected {n_columns}"
                )
            rows.append(tokens)
    if not header_seen:
        raise RecomputeError(f"{label}: missing header line in {path}")
    return rows


# ---------------------------------------------------------------------------
# Dump manifest + tables (src/qsgw/hartree_dump.cpp:69-203)
# ---------------------------------------------------------------------------


def load_manifest(call_dir: Path) -> dict:
    path = call_dir / "manifest.txt"
    if not path.is_file():
        raise RecomputeError(f"missing dump manifest: {path}")
    entries: dict[str, str] = {}
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        if "=" not in line:
            raise RecomputeError(
                f"manifest line {line_number} is not key=value: {line!r}"
            )
        key, value = line.split("=", 1)
        if key in entries:
            raise RecomputeError(f"manifest repeats key {key!r}")
        entries[key] = value

    schema = entries.get("schema")
    if schema != SCHEMA:
        raise RecomputeError(
            f"manifest schema {schema!r} does not match {SCHEMA!r}"
        )
    normalization = entries.get("normalization")
    if normalization not in NORMALIZATIONS:
        raise RecomputeError(
            f"manifest normalization {normalization!r} is not one of "
            f"{list(NORMALIZATIONS)!r}"
        )

    def required_int(key: str) -> int:
        if key not in entries:
            raise RecomputeError(f"manifest misses key {key!r}")
        return _parse_int(entries[key], f"manifest {key}")

    kpoint_count = required_int("kpoint_count")
    translation_count = required_int("translation_count")
    density_count = required_int("density_delta_k_count")
    hartree_k_atoms = required_int("hartree_k_atom_count")
    hartree_r_atoms = required_int("hartree_r_atom_count")
    if kpoint_count <= 0 or translation_count <= 0:
        raise RecomputeError("manifest grid counts must be positive")

    period_tokens = entries.get("period", "").split()
    if len(period_tokens) != 3:
        raise RecomputeError("manifest period must have three components")
    period = tuple(_parse_int(t, "manifest period") for t in period_tokens)
    translations = construct_r_grid(period)
    if len(translations) != translation_count:
        raise RecomputeError(
            f"manifest translation_count={translation_count} does not match "
            f"the BvK grid of period {period} ({len(translations)})"
        )

    ao_tokens = entries.get("atom_ao_sizes", "").split()
    if not ao_tokens:
        raise RecomputeError("manifest atom_ao_sizes is empty")
    atom_ao_sizes: dict[int, int] = {}
    for token in ao_tokens:
        if ":" not in token:
            raise RecomputeError(f"manifest atom_ao_sizes token {token!r}")
        atom_s, size_s = token.split(":", 1)
        atom = _parse_int(atom_s, "manifest atom_ao_sizes atom")
        size = _parse_int(size_s, "manifest atom_ao_sizes size")
        if atom in atom_ao_sizes:
            raise RecomputeError(f"manifest atom_ao_sizes repeats atom {atom}")
        if size <= 0:
            raise RecomputeError("manifest atom AO sizes must be positive")
        atom_ao_sizes[atom] = size
    if sorted(atom_ao_sizes) != list(range(len(atom_ao_sizes))):
        raise RecomputeError(
            "manifest atom_ao_sizes keys must be contiguous 0..n-1"
        )

    if density_count != kpoint_count:
        raise RecomputeError(
            "manifest density_delta_k_count does not match kpoint_count"
        )
    if hartree_k_atoms != len(atom_ao_sizes) or (
        hartree_r_atoms != len(atom_ao_sizes)
    ):
        raise RecomputeError(
            "manifest hartree atom counts do not match atom_ao_sizes"
        )

    expected_files = {
        "density_delta_k_file": "density_delta_k.txt",
        "density_delta_k_columns": "kpoint row column real imag",
        "hartree_k_file": "hartree_k.txt",
        "hartree_k_columns": "atom_i atom_j kpoint row column real imag",
        "hartree_r_file": "hartree_r.txt",
        "hartree_r_columns": "atom_i atom_j R_x R_y R_z row column real imag",
    }
    for key, expected in expected_files.items():
        if key in entries and entries[key] != expected:
            raise RecomputeError(
                f"manifest {key}={entries[key]!r}, expected {expected!r}"
            )

    return {
        "normalization": normalization,
        "kpoint_count": kpoint_count,
        "translation_count": translation_count,
        "period": period,
        "atom_ao_sizes": atom_ao_sizes,
    }


def load_density_delta_k(
    call_dir: Path, kpoint_count: int, n_aos: int
) -> np.ndarray:
    rows = _read_table(
        call_dir / "density_delta_k.txt",
        5,
        ("kpoint", "row", "column", "real", "imag"),
        "density_delta_k",
    )
    result = np.zeros((kpoint_count, n_aos, n_aos), dtype=np.complex128)
    filled = np.zeros((kpoint_count, n_aos, n_aos), dtype=bool)
    for tokens in rows:
        kpoint = _parse_int(tokens[0], "density_delta_k kpoint")
        row = _parse_int(tokens[1], "density_delta_k row")
        column = _parse_int(tokens[2], "density_delta_k column")
        if not (0 <= kpoint < kpoint_count) or not (0 <= row < n_aos) or not (
            0 <= column < n_aos
        ):
            raise RecomputeError(
                f"density_delta_k index out of range: {tokens[:3]}"
            )
        if filled[kpoint, row, column]:
            raise RecomputeError(
                f"density_delta_k repeats entry {(kpoint, row, column)}"
            )
        filled[kpoint, row, column] = True
        result[kpoint, row, column] = complex(
            _parse_float(tokens[3], "density_delta_k real"),
            _parse_float(tokens[4], "density_delta_k imag"),
        )
    if not bool(filled.all()):
        raise RecomputeError("density_delta_k has incomplete k/row/column coverage")
    return result


def load_hartree_k(
    call_dir: Path, atom_ao_sizes: dict[int, int], kpoint_count: int
) -> dict[tuple[int, int], np.ndarray]:
    rows = _read_table(
        call_dir / "hartree_k.txt",
        7,
        ("atom_i", "atom_j", "kpoint", "row", "column", "real", "imag"),
        "hartree_k",
    )
    n_atoms = len(atom_ao_sizes)
    result = {
        (i, j): np.zeros(
            (kpoint_count, atom_ao_sizes[i], atom_ao_sizes[j]),
            dtype=np.complex128,
        )
        for i in range(n_atoms)
        for j in range(n_atoms)
    }
    filled = {pair: np.zeros(block.shape, dtype=bool) for pair, block in result.items()}
    for tokens in rows:
        atom_i = _parse_int(tokens[0], "hartree_k atom_i")
        atom_j = _parse_int(tokens[1], "hartree_k atom_j")
        kpoint = _parse_int(tokens[2], "hartree_k kpoint")
        row = _parse_int(tokens[3], "hartree_k row")
        column = _parse_int(tokens[4], "hartree_k column")
        pair = (atom_i, atom_j)
        if pair not in result or not (0 <= kpoint < kpoint_count):
            raise RecomputeError(f"hartree_k index out of range: {tokens[:3]}")
        block = result[pair]
        if not (0 <= row < block.shape[1]) or not (0 <= column < block.shape[2]):
            raise RecomputeError(f"hartree_k block index out of range: {tokens}")
        if filled[pair][kpoint, row, column]:
            raise RecomputeError(
                f"hartree_k repeats entry {(atom_i, atom_j, kpoint, row, column)}"
            )
        filled[pair][kpoint, row, column] = True
        block[kpoint, row, column] = complex(
            _parse_float(tokens[5], "hartree_k real"),
            _parse_float(tokens[6], "hartree_k imag"),
        )
    for pair, mask in filled.items():
        if not bool(mask.all()):
            raise RecomputeError(f"hartree_k has incomplete coverage for {pair}")
    return result


def load_hartree_r(
    call_dir: Path, atom_ao_sizes: dict[int, int]
) -> dict[tuple[int, int, tuple[int, int, int]], np.ndarray]:
    rows = _read_table(
        call_dir / "hartree_r.txt",
        9,
        (
            "atom_i",
            "atom_j",
            "R_x",
            "R_y",
            "R_z",
            "row",
            "column",
            "real",
            "imag",
        ),
        "hartree_r",
    )
    n_atoms = len(atom_ao_sizes)
    blocks: dict[tuple[int, int, tuple[int, int, int]], np.ndarray] = {}
    filled: dict[tuple[int, int, tuple[int, int, int]], np.ndarray] = {}
    for tokens in rows:
        atom_i = _parse_int(tokens[0], "hartree_r atom_i")
        atom_j = _parse_int(tokens[1], "hartree_r atom_j")
        translation = (
            _parse_int(tokens[2], "hartree_r R_x"),
            _parse_int(tokens[3], "hartree_r R_y"),
            _parse_int(tokens[4], "hartree_r R_z"),
        )
        row = _parse_int(tokens[5], "hartree_r row")
        column = _parse_int(tokens[6], "hartree_r column")
        if not (0 <= atom_i < n_atoms) or not (0 <= atom_j < n_atoms):
            raise RecomputeError(f"hartree_r atom index out of range: {tokens}")
        key = (atom_i, atom_j, translation)
        if key not in blocks:
            blocks[key] = np.zeros(
                (atom_ao_sizes[atom_i], atom_ao_sizes[atom_j]),
                dtype=np.complex128,
            )
            filled[key] = np.zeros(blocks[key].shape, dtype=bool)
        block = blocks[key]
        if not (0 <= row < block.shape[0]) or not (0 <= column < block.shape[1]):
            raise RecomputeError(f"hartree_r block index out of range: {tokens}")
        if filled[key][row, column]:
            raise RecomputeError(f"hartree_r repeats entry {key + (row, column)}")
        filled[key][row, column] = True
        block[row, column] = complex(
            _parse_float(tokens[7], "hartree_r real"),
            _parse_float(tokens[8], "hartree_r imag"),
        )
    for key, mask in filled.items():
        if not bool(mask.all()):
            raise RecomputeError(f"hartree_r has incomplete block {key}")
    return blocks


# ---------------------------------------------------------------------------
# Frozen-input readers
# ---------------------------------------------------------------------------


def read_bz_sampling(path: Path) -> dict:
    """driver/read_data.cpp:1535-1658 (read_bz_sampling)."""
    tokens = _tokens(path)
    cursor = 0

    def take(label: str) -> str:
        nonlocal cursor
        if cursor >= len(tokens):
            raise RecomputeError(f"unexpected end of {path} at {label}")
        token = tokens[cursor]
        cursor += 1
        return token

    period = tuple(_parse_int(take("nk"), "bz_sampling nk") for _ in range(3))
    if any(p <= 0 for p in period):
        raise RecomputeError("bz_sampling k-grid must be positive")
    n_kpoints = _parse_int(take("n_kpoints_scf"), "bz_sampling n_kpoints_scf")
    n_ibz = _parse_int(take("nk_ibz"), "bz_sampling nk_ibz")
    if n_kpoints <= 0 or n_ibz <= 0:
        raise RecomputeError("bz_sampling k-point counts must be positive")
    if n_kpoints > period[0] * period[1] * period[2]:
        raise RecomputeError("bz_sampling SCF count exceeds the full grid")

    kfracs: list[tuple[float, float, float]] = []
    kvecs: list[tuple[float, float, float]] = []
    weights: list[float] = []
    representatives: list[int] = []
    for row in range(n_kpoints):
        index = _parse_int(take(f"k-point row {row + 1} index"), "bz_sampling ik")
        if index != row + 1:
            raise RecomputeError(
                "bz_sampling k-point index does not match row order"
            )
        weight = _parse_float(take("weight"), "bz_sampling weight")
        if weight < 0.0:
            raise RecomputeError("bz_sampling weight must be nonnegative")
        kfrac = tuple(
            _parse_float(take("kfrac"), "bz_sampling kfrac") for _ in range(3)
        )
        kvec = tuple(
            _parse_float(take("kvec"), "bz_sampling kvec") for _ in range(3)
        )
        ik_ibz = _parse_int(take("ik_ibz"), "bz_sampling ik_ibz")
        ik_rep = _parse_int(take("ik_rep"), "bz_sampling ik_rep")
        if not (1 <= ik_ibz <= n_ibz) or not (1 <= ik_rep <= n_kpoints):
            raise RecomputeError("bz_sampling IBZ/representative index out of range")
        weights.append(weight)
        kfracs.append(kfrac)
        kvecs.append(kvec)
        representatives.append(ik_rep - 1)
    if abs(sum(weights) - 1.0) > BZ_WEIGHT_SUM_TOL:
        raise RecomputeError("bz_sampling k-point weights do not sum to 1")

    gamma_candidates = [
        index
        for index, kfrac in enumerate(kfracs)
        if all(abs(component - round(component)) <= GAMMA_TOLERANCE for component in kfrac)
    ]
    if len(gamma_candidates) != 1:
        raise RecomputeError(
            f"bz_sampling must contain exactly one periodic-zero (Gamma) "
            f"k-point, found {len(gamma_candidates)}"
        )
    return {
        "period": period,
        "kfracs": kfracs,
        "kvecs": kvecs,
        "weights": weights,
        "representatives": representatives,
        "gamma_index": gamma_candidates[0],
    }


def read_band_out(path: Path) -> dict:
    """driver/read_data.cpp:411-474 (read_scf_occ_eigenvalues)."""
    tokens = _tokens(path)
    cursor = 0

    def take(label: str) -> str:
        nonlocal cursor
        if cursor >= len(tokens):
            raise RecomputeError(f"unexpected end of {path} at {label}")
        token = tokens[cursor]
        cursor += 1
        return token

    n_kpoints = _parse_int(take("n_kpoints"), "band_out n_kpoints")
    n_spins = _parse_int(take("n_spins"), "band_out n_spins")
    n_states = _parse_int(take("n_states"), "band_out n_states")
    n_basis_wfc = _parse_int(take("n_basis_wfc"), "band_out n_basis_wfc")
    efermi = _parse_float(take("efermi"), "band_out efermi")
    if min(n_kpoints, n_spins, n_states, n_basis_wfc) <= 0:
        raise RecomputeError("band_out dimensions must be positive")
    seen: set[tuple[int, int]] = set()
    for _ik in range(n_kpoints):
        for spin in range(n_spins):
            k_index = _parse_int(take("k index"), "band_out k index") - 1
            s_index = _parse_int(take("spin index"), "band_out spin index") - 1
            if not (0 <= k_index < n_kpoints) or s_index != spin:
                raise RecomputeError("band_out k/spin index is out of order")
            if (k_index, spin) in seen:
                raise RecomputeError("band_out repeats a k/spin block")
            seen.add((k_index, spin))
            for _band in range(n_states):
                take("band index")
                _parse_float(take("occupation"), "band_out occupation")
                _parse_float(take("eigenvalue Ha"), "band_out eigenvalue")
                take("eigenvalue eV")
    return {
        "n_kpoints": n_kpoints,
        "n_spins": n_spins,
        "n_states": n_states,
        "n_basis_wfc": n_basis_wfc,
        "efermi": efermi,
    }


def read_cs_text(
    input_dir: Path, prefix: str = "Cs_data_"
) -> dict[tuple[int, int], dict[tuple[int, int, int], np.ndarray]]:
    """driver/reader_lri.cpp:794-868 (handle_Cs_file, legacy text)."""
    if not input_dir.is_dir():
        raise RecomputeError(f"missing input directory: {input_dir}")
    files = sorted(p for p in input_dir.iterdir() if p.name.startswith(prefix))
    if not files:
        raise RecomputeError(f"no {prefix}* files found in {input_dir}")
    result: dict[tuple[int, int], dict[tuple[int, int, int], np.ndarray]] = {}
    for path in files:
        tokens = _tokens(path)
        try:
            natom = int(tokens[0])
            ncell = int(tokens[1])
        except (IndexError, ValueError):
            raise RecomputeError(
                f"{path}: not a legacy text Cs file (binary formats are "
                f"unsupported)"
            ) from None
        if natom <= 0 or ncell < 0:
            raise RecomputeError(f"{path}: invalid Cs header")
        cursor = 2
        while cursor < len(tokens):
            if cursor + 8 > len(tokens):
                raise RecomputeError(f"{path}: truncated Cs block header")
            header = tokens[cursor : cursor + 8]
            cursor += 8
            atom_i = _parse_int(header[0], f"{path} ia1") - 1
            atom_j = _parse_int(header[1], f"{path} ia2") - 1
            translation = (
                _parse_int(header[2], f"{path} R_x"),
                _parse_int(header[3], f"{path} R_y"),
                _parse_int(header[4], f"{path} R_z"),
            )
            n_i = _parse_int(header[5], f"{path} n_i")
            n_j = _parse_int(header[6], f"{path} n_j")
            n_mu = _parse_int(header[7], f"{path} n_mu")
            if min(atom_i, atom_j, n_i, n_j, n_mu) < 0 or min(n_i, n_j, n_mu) <= 0:
                raise RecomputeError(f"{path}: invalid Cs block header")
            count = n_i * n_j * n_mu
            if cursor + count > len(tokens):
                raise RecomputeError(f"{path}: truncated Cs block payload")
            values = np.array(
                [
                    _parse_float(token, f"{path} Cs value")
                    for token in tokens[cursor : cursor + count]
                ]
            ).reshape(n_i * n_j, n_mu)
            cursor += count
            pair_blocks = result.setdefault((atom_i, atom_j), {})
            if translation in pair_blocks:
                raise RecomputeError(
                    f"{path}: duplicate Cs block {(atom_i, atom_j, translation)}"
                )
            pair_blocks[translation] = values
    return result


def infer_atom_aux_sizes(
    cs: dict[tuple[int, int], dict[tuple[int, int, int], np.ndarray]],
    atom_ao_sizes: dict[int, int],
) -> dict[int, int]:
    aux_sizes: dict[int, int] = {}
    for (atom_i, atom_j), blocks in cs.items():
        for translation, block in blocks.items():
            del translation
            if block.shape[0] != atom_ao_sizes.get(atom_i, -1) * atom_ao_sizes.get(
                atom_j, -1
            ):
                raise RecomputeError(
                    f"Cs block {(atom_i, atom_j)} row count {block.shape[0]} "
                    f"does not match ao sizes"
                )
            previous = aux_sizes.setdefault(atom_i, block.shape[1])
            if previous != block.shape[1]:
                raise RecomputeError(
                    f"Cs aux size for atom {atom_i} is inconsistent"
                )
    missing = [atom for atom in atom_ao_sizes if atom not in aux_sizes]
    if missing:
        raise RecomputeError(
            f"Cs data does not determine aux sizes for atoms {missing}"
        )
    return aux_sizes


def build_c_k(
    cs: dict[tuple[int, int], dict[tuple[int, int, int], np.ndarray]],
    atom_ao_sizes: dict[int, int],
    atom_aux_sizes: dict[int, int],
    kfracs: list[tuple[float, float, float]],
) -> dict[tuple[int, int], np.ndarray]:
    """src/qsgw/hartree_workflow.cpp:208-298 (build_hartree_c_k).

    c_k[i][j][k](aux, orb) = sum_R exp(+2 pi i k.R) Cs_R(orb, aux); pairs
    missing from the Cs data stay zero exactly like the pre-initialized C++ map.
    """
    n_k = len(kfracs)
    n_atoms = len(atom_ao_sizes)
    result = {
        (i, j): np.zeros(
            (n_k, atom_aux_sizes[i], atom_ao_sizes[i] * atom_ao_sizes[j]),
            dtype=np.complex128,
        )
        for i in range(n_atoms)
        for j in range(n_atoms)
    }
    for (atom_i, atom_j), blocks in cs.items():
        target = result[(atom_i, atom_j)]
        for translation, block in blocks.items():
            phases = np.array(
                [fourier_phase(kfrac, translation, 1.0) for kfrac in kfracs]
            )
            # block is (orbital, aux); c_k stores (aux, orbital)
            target += phases[:, None, None] * block.T[None, :, :]
    return result


def read_coulomb_gamma_full(
    input_dir: Path, gamma_q_num: int, prefix: str = "coulomb_cut_"
) -> tuple[np.ndarray, int]:
    """driver/reader_coulomb.cpp:71-174 (handle_Vq_full_file, legacy text).

    Only blocks whose 1-based q_num equals gamma_q_num are merged; the q
    vector of q_num is pbc.klist[q_num - 1] (src/api/input.cpp:938) with
    klist := klist_full staged by the Hartree reader
    (driver/tasks/qsgw.cpp:861), so gamma_q_num = gamma bz row index + 1.
    Returns (full_matrix, zero_filled_entry_count).
    """
    if not input_dir.is_dir():
        raise RecomputeError(f"missing input directory: {input_dir}")
    files = sorted(
        p
        for p in input_dir.iterdir()
        if p.name.startswith(prefix) and p.name.endswith(".txt")
    )
    if not files:
        raise RecomputeError(f"no {prefix}*.txt files found in {input_dir}")
    full: np.ndarray | None = None
    filled: np.ndarray | None = None
    for path in files:
        tokens = _tokens(path)
        try:
            int(tokens[0])
        except (IndexError, ValueError):
            raise RecomputeError(
                f"{path}: not a legacy text Coulomb file"
            ) from None
        cursor = 1
        while cursor < len(tokens):
            if cursor + 7 > len(tokens):
                raise RecomputeError(f"{path}: truncated Coulomb block header")
            header = tokens[cursor : cursor + 7]
            cursor += 7
            nbasbas = _parse_int(header[0], f"{path} nbasbas")
            brow = _parse_int(header[1], f"{path} begin_row") - 1
            erow = _parse_int(header[2], f"{path} end_row") - 1
            bcol = _parse_int(header[3], f"{path} begin_col") - 1
            ecol = _parse_int(header[4], f"{path} end_col") - 1
            q_num = _parse_int(header[5], f"{path} q_num")
            _parse_float(header[6], f"{path} q_weight")
            if nbasbas <= 0 or erow < brow or ecol < bcol or q_num < 1:
                raise RecomputeError(f"{path}: invalid Coulomb block header")
            nrow = erow - brow + 1
            ncol = ecol - bcol + 1
            count = nrow * ncol * 2
            if cursor + count > len(tokens):
                raise RecomputeError(f"{path}: truncated Coulomb block payload")
            payload = tokens[cursor : cursor + count]
            cursor += count
            if q_num != gamma_q_num:
                continue
            if full is None:
                full = np.zeros((nbasbas, nbasbas), dtype=np.complex128)
                filled = np.zeros((nbasbas, nbasbas), dtype=bool)
            if full.shape != (nbasbas, nbasbas):
                raise RecomputeError(
                    f"{path}: inconsistent Coulomb nbasbas {nbasbas}"
                )
            for local_row in range(nrow):
                for local_col in range(ncol):
                    re = _parse_float(
                        payload[2 * (local_row * ncol + local_col)],
                        f"{path} Vq real",
                    )
                    im = _parse_float(
                        payload[2 * (local_row * ncol + local_col) + 1],
                        f"{path} Vq imag",
                    )
                    grow = brow + local_row
                    gcol = bcol + local_col
                    if not (0 <= grow < nbasbas) or not (0 <= gcol < nbasbas):
                        raise RecomputeError(
                            f"{path}: Coulomb block range exceeds nbasbas"
                        )
                    if filled[grow, gcol]:
                        raise RecomputeError(
                            f"{path}: overlapping Coulomb Gamma blocks"
                        )
                    filled[grow, gcol] = True
                    full[grow, gcol] = complex(re, im)
    if full is None or filled is None:
        raise RecomputeError(
            f"no Coulomb block with q_num={gamma_q_num} (Gamma) found"
        )
    zero_filled = int((~filled).sum())
    return full, zero_filled


def build_v_q0(
    v_full: np.ndarray, atom_aux_sizes: dict[int, int]
) -> dict[tuple[int, int], np.ndarray]:
    """src/qsgw/hartree_workflow.cpp:300-400 (build_hartree_v_q0).

    Legacy full reader stores only upper-triangle atom pairs, so off-diagonal
    blocks pass through, diagonal blocks are Hermitized 0.5*(V+V^dag), and
    the lower triangle is the conjugate transpose of the upper one.
    """
    if v_full.shape[0] != v_full.shape[1]:
        raise RecomputeError("Coulomb Gamma matrix is not square")
    n_atoms = len(atom_aux_sizes)
    offsets: dict[int, int] = {}
    total = 0
    for atom in range(n_atoms):
        offsets[atom] = total
        total += atom_aux_sizes[atom]
    if total != v_full.shape[0]:
        raise RecomputeError(
            f"Coulomb Gamma dimension {v_full.shape[0]} does not match the "
            f"aux basis total {total}"
        )
    result: dict[tuple[int, int], np.ndarray] = {}
    for atom_i in range(n_atoms):
        for atom_j in range(atom_i, n_atoms):
            block = v_full[
                offsets[atom_i] : offsets[atom_i] + atom_aux_sizes[atom_i],
                offsets[atom_j] : offsets[atom_j] + atom_aux_sizes[atom_j],
            ]
            if atom_i == atom_j:
                projected = 0.5 * (block + block.conj().T)
            else:
                projected = np.array(block, dtype=np.complex128)
            result[(atom_i, atom_j)] = projected
            result[(atom_j, atom_i)] = projected.conj().T.copy()
    return result


def read_ks_eigenvectors_text(
    input_dir: Path,
    n_spins: int,
    n_states: int,
    n_aos: int,
    n_kpoints: int,
    prefix: str = "KS_eigenvector",
) -> dict[int, dict[int, np.ndarray]]:
    """driver/reader_eigenvec.cpp:111-173 (legacy text, BasisSpinorBandSpin).

    Per k block: token k (1-based), then loops iw -> isoc -> ib -> is of
    're im'; stored as wfc[spin][k](band, ao). Only n_spinor == 1 here.
    """
    if not input_dir.is_dir():
        raise RecomputeError(f"missing input directory: {input_dir}")
    files = sorted(p for p in input_dir.iterdir() if p.name.startswith(prefix))
    if not files:
        raise RecomputeError(f"no {prefix}* files found in {input_dir}")
    wfc: dict[int, dict[int, np.ndarray]] = {
        spin: {} for spin in range(n_spins)
    }
    for path in files:
        tokens = _tokens(path)
        cursor = 0
        block_values = n_spins * n_states * n_aos
        while cursor < len(tokens):
            k_index = _parse_int(tokens[cursor], f"{path} k index") - 1
            cursor += 1
            if not (0 <= k_index < n_kpoints):
                raise RecomputeError(f"{path}: k index out of range")
            if cursor + 2 * block_values > len(tokens):
                raise RecomputeError(f"{path}: truncated eigenvector block")
            values = tokens[cursor : cursor + 2 * block_values]
            cursor += 2 * block_values
            for spin in range(n_spins):
                if k_index in wfc[spin]:
                    raise RecomputeError(
                        f"{path}: duplicate eigenvector block for k {k_index}"
                    )
                wfc[spin][k_index] = np.zeros((n_states, n_aos), dtype=np.complex128)
            position = 0
            for iw in range(n_aos):
                for _isoc in range(1):
                    for ib in range(n_states):
                        for spin in range(n_spins):
                            re = _parse_float(values[2 * position], f"{path} re")
                            im = _parse_float(
                                values[2 * position + 1], f"{path} im"
                            )
                            wfc[spin][k_index][ib, iw] = complex(re, im)
                            position += 1
    for spin in range(n_spins):
        if sorted(wfc[spin]) != list(range(n_kpoints)):
            raise RecomputeError(
                f"KS eigenvectors do not cover all k-points for spin {spin}"
            )
    return wfc


# ---------------------------------------------------------------------------
# Trace parsing (src/qsgw/iteration_trace.cpp:89-124)
# ---------------------------------------------------------------------------


def _iter_trace_rows(path: Path):
    if not path.is_file():
        raise RecomputeError(f"missing trace file: {path}")
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            tokens = stripped.split()
            if len(tokens) != 11:
                raise RecomputeError(
                    f"trace row {line_number} has {len(tokens)} columns, "
                    f"expected 11"
                )
            yield (
                _parse_int(tokens[0], "trace iter"),
                _parse_int(tokens[1], "trace channel"),
                tokens[2],
                _parse_int(tokens[3], "trace spin"),
                _parse_int(tokens[4], "trace kpoint"),
                _parse_int(tokens[5], "trace frequency_index"),
                _parse_float(tokens[6], "trace frequency_Ha"),
                _parse_int(tokens[7], "trace row"),
                _parse_int(tokens[8], "trace column"),
                _parse_float(tokens[9], "trace real"),
                _parse_float(tokens[10], "trace imag"),
            )


def trace_occupations(
    path: Path, iterations: set[int]
) -> dict[tuple[int, int, int], np.ndarray]:
    """Occupation rows are a 1 x n_bands matrix: row=0, column=band
    (src/qsgw/iteration_trace.cpp:365-371)."""
    entries: dict[tuple[int, int, int], dict[int, complex]] = {}
    for row in _iter_trace_rows(path):
        iteration, channel, component, spin, kpoint = row[:5]
        if component != "occupation" or channel != GRID_CHANNEL:
            continue
        if iteration not in iterations:
            continue
        m_row, m_col = row[7], row[8]
        if m_row != 0:
            raise RecomputeError(
                "occupation trace row index must be 0 (bands live in the "
                "column index per iteration_trace.cpp:365-371)"
            )
        key = (iteration, spin, kpoint)
        target = entries.setdefault(key, {})
        if m_col in target:
            raise RecomputeError(f"trace repeats occupation entry {key + (m_col,)}")
        target[m_col] = complex(row[9], row[10])
    result: dict[tuple[int, int, int], np.ndarray] = {}
    for key, bands in entries.items():
        if sorted(bands) != list(range(len(bands))):
            raise RecomputeError(f"occupation bands are not contiguous at {key}")
        result[key] = np.array([bands[i] for i in range(len(bands))])
    return result


def trace_delta_vh(
    path: Path, iterations: set[int]
) -> dict[tuple[int, int, int], np.ndarray]:
    entries: dict[tuple[int, int, int], dict[tuple[int, int], complex]] = {}
    for row in _iter_trace_rows(path):
        iteration, channel, component, spin, kpoint = row[:5]
        if component != "delta_vh" or channel != GRID_CHANNEL:
            continue
        if iteration not in iterations:
            continue
        key = (iteration, spin, kpoint)
        target = entries.setdefault(key, {})
        index = (row[7], row[8])
        if index in target:
            raise RecomputeError(f"trace repeats delta_vh entry {key + index}")
        target[index] = complex(row[9], row[10])
    result: dict[tuple[int, int, int], np.ndarray] = {}
    for key, cells in entries.items():
        rows = {index[0] for index in cells}
        columns = {index[1] for index in cells}
        if rows != set(range(len(rows))) or columns != set(range(len(columns))):
            raise RecomputeError(f"delta_vh indices are not contiguous at {key}")
        if len(rows) != len(columns):
            raise RecomputeError(f"delta_vh matrix is not square at {key}")
        size = len(rows)
        matrix = np.zeros((size, size), dtype=np.complex128)
        for (m_row, m_col), value in cells.items():
            matrix[m_row, m_col] = value
        result[key] = matrix
    return result


# ---------------------------------------------------------------------------
# Pipeline math (C++ convention mirrors)
# ---------------------------------------------------------------------------


def split_weighted_density_by_atom(
    density_k: np.ndarray, atom_ao_sizes: dict[int, int]
) -> dict[tuple[int, int], np.ndarray]:
    """src/qsgw/hartree_workflow.cpp:402-457."""
    n_atoms = len(atom_ao_sizes)
    offsets: dict[int, int] = {}
    total = 0
    for atom in range(n_atoms):
        if atom_ao_sizes[atom] <= 0:
            raise RecomputeError("atom AO sizes must be positive")
        offsets[atom] = total
        total += atom_ao_sizes[atom]
    if density_k.ndim != 3 or density_k.shape[1] != total or (
        density_k.shape[2] != total
    ):
        raise RecomputeError(
            f"density shape {density_k.shape} does not match AO total {total}"
        )
    return {
        (i, j): np.array(
            density_k[
                :,
                offsets[i] : offsets[i] + atom_ao_sizes[i],
                offsets[j] : offsets[j] + atom_ao_sizes[j],
            ]
        )
        for i in range(n_atoms)
        for j in range(n_atoms)
    }


def compute_density_aux(
    c_k: dict[tuple[int, int], np.ndarray],
    density_blocks: dict[tuple[int, int], np.ndarray],
    atom_ao_sizes: dict[int, int],
    atom_aux_sizes: dict[int, int],
) -> dict[int, np.ndarray]:
    """src/qsgw/hartree_kernel.cpp:167-214 (density_aux accumulation)."""
    density_aux = {
        atom: np.zeros(size, dtype=np.complex128)
        for atom, size in atom_aux_sizes.items()
    }
    for atom_v, ao_v in atom_ao_sizes.items():
        for atom_u, ao_u in atom_ao_sizes.items():
            density_vu = density_blocks[(atom_v, atom_u)]
            c_uv = c_k[(atom_u, atom_v)]
            c_vu = c_k[(atom_v, atom_u)]
            if density_vu.shape[1:] != (ao_v, ao_u):
                raise RecomputeError("density block shape mismatch")
            # orb index of c_uv is orb_u*ao_v + orb_v
            flat_uo = density_vu.transpose(0, 2, 1).reshape(
                density_vu.shape[0], ao_u * ao_v
            )
            density_aux[atom_u] += np.einsum("kab,kb->a", c_uv, flat_uo)
            # orb index of c_vu is orb_v*ao_u + orb_u
            flat_vo = density_vu.reshape(density_vu.shape[0], ao_v * ao_u)
            density_aux[atom_v] += np.einsum("kab,kb->a", c_vu.conj(), flat_vo)
    return density_aux


def compute_potential_aux(
    v_q0: dict[tuple[int, int], np.ndarray],
    density_aux: dict[int, np.ndarray],
    normalization: str,
    kpoint_count: int,
) -> dict[int, np.ndarray]:
    """src/qsgw/hartree_kernel.cpp:216-246 (Coulomb contraction + norm)."""
    potential_aux: dict[int, np.ndarray] = {}
    for atom_mu, aux_mu in ((atom, vec.shape[0]) for atom, vec in density_aux.items()):
        accumulator = np.zeros(aux_mu, dtype=np.complex128)
        for atom_nu, density_nu in density_aux.items():
            accumulator += v_q0[(atom_mu, atom_nu)] @ density_nu
        if normalization == NORMALIZATION_LEGACY:
            accumulator /= float(kpoint_count)
        elif normalization != NORMALIZATION_WEIGHTED:
            raise RecomputeError(
                f"unknown Hartree k-point normalization {normalization!r}"
            )
        potential_aux[atom_mu] = accumulator
    return potential_aux


def contract_hartree_full_grid(
    c_k: dict[tuple[int, int], np.ndarray],
    v_q0: dict[tuple[int, int], np.ndarray],
    density_blocks: dict[tuple[int, int], np.ndarray],
    atom_ao_sizes: dict[int, int],
    atom_aux_sizes: dict[int, int],
    normalization: str,
) -> dict[tuple[int, int], np.ndarray]:
    """src/qsgw/hartree_kernel.cpp:156-290 (contract_hartree_full_grid)."""
    kpoint_counts = {block.shape[0] for block in density_blocks.values()}
    if len(kpoint_counts) != 1:
        raise RecomputeError("density blocks disagree on the k-point count")
    kpoint_count = kpoint_counts.pop()
    density_aux = compute_density_aux(
        c_k, density_blocks, atom_ao_sizes, atom_aux_sizes
    )
    potential_aux = compute_potential_aux(
        v_q0, density_aux, normalization, kpoint_count
    )
    result: dict[tuple[int, int], np.ndarray] = {}
    for atom_s, ao_s in atom_ao_sizes.items():
        for atom_t, ao_t in atom_ao_sizes.items():
            c_st = c_k[(atom_s, atom_t)]
            c_ts = c_k[(atom_t, atom_s)]
            # orb index of c_st is orb_s*ao_t + orb_t (row-major (s, t))
            first = np.einsum("kab,a->kb", c_st, potential_aux[atom_s])
            # orb index of c_ts is orb_t*ao_s + orb_s; transpose to (s, t)
            second = np.einsum("kab,a->kb", c_ts.conj(), potential_aux[atom_t])
            result[(atom_s, atom_t)] = first.reshape(
                kpoint_count, ao_s, ao_t
            ) + second.reshape(kpoint_count, ao_t, ao_s).transpose(0, 2, 1)
    return result


def inverse_fourier_hartree_operator(
    d_k: dict[tuple[int, int], np.ndarray],
    kfracs: list[tuple[float, float, float]],
    translations: list[tuple[int, int, int]],
    bvk_remap: dict[tuple[int, int, tuple[int, int, int]], list[tuple[int, int, int]]]
    | None = None,
) -> dict[tuple[int, int, tuple[int, int, int]], np.ndarray]:
    """src/qsgw/hartree_workflow.cpp:459-545 (+55-80 add_weighted_operator_block).

    D(R) = (1/N_k) sum_k D_k exp(-2 pi i k.R); when a BvK remap entry exists
    the block is scattered to the remapped cells with weight 1/|remapped|.
    """
    n_k = len(kfracs)
    if n_k == 0 or not translations:
        raise RecomputeError("inverse Fourier grids are empty")
    result: dict[tuple[int, int, tuple[int, int, int]], np.ndarray] = {}
    for (atom_i, atom_j), blocks in d_k.items():
        if blocks.shape[0] != n_k:
            raise RecomputeError(
                f"operator block {(atom_i, atom_j)} does not cover the k grid"
            )
        for translation in translations:
            block = np.zeros(blocks.shape[1:], dtype=np.complex128)
            for kpoint, kfrac in enumerate(kfracs):
                block += (
                    fourier_phase(kfrac, translation, -1.0) / n_k
                ) * blocks[kpoint]
            remapped = bvk_remap.get((atom_i, atom_j, translation)) if bvk_remap else None
            if not remapped:
                targets = [(translation, 1.0)]
            else:
                weight = 1.0 / len(remapped)
                targets = [(target, weight) for target in remapped]
            for target_translation, weight in targets:
                key = (atom_i, atom_j, target_translation)
                if key in result and result[key].shape != block.shape:
                    raise RecomputeError(
                        "BvK remap combines incompatible blocks"
                    )
                if key not in result:
                    result[key] = np.zeros(block.shape, dtype=np.complex128)
                result[key] += weight * block
    return result


def assemble_operator_k(
    hartree_r: dict[tuple[int, int, tuple[int, int, int]], np.ndarray],
    atom_ao_sizes: dict[int, int],
    kfrac: tuple[float, float, float],
) -> np.ndarray:
    """src/qsgw/hartree_density.cpp:533-555 (R -> k with phase sign +1.0)."""
    n_atoms = len(atom_ao_sizes)
    offsets: dict[int, int] = {}
    total = 0
    for atom in range(n_atoms):
        offsets[atom] = total
        total += atom_ao_sizes[atom]
    operator_k = np.zeros((total, total), dtype=np.complex128)
    for (atom_i, atom_j, translation), block in hartree_r.items():
        factor = fourier_phase(kfrac, translation, 1.0)
        operator_k[
            offsets[atom_i] : offsets[atom_i] + block.shape[0],
            offsets[atom_j] : offsets[atom_j] + block.shape[1],
        ] += factor * block
    return operator_k


def project_periodic_operator_to_fixed_basis(
    hartree_r: dict[tuple[int, int, tuple[int, int, int]], np.ndarray],
    atom_ao_sizes: dict[int, int],
    wfc: dict[int, dict[int, np.ndarray]],
    kfracs: list[tuple[float, float, float]],
) -> dict[tuple[int, int], np.ndarray]:
    """src/qsgw/hartree_density.cpp:511-601.

    projected = W^dag O_k W (plain spectral projection, NO overlap S), then
    Hermitized with 0.5*(P + P^dag).
    """
    result: dict[tuple[int, int], np.ndarray] = {}
    for spin, by_kpoint in wfc.items():
        for kpoint, wavefunctions in by_kpoint.items():
            operator_k = assemble_operator_k(
                hartree_r, atom_ao_sizes, kfracs[kpoint]
            )
            projected = wavefunctions.conj() @ operator_k @ wavefunctions.T
            result[(spin, kpoint)] = 0.5 * (projected + projected.conj().T)
    return result


# ---------------------------------------------------------------------------
# Comparison metrics (relative Frobenius mirrors
# src/qsgw/operator_fourier.cpp:41-61: sqrt(diff/max(1, scale)))
# ---------------------------------------------------------------------------


def _block_metrics(
    actual: dict, expected: dict
) -> tuple[float, float]:
    if set(actual) != set(expected):
        missing = sorted(set(actual) ^ set(expected))
        raise RecomputeError(
            f"block-key mismatch between recomputation and reference: {missing[:8]}"
        )
    max_abs = 0.0
    diff_sq = 0.0
    ref_sq = 0.0
    for key in actual:
        if actual[key].shape != expected[key].shape:
            raise RecomputeError(f"block shape mismatch at {key}")
        difference = actual[key] - expected[key]
        if difference.size:
            max_abs = max(max_abs, float(np.abs(difference).max()))
        diff_sq += float(np.sum(np.abs(difference) ** 2))
        ref_sq += float(np.sum(np.abs(expected[key]) ** 2))
    return max_abs, math.sqrt(diff_sq / max(1.0, ref_sq))


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def _parse_call_index(call_dir: Path) -> int:
    name = call_dir.name
    if not name.startswith("call_") or not name[5:].isdigit():
        raise RecomputeError(
            f"dump call directory {name!r} does not match call_NNN"
        )
    return int(name[5:])


def _parse_iterations(spec: str) -> tuple[int, int]:
    parts = spec.split(":")
    if len(parts) != 2:
        raise RecomputeError(f"--iterations must be A:B, got {spec!r}")
    first = _parse_int(parts[0], "--iterations start")
    last = _parse_int(parts[1], "--iterations end")
    if first < 0 or last < first:
        raise RecomputeError(f"invalid --iterations window {spec!r}")
    return first, last


def run_checks(args: argparse.Namespace) -> dict:
    input_dir = Path(args.input_dir)
    call_dir = Path(args.dump_call)
    trace_path = Path(args.trace)
    abs_tol = args.matrix_max_abs_tolerance_ha
    rel_tol = args.matrix_relative_tolerance
    if not (abs_tol > 0.0 and rel_tol > 0.0):
        raise RecomputeError("tolerances must be positive")
    window = _parse_iterations(args.iterations)
    notes: list[str] = []

    manifest = load_manifest(call_dir)
    atom_ao_sizes: dict[int, int] = manifest["atom_ao_sizes"]
    n_atoms = len(atom_ao_sizes)
    n_aos = sum(atom_ao_sizes.values())
    kpoint_count: int = manifest["kpoint_count"]
    period: tuple[int, int, int] = manifest["period"]
    normalization: str = manifest["normalization"]
    translations = construct_r_grid(period)

    call_index = _parse_call_index(call_dir)
    required_iterations = {0, 1, max(0, call_index - 1), call_index}
    outside = sorted(
        iteration
        for iteration in required_iterations
        if not (window[0] <= iteration <= window[1])
    )
    if outside:
        raise RecomputeError(
            f"trace iterations {outside} required for dump call_{call_index:03d} "
            f"fall outside --iterations {window[0]}:{window[1]}"
        )
    notes.append(
        f"dump call_{call_index:03d} maps to QSGW iteration {call_index}: "
        f"occupation anchors are iterations 0 (reference) and "
        f"{max(0, call_index - 1)} (live); delta_vh is compared at iteration "
        f"{call_index} (reported under the iteration2_* keys) and iteration 1 "
        f"must vanish"
    )

    density_delta_k = load_density_delta_k(call_dir, kpoint_count, n_aos)
    hartree_k_dump = load_hartree_k(call_dir, atom_ao_sizes, kpoint_count)
    hartree_r_dump = load_hartree_r(call_dir, atom_ao_sizes)

    bz = read_bz_sampling(input_dir / "bz_sampling_out")
    if len(bz["kfracs"]) != kpoint_count:
        raise RecomputeError(
            "bz_sampling k-point count does not match the dump kpoint_count"
        )
    if tuple(bz["period"]) != tuple(period):
        raise RecomputeError(
            f"bz_sampling grid {bz['period']} does not match dump period {period}"
        )
    kfracs = list(bz["kfracs"])

    band = read_band_out(input_dir / "band_out")
    if band["n_kpoints"] != kpoint_count:
        raise RecomputeError("band_out k-point count does not match the dump")
    if band["n_basis_wfc"] != n_aos:
        raise RecomputeError(
            f"band_out basis {band['n_basis_wfc']} does not match the dump "
            f"AO total {n_aos}"
        )
    if band["n_spins"] != 1:
        raise RecomputeError(
            "only n_spins == 1 is supported by this recomputation tool"
        )

    cs = read_cs_text(input_dir)
    atom_aux_sizes = infer_atom_aux_sizes(cs, atom_ao_sizes)
    missing_pairs = [
        (i, j)
        for i in range(n_atoms)
        for j in range(n_atoms)
        if (i, j) not in cs
    ]
    if missing_pairs:
        notes.append(
            f"Cs pairs {missing_pairs} are absent from the Cs files and are "
            f"zero-filled exactly like the pre-initialized C++ c_k map "
            f"(hartree_workflow.cpp:229-241)"
        )
    c_k = build_c_k(cs, atom_ao_sizes, atom_aux_sizes, kfracs)

    gamma_q_num = bz["gamma_index"] + 1
    v_full_gamma, coulomb_zero_filled = read_coulomb_gamma_full(
        input_dir, gamma_q_num
    )
    if coulomb_zero_filled:
        notes.append(
            f"Coulomb Gamma full matrix had {coulomb_zero_filled} entries "
            f"absent from the files; they are zero-filled like the C++ "
            f"Vq_full.create (reader_coulomb.cpp:114-117)"
        )
    notes.append(
        f"Coulomb q=Gamma block selected as q_num={gamma_q_num} "
        f"(Gamma is bz row {bz['gamma_index'] + 1}; the Hartree reader stages "
        f"klist := klist_full so q_num-1 indexes the full k list)"
    )
    v_q0 = build_v_q0(v_full_gamma, atom_aux_sizes)

    # ---- Check A: contraction ----
    density_blocks = split_weighted_density_by_atom(density_delta_k, atom_ao_sizes)
    d_k_python = contract_hartree_full_grid(
        c_k, v_q0, density_blocks, atom_ao_sizes, atom_aux_sizes, normalization
    )
    contraction_max_abs, contraction_rel = _block_metrics(
        d_k_python, hartree_k_dump
    )

    # ---- Check B: inverse Fourier (sourced from the dump hartree_k so the
    # check is independent of check A) ----
    notes.append(
        "check B transforms the dump hartree_k (not the check-A output) so "
        "the inverse-Fourier convention is validated independently"
    )
    notes.append(
        "BvK remap is assumed identity (no remap table is read); a "
        "non-identity C++ remap would surface as a check-B/D mismatch"
    )
    hartree_r_python = inverse_fourier_hartree_operator(
        hartree_k_dump, kfracs, translations, bvk_remap=None
    )
    fourier_max_abs, fourier_rel = _block_metrics(hartree_r_python, hartree_r_dump)

    # ---- Check C: density invariants ----
    occ = trace_occupations(trace_path, {0, max(0, call_index - 1)})
    hermiticity_max = 0.0
    per_k_trace_max_abs = 0.0
    total_charge = 0.0 + 0.0j
    for kpoint in range(kpoint_count):
        matrix = density_delta_k[kpoint]
        hermiticity_max = max(
            hermiticity_max, float(np.abs(matrix - matrix.conj().T).max())
        )
        delta_n = 0.0 + 0.0j
        for spin in range(band["n_spins"]):
            key_live = (max(0, call_index - 1), spin, kpoint)
            key_ref = (0, spin, kpoint)
            if key_live not in occ or key_ref not in occ:
                raise RecomputeError(
                    f"trace misses occupation rows for {key_live} or {key_ref}"
                )
            delta_n += float(np.sum(occ[key_live]).real - np.sum(occ[key_ref]).real)
        per_k_trace_max_abs = max(
            per_k_trace_max_abs, abs(complex(np.trace(matrix)) - delta_n)
        )
        total_charge += bz["weights"][kpoint] * complex(np.trace(matrix))
    notes.append(
        "occupation trace values are the MeanField weights (band_out "
        "occupation / n_kpoints, read_data.cpp:469) with bands in the column "
        "index (iteration_trace.cpp:365-371); Tr delta_rho(k) equals "
        "sum_spin sum_b (f_live - f_reference) because the dumped density is "
        "the exact DFT inverse of the weighted k-space density delta"
    )

    # ---- Check D: projection + end-to-end delta_vh comparison ----
    wfc = read_ks_eigenvectors_text(
        input_dir,
        band["n_spins"],
        band["n_states"],
        n_aos,
        kpoint_count,
    )
    projected = project_periodic_operator_to_fixed_basis(
        hartree_r_python, atom_ao_sizes, wfc, kfracs
    )
    delta_vh = trace_delta_vh(trace_path, {1, call_index})
    actual_blocks = {
        (spin, kpoint): block for (spin, kpoint), block in projected.items()
    }
    expected_blocks = {
        (spin, kpoint): block
        for (iteration, spin, kpoint), block in delta_vh.items()
        if iteration == call_index
    }
    if not expected_blocks:
        raise RecomputeError(
            f"trace misses delta_vh rows at iteration {call_index}"
        )
    projection_max_abs, projection_rel = _block_metrics(
        actual_blocks, expected_blocks
    )
    iteration1_max_abs = 0.0
    for (iteration, spin, kpoint), block in delta_vh.items():
        if iteration == 1:
            if block.size:
                iteration1_max_abs = max(
                    iteration1_max_abs, float(np.abs(block).max())
                )
    if call_index > 1 and not any(
        iteration == 1 for (iteration, _spin, _kpoint) in delta_vh
    ):
        # call_001 is itself the first (vanishing) update, so for N == 1 the
        # comparison target above already covers the iteration-1 block.
        raise RecomputeError("trace misses delta_vh rows at iteration 1")
    notes.append(
        "projection is the plain spectral W^dag O W without an overlap "
        "matrix S, followed by 0.5*(P+P^dag) (hartree_density.cpp:557-596)"
    )

    contraction_passed = contraction_max_abs <= abs_tol and contraction_rel <= rel_tol
    fourier_passed = fourier_max_abs <= abs_tol and fourier_rel <= rel_tol
    invariants_passed = (
        hermiticity_max <= abs_tol and per_k_trace_max_abs <= abs_tol
    )
    projection_passed = (
        projection_max_abs <= abs_tol
        and projection_rel <= rel_tol
        and iteration1_max_abs <= abs_tol
    )
    passed = (
        contraction_passed
        and fourier_passed
        and invariants_passed
        and projection_passed
    )

    return {
        "passed": passed,
        "contraction": {
            "max_abs": contraction_max_abs,
            "rel_frobenius": contraction_rel,
            "passed": contraction_passed,
        },
        "inverse_fourier": {
            "max_abs": fourier_max_abs,
            "rel_frobenius": fourier_rel,
            "passed": fourier_passed,
        },
        "density_invariants": {
            "hermiticity_max": hermiticity_max,
            "per_k_trace_max_abs": per_k_trace_max_abs,
            "total_charge": total_charge.real,
            "total_charge_imag": total_charge.imag,
            "passed": invariants_passed,
        },
        "projection_end_to_end": {
            "iteration2_max_abs": projection_max_abs,
            "iteration2_rel_frobenius": projection_rel,
            "iteration1_max_abs": iteration1_max_abs,
            "passed": projection_passed,
        },
        "notes": notes,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Independently recompute one QSGW Hartree delta-V update and "
            "compare against the C++ pipeline dump and frozen trace."
        )
    )
    parser.add_argument("--input-dir", required=True, help="frozen input directory")
    parser.add_argument("--dump-call", required=True, help="dump call_NNN directory")
    parser.add_argument("--trace", required=True, help="qsgw_matrices.dat trace")
    parser.add_argument(
        "--iterations",
        default="0:2",
        help="inclusive trace iteration window A:B (default 0:2)",
    )
    parser.add_argument("--output", required=True, help="JSON report path")
    parser.add_argument(
        "--matrix-max-abs-tolerance-ha",
        type=float,
        default=1.0e-8,
        help="max-abs tolerance in Hartree (default 1e-8)",
    )
    parser.add_argument(
        "--matrix-relative-tolerance",
        type=float,
        default=1.0e-8,
        help="relative Frobenius tolerance (default 1e-8)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_path = Path(args.output)
    try:
        report = run_checks(args)
    except Exception as exc:  # fail-closed: any error -> exit 1
        report = {"passed": False, "error": str(exc)}
        exit_code = 1
    else:
        exit_code = 0 if report["passed"] else 2
    output_path.write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print(json.dumps(report, indent=2))
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
