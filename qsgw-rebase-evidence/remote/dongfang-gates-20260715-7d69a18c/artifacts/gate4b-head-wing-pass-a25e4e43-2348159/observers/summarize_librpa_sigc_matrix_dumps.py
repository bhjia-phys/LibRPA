#!/usr/bin/env python3
"""Summarize LibRPA SigC AO/KS matrix dumps.

The AO matrix is written as MatrixMarket:

  SigcKF_<source>_ispin_<spin>_ik_<k>_ifreq_<ifreq>.mtx

The newarch KS matrix is written as a compact binary square matrix:

  Sigc_fk_mn_<source>_ispin_<spin>_ik_<k>_ifreq_<ifreq>.bin

The legacy driver can instead write KS matrices as MatrixMarket:

  Sigc_fk_mn_ispin_<spin>_ik_<k>_ifreq_<ifreq>.mtx

Dump indices in filenames are zero-based. The optional --state argument is
one-based, matching LibRPA band/state reporting.
"""

from __future__ import annotations

import argparse
import json
import math
import struct
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


HA2EV = 27.211386245988


@dataclass
class MatrixStat:
    shape: tuple[int, int]
    max_abs: float
    max_abs_location: tuple[int, int] | None
    max_abs_value: complex
    max_diag_abs: float
    max_diag_location: int | None
    max_diag_value: complex
    max_offdiag_abs: float
    hermitian_resid: float

    def scaled(self, scale: float) -> dict[str, object]:
        return {
            "shape": list(self.shape),
            "max_abs": self.max_abs * scale,
            "max_abs_location": self.max_abs_location,
            "max_abs_value": scale_complex(self.max_abs_value, scale),
            "max_diag_abs": self.max_diag_abs * scale,
            "max_diag_location": self.max_diag_location,
            "max_diag_value": scale_complex(self.max_diag_value, scale),
            "max_offdiag_abs": self.max_offdiag_abs * scale,
            "hermitian_resid": self.hermitian_resid * scale,
        }


def scale_complex(value: complex, scale: float) -> dict[str, float]:
    return {"re": value.real * scale, "im": value.imag * scale}


def resolve_output_dir(path: Path) -> Path:
    if (path / "librpa.d").is_dir():
        return path / "librpa.d"
    return path


def read_matrix_market(path: Path) -> list[list[complex]]:
    rows: int | None = None
    cols: int | None = None
    data: dict[tuple[int, int], complex] = {}
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("%"):
                continue
            parts = stripped.split()
            if rows is None:
                if len(parts) < 3:
                    raise ValueError(f"bad MatrixMarket dimension line in {path}: {stripped}")
                rows = int(parts[0])
                cols = int(parts[1])
                continue
            if len(parts) < 3:
                raise ValueError(f"bad MatrixMarket entry in {path}: {stripped}")
            i = int(parts[0]) - 1
            j = int(parts[1]) - 1
            re_val = float(parts[2])
            im_val = float(parts[3]) if len(parts) > 3 else 0.0
            data[(i, j)] = complex(re_val, im_val)
    if rows is None or cols is None:
        raise ValueError(f"missing MatrixMarket dimensions in {path}")
    mat = [[0j for _ in range(cols)] for _ in range(rows)]
    for (i, j), value in data.items():
        mat[i][j] = value
    return mat


def read_sigc_ks_binary(path: Path) -> list[list[complex]]:
    with path.open("rb") as handle:
        header = handle.read(8)
        if len(header) != 8:
            raise ValueError(f"{path} is too short for a SigC binary header")
        n_states, type_bytes = struct.unpack("=ii", header)
        if n_states <= 0:
            raise ValueError(f"{path} has invalid n_states={n_states}")
        if type_bytes != 8:
            raise ValueError(f"{path} has unsupported scalar byte size={type_bytes}")
        expected_bytes = n_states * n_states * 16
        payload = handle.read()
    if len(payload) != expected_bytes:
        raise ValueError(
            f"{path} payload has {len(payload)} bytes, expected {expected_bytes}"
        )
    values = struct.unpack(f"={n_states * n_states * 2}d", payload)
    mat = [[0j for _ in range(n_states)] for _ in range(n_states)]
    cursor = 0
    for i in range(n_states):
        for j in range(n_states):
            mat[i][j] = complex(values[cursor], values[cursor + 1])
            cursor += 2
    return mat


def read_ks_matrix(path: Path) -> list[list[complex]]:
    if path.suffix == ".bin":
        return read_sigc_ks_binary(path)
    if path.suffix == ".mtx":
        return read_matrix_market(path)
    raise ValueError(f"unsupported KS matrix file suffix: {path}")


def matrix_stats(mat: list[list[complex]]) -> MatrixStat:
    rows = len(mat)
    cols = len(mat[0]) if rows else 0
    max_abs = 0.0
    max_abs_location: tuple[int, int] | None = None
    max_abs_value = 0j
    max_diag_abs = 0.0
    max_diag_location: int | None = None
    max_diag_value = 0j
    max_offdiag_abs = 0.0
    hermitian_resid = 0.0
    for i in range(rows):
        for j in range(cols):
            value = mat[i][j]
            value_abs = abs(value)
            if value_abs > max_abs:
                max_abs = value_abs
                max_abs_location = (i + 1, j + 1)
                max_abs_value = value
            if i == j:
                if value_abs > max_diag_abs:
                    max_diag_abs = value_abs
                    max_diag_location = i + 1
                    max_diag_value = value
            else:
                max_offdiag_abs = max(max_offdiag_abs, value_abs)
            if j < rows and i < cols:
                hermitian_resid = max(hermitian_resid, abs(value - mat[j][i].conjugate()))
    return MatrixStat(
        shape=(rows, cols),
        max_abs=max_abs,
        max_abs_location=max_abs_location,
        max_abs_value=max_abs_value,
        max_diag_abs=max_diag_abs,
        max_diag_location=max_diag_location,
        max_diag_value=max_diag_value,
        max_offdiag_abs=max_offdiag_abs,
        hermitian_resid=hermitian_resid,
    )


def parse_self_energy_point(
    path: Path, spin: int, kpoint: int, state: int, ifreq: int
) -> complex:
    lines = [line.strip() for line in path.read_text(encoding="utf-8", errors="replace").splitlines() if line.strip()]
    if not lines:
        raise ValueError(f"empty self-energy file: {path}")
    header = lines[0].split()
    if len(header) != 4:
        raise ValueError(f"bad self-energy header in {path}")
    nfreq, nspin, nkpt, nbands = (int(item) for item in header)
    if not (1 <= spin <= nspin and 1 <= kpoint <= nkpt and 1 <= state <= nbands and 1 <= ifreq <= nfreq):
        raise ValueError(
            "requested self-energy point is outside header bounds: "
            f"spin={spin}/{nspin}, k={kpoint}/{nkpt}, state={state}/{nbands}, ifreq={ifreq}/{nfreq}"
        )
    value_lines = lines[1 + nfreq :]
    row = (((spin - 1) * nkpt + (kpoint - 1)) * nbands + (state - 1)) * nfreq + (ifreq - 1)
    if row >= len(value_lines):
        raise ValueError(f"self-energy file ended before row {row}")
    parts = value_lines[row].split()
    if len(parts) != 2:
        raise ValueError(f"bad self-energy row {row}: {value_lines[row]!r}")
    return complex(float(parts[0]), float(parts[1]))


def format_complex(value: complex, scale: float) -> str:
    return f"({value.real * scale:.10e},{value.imag * scale:.10e})"


def print_text(summary: dict[str, object], unit: str) -> None:
    scale = HA2EV if unit == "eV" else 1.0
    print(f"UNIT {unit}")
    print(f"AO_FILE {summary['ao_file']}")
    print(f"KS_FILE {summary['ks_file']}")
    for key in ("ao", "ks"):
        stat = summary[key]
        if stat is None:
            continue
        assert isinstance(stat, MatrixStat)
        print(
            key.upper()
            + f" shape={stat.shape[0]}x{stat.shape[1]}"
            + f" max_abs={stat.max_abs * scale:.10e}"
            + f" max_abs_loc={stat.max_abs_location}"
            + f" max_abs_value={format_complex(stat.max_abs_value, scale)}"
            + f" max_diag_abs={stat.max_diag_abs * scale:.10e}"
            + f" max_diag_loc={stat.max_diag_location}"
            + f" max_diag_value={format_complex(stat.max_diag_value, scale)}"
            + f" max_offdiag_abs={stat.max_offdiag_abs * scale:.10e}"
            + f" hermitian_resid={stat.hermitian_resid * scale:.10e}"
        )
    state_value = summary.get("state_diag")
    if isinstance(state_value, complex):
        print(f"STATE_DIAG state={summary['state']} value={format_complex(state_value, scale)}")
    se_value = summary.get("self_energy_omega")
    if isinstance(se_value, complex):
        print(f"SELF_ENERGY_OMEGA value={format_complex(se_value, scale)}")
        if isinstance(state_value, complex):
            print(f"STATE_MINUS_SELF_ENERGY value={format_complex(state_value - se_value, scale)}")


def build_summary(args: argparse.Namespace) -> dict[str, object]:
    output_dir = resolve_output_dir(args.path)
    ao_file = output_dir / f"SigcKF_{args.source}_ispin_{args.spin}_ik_{args.k}_ifreq_{args.ifreq}.mtx"
    ks_candidates = [
        output_dir / f"Sigc_fk_mn_{args.source}_ispin_{args.spin}_ik_{args.k}_ifreq_{args.ifreq}.bin",
        output_dir / f"Sigc_fk_mn_{args.source}_ispin_{args.spin}_ik_{args.k}_ifreq_{args.ifreq}.mtx",
        output_dir / f"Sigc_fk_mn_ispin_{args.spin}_ik_{args.k}_ifreq_{args.ifreq}.mtx",
    ]
    ks_file = next((candidate for candidate in ks_candidates if candidate.exists()), ks_candidates[0])
    ao = read_matrix_market(ao_file) if ao_file.exists() else None
    ks = read_ks_matrix(ks_file)
    summary: dict[str, object] = {
        "ao_file": str(ao_file) if ao is not None else None,
        "ks_file": str(ks_file),
        "ao": matrix_stats(ao) if ao is not None else None,
        "ks": matrix_stats(ks),
    }
    if args.state is not None:
        state_index = args.state - 1
        if state_index < 0 or state_index >= len(ks):
            raise ValueError(f"--state {args.state} outside KS matrix shape {len(ks)}")
        summary["state"] = args.state
        summary["state_diag"] = ks[state_index][state_index]
    if args.self_energy is not None:
        summary["self_energy_omega"] = parse_self_energy_point(
            args.self_energy,
            spin=args.spin + 1,
            kpoint=args.k + 1,
            state=args.state if args.state is not None else 1,
            ifreq=args.ifreq + 1,
        )
    return summary


def json_default(value: object) -> object:
    if isinstance(value, MatrixStat):
        return asdict(value)
    if isinstance(value, complex):
        return {"re": value.real, "im": value.imag}
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"cannot serialize {type(value)!r}")


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, help="Run directory or librpa.d directory.")
    parser.add_argument("--source", default="kgrid")
    parser.add_argument("--spin", type=int, default=0, help="Zero-based filename spin index.")
    parser.add_argument("--k", type=int, default=0, help="Zero-based filename k-point index.")
    parser.add_argument("--ifreq", type=int, default=0, help="Zero-based filename frequency index.")
    parser.add_argument("--state", type=int, default=None, help="One-based state index for KS diagonal reporting.")
    parser.add_argument("--self-energy", type=Path, default=None)
    parser.add_argument("--unit", choices=("Ha", "eV"), default="eV")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    summary = build_summary(args)
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True, default=json_default))
    else:
        print_text(summary, args.unit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
