#!/usr/bin/env python3

import argparse
import json
import math
import re
from pathlib import Path

import numpy as np

HA2EV = 27.211386245988
RY2EV = 0.5 * HA2EV

BLOCK_RE = re.compile(r"^\s*(-?\d+)\s+(-?\d+)\s+(-?\d+)\s+(\d+)\s*$")
VX_COMPLEX_RE = re.compile(r"\(\s*([^,]+)\s*,\s*([^)]+)\)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Adapt ABACUS SCF outputs into the matrix-text inputs expected by LibRPA QSGW."
    )
    parser.add_argument("--scf-dir", type=Path, required=True, help="ABACUS SCF output directory")
    parser.add_argument(
        "--outdir",
        type=Path,
        required=True,
        help="Directory where s1k*/vxcs1k* files will be written",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Optional JSON report path. Defaults to <outdir>/qsgw_adapter_validation.json",
    )
    parser.add_argument(
        "--write-legacy-sks-alias",
        action="store_true",
        help="Also write sks1k*_nao.txt aliases alongside s1k*_nao.txt",
    )
    parser.add_argument(
        "--max-rms-error-ev",
        type=float,
        default=None,
        help="Fail if the H(R)/S(R) reconstruction RMS error exceeds this threshold",
    )
    return parser.parse_args()


def parse_csr_dense_blocks(path: Path) -> tuple[int, dict[tuple[int, int, int], np.ndarray]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    dim = None
    blocks: dict[tuple[int, int, int], np.ndarray] = {}
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue
        if line.startswith("STEP:") or line.startswith("Matrix number"):
            i += 1
            continue
        if line.startswith("Matrix Dimension"):
            dim = int(line.split(":", 1)[1].strip())
            i += 1
            continue

        match = BLOCK_RE.match(line)
        if match is None:
            raise ValueError(f"Unrecognized CSR line in {path}: {lines[i]!r}")
        if dim is None:
            raise ValueError(f"CSR dimension header missing before data block in {path}")

        rx, ry, rz, nnz = map(int, match.groups())
        i += 1
        if nnz == 0:
            continue
        if i + 2 >= len(lines):
            raise ValueError(f"Incomplete CSR block for R=({rx},{ry},{rz}) in {path}")

        values = [float(x) for x in lines[i].split()]
        cols = [int(x) for x in lines[i + 1].split()]
        row_ptr = [int(x) for x in lines[i + 2].split()]
        i += 3

        if len(values) != nnz or len(cols) != nnz:
            raise ValueError(
                f"CSR nnz mismatch for R=({rx},{ry},{rz}) in {path}: "
                f"expected {nnz}, got {len(values)} values and {len(cols)} indices"
            )
        if len(row_ptr) != dim + 1:
            raise ValueError(
                f"CSR row_ptr size mismatch for R=({rx},{ry},{rz}) in {path}: "
                f"expected {dim + 1}, got {len(row_ptr)}"
            )

        mat = np.zeros((dim, dim), dtype=np.complex128)
        for row in range(dim):
            start = row_ptr[row]
            end = row_ptr[row + 1]
            for idx in range(start, end):
                mat[row, cols[idx]] = values[idx]
        blocks[(rx, ry, rz)] = mat

    if dim is None:
        raise ValueError(f"Failed to read matrix dimension from {path}")
    return dim, blocks


def parse_kpt_info(path: Path) -> list[np.ndarray]:
    kpts: list[np.ndarray] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) == 5 and parts[0].isdigit():
            kpts.append(np.array([float(parts[1]), float(parts[2]), float(parts[3])], dtype=float))
    if not kpts:
        raise ValueError(f"Failed to parse any k-points from {path}")
    return kpts


def parse_eig_occ(path: Path) -> list[np.ndarray]:
    eigenvalues_ev: list[np.ndarray] = []
    current: list[float] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("spin="):
            if current:
                eigenvalues_ev.append(np.array(current, dtype=float))
                current = []
            continue
        parts = line.split()
        if len(parts) == 3 and parts[0].isdigit():
            current.append(float(parts[1]))
    if current:
        eigenvalues_ev.append(np.array(current, dtype=float))
    if not eigenvalues_ev:
        raise ValueError(f"Failed to parse eigenvalues from {path}")
    return eigenvalues_ev


def reconstruct_k_matrices(
    blocks: dict[tuple[int, int, int], np.ndarray],
    kpts: list[np.ndarray],
    phase_sign: int,
) -> list[np.ndarray]:
    sample = next(iter(blocks.values()))
    mats: list[np.ndarray] = []
    for kpt in kpts:
        accum = np.zeros_like(sample)
        for rvec, block in blocks.items():
            phase = np.exp(phase_sign * 2j * math.pi * float(np.dot(kpt, np.array(rvec))))
            accum += phase * block
        mats.append(0.5 * (accum + accum.conj().T))
    return mats


def generalized_hermitian_eigvals(hmat: np.ndarray, smat: np.ndarray) -> tuple[np.ndarray, float]:
    smat = 0.5 * (smat + smat.conj().T)
    hmat = 0.5 * (hmat + hmat.conj().T)
    s_eval, s_vec = np.linalg.eigh(smat)
    s_min = float(np.min(s_eval.real))
    if s_min <= 1e-10:
        raise ValueError(f"Overlap matrix is not positive definite; min eigenvalue = {s_min}")
    inv_sqrt_s = s_vec @ np.diag(np.power(s_eval.real, -0.5)) @ s_vec.conj().T
    ortho_h = inv_sqrt_s @ hmat @ inv_sqrt_s
    ortho_h = 0.5 * (ortho_h + ortho_h.conj().T)
    evals = np.linalg.eigvalsh(ortho_h)
    return evals.real, s_min


def pick_best_fourier_convention(
    h_blocks: dict[tuple[int, int, int], np.ndarray],
    s_blocks: dict[tuple[int, int, int], np.ndarray],
    kpts: list[np.ndarray],
    ref_eigs_ev: list[np.ndarray],
) -> tuple[dict[str, float | int | str], list[np.ndarray], list[np.ndarray]]:
    candidates = []
    for phase_sign in (+1, -1):
        s_k = reconstruct_k_matrices(s_blocks, kpts, phase_sign)
        h_k = reconstruct_k_matrices(h_blocks, kpts, phase_sign)
        for unit_name, unit_scale_ev in (("ry", RY2EV), ("ha", HA2EV)):
            diffs = []
            min_s_eig = float("inf")
            for hk, sk, ref in zip(h_k, s_k, ref_eigs_ev):
                evals, s_min = generalized_hermitian_eigvals(hk, sk)
                min_s_eig = min(min_s_eig, s_min)
                diffs.append(evals[: len(ref)] * unit_scale_ev - ref)
            all_diffs = np.concatenate(diffs)
            rms = float(np.sqrt(np.mean(np.square(np.abs(all_diffs)))))
            max_abs = float(np.max(np.abs(all_diffs)))
            candidates.append(
                (
                    rms,
                    {
                        "phase_sign": phase_sign,
                        "hamiltonian_unit": unit_name,
                        "rms_error_ev": rms,
                        "max_abs_error_ev": max_abs,
                        "min_overlap_eigenvalue": min_s_eig,
                    },
                    s_k,
                    h_k,
                )
            )
    best = min(candidates, key=lambda item: item[0])
    return best[1], best[2], best[3]


def parse_abacus_upper_triangle_matrix(path: Path) -> np.ndarray:
    text = path.read_text(encoding="utf-8")
    nrows = None
    rows: dict[int, list[complex]] = {}
    current_row = None
    current_values: list[complex] = []

    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("# rows"):
            nrows = int(line.split()[-1])
            continue
        if line.startswith("#"):
            continue
        if line.startswith("Row "):
            if current_row is not None:
                rows[current_row] = current_values
            current_row = int(line.split()[1]) - 1
            current_values = []
            continue
        for real_part, imag_part in VX_COMPLEX_RE.findall(line):
            current_values.append(complex(float(real_part), float(imag_part)))

    if current_row is not None:
        rows[current_row] = current_values
    if nrows is None:
        raise ValueError(f"Failed to parse matrix size from {path}")

    mat = np.zeros((nrows, nrows), dtype=np.complex128)
    for row in range(nrows):
        values = rows.get(row)
        if values is None:
            raise ValueError(f"Missing Row {row + 1} in {path}")
        expected = nrows - row
        if len(values) != expected:
            raise ValueError(
                f"Upper-triangle entry count mismatch in {path} row {row + 1}: "
                f"expected {expected}, got {len(values)}"
            )
        for offset, value in enumerate(values):
            col = row + offset
            mat[row, col] = value
            mat[col, row] = np.conj(value)
    return 0.5 * (mat + mat.conj().T)


def write_upper_triangle_matrix(path: Path, mat: np.ndarray) -> None:
    hermitian = 0.5 * (mat + mat.conj().T)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write(f"{hermitian.shape[0]}\n")
        for row in range(hermitian.shape[0]):
            entries = []
            for col in range(row, hermitian.shape[1]):
                value = hermitian[row, col]
                entries.append(f"({value.real:.16e},{value.imag:.16e})")
            handle.write(" ".join(entries))
            handle.write("\n")


def main() -> int:
    args = parse_args()
    scf_dir = args.scf_dir.resolve()
    outdir = args.outdir.resolve()
    report_path = args.report.resolve() if args.report is not None else outdir / "qsgw_adapter_validation.json"

    hrs_path = scf_dir / "hrs1_nao.csr"
    srs_path = scf_dir / "srs1_nao.csr"
    kpt_path = scf_dir / "KPT.info"
    eig_occ_path = scf_dir / "eig_occ.txt"

    n_h, h_blocks = parse_csr_dense_blocks(hrs_path)
    n_s, s_blocks = parse_csr_dense_blocks(srs_path)
    if n_h != n_s:
        raise ValueError(f"H/S dimension mismatch: {n_h} vs {n_s}")

    kpts = parse_kpt_info(kpt_path)
    ref_eigs_ev = parse_eig_occ(eig_occ_path)
    if len(kpts) != len(ref_eigs_ev):
        raise ValueError(
            f"k-point count mismatch between {kpt_path} ({len(kpts)}) and {eig_occ_path} ({len(ref_eigs_ev)})"
        )

    validation, s_k, h_k = pick_best_fourier_convention(h_blocks, s_blocks, kpts, ref_eigs_ev)

    written_s_files = []
    written_vxc_files = []
    for ik, smat in enumerate(s_k, start=1):
        s_name = outdir / f"s1k{ik}_nao.txt"
        write_upper_triangle_matrix(s_name, smat)
        written_s_files.append(str(s_name))
        if args.write_legacy_sks_alias:
            alias_name = outdir / f"sks1k{ik}_nao.txt"
            write_upper_triangle_matrix(alias_name, smat)

    for ik in range(1, len(kpts) + 1):
        vxc_source = scf_dir / f"vxck{ik}_nao.txt"
        vxc_target = outdir / f"vxcs1k{ik}_nao.txt"
        vxc_matrix = parse_abacus_upper_triangle_matrix(vxc_source)
        write_upper_triangle_matrix(vxc_target, vxc_matrix)
        written_vxc_files.append(str(vxc_target))

    report = {
        "task": "abacus_qsgw_adapter",
        "scf_dir": str(scf_dir),
        "outdir": str(outdir),
        "nkpoints": len(kpts),
        "matrix_dimension": n_h,
        "phase_sign": validation["phase_sign"],
        "hamiltonian_unit": validation["hamiltonian_unit"],
        "rms_error_ev": validation["rms_error_ev"],
        "max_abs_error_ev": validation["max_abs_error_ev"],
        "min_overlap_eigenvalue": validation["min_overlap_eigenvalue"],
        "written_s_files": len(written_s_files),
        "written_vxc_files": len(written_vxc_files),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps(report, indent=2))

    if args.max_rms_error_ev is not None and validation["rms_error_ev"] > args.max_rms_error_ev:
        raise SystemExit(
            f"RMS reconstruction error {validation['rms_error_ev']:.6e} eV exceeds "
            f"threshold {args.max_rms_error_ev:.6e} eV"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
