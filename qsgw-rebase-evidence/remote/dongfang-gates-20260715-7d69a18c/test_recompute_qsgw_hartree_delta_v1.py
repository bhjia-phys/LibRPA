#!/usr/bin/env python3
"""Tests for the independent QSGW Hartree delta-V recomputation tool v1.

All expectations in this file are hand-computed from the C++ sources:

- contraction:        src/qsgw/hartree_kernel.cpp (contract_hartree_full_grid)
- density split:      src/qsgw/hartree_workflow.cpp (split_weighted_density_by_atom)
- inverse Fourier:    src/qsgw/hartree_workflow.cpp (inverse_fourier_hartree_operator,
                      fourier_phase with sign=-1, 1/N_k prefactor, BvK remap weights)
- projection:         src/qsgw/hartree_density.cpp (project_periodic_operator_to_fixed_basis,
                      plain W^dagger O W without overlap, then 0.5*(P+P^dagger))
- dump schema:        src/qsgw/hartree_dump.cpp (qsgw_hartree_pipeline_dump_v1)
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from recompute_qsgw_hartree_delta_v1 import (  # noqa: E402
    RecomputeError,
    build_c_k,
    build_v_q0,
    compute_density_aux,
    compute_potential_aux,
    contract_hartree_full_grid,
    construct_r_grid,
    fourier_phase,
    inverse_fourier_hartree_operator,
    load_manifest,
    project_periodic_operator_to_fixed_basis,
    read_coulomb_gamma_full,
    read_cs_text,
    split_weighted_density_by_atom,
    trace_occupations,
)

MODULE = Path(__file__).resolve().parent / "recompute_qsgw_hartree_delta_v1.py"

# ---------------------------------------------------------------------------
# Hand-computed synthetic fixture: 2 atoms, ao=(1,1), aux=(1,1), 2 k-points
# k0=(0,0,0), k1=(0.5,0,0), period (2,1,1) -> R in {(-1,0,0), (0,0,0)}.
# ---------------------------------------------------------------------------

P_CS, Q_CS, R_CS, S_CS = 2.0, 3.0, 5.0, 7.0
V00_FILE = 1.5 + 0.3j       # diagonal -> 0.5*(V+V^dag) = 1.5
V01_FILE = 0.25 + 0.5j      # upper off-diagonal passes through
V10_FILE_IGNORED = 9.0 + 9.0j  # lower triangle of the file is never read
V11_FILE = 2.5 - 0.4j       # diagonal -> 2.5

DR_K0 = np.array([[1.0, 0.1 + 0.2j], [0.1 - 0.2j, 4.0]], dtype=np.complex128)
DR_K1 = np.array([[2.0, 0.3 - 0.1j], [0.3 + 0.1j, 6.0]], dtype=np.complex128)

# density_aux[0] = sum_k 2*P*x0(k) + 2*Q*Re(y(k)) = 4.6 + 9.8 = 14.4
# density_aux[1] = sum_k 2*S*x1(k) + 2*R*Re(y(k)) = 57.0 + 87.0 = 144.0
DENSITY_AUX_0 = 14.4 + 0.0j
DENSITY_AUX_1 = 144.0 + 0.0j

# legacy_extra_inverse_nk: divide by N_k = 2
POT_AUX_0 = (1.5 * DENSITY_AUX_0 + V01_FILE * DENSITY_AUX_1) / 2.0
POT_AUX_1 = (V01_FILE.conjugate() * DENSITY_AUX_0 + 2.5 * DENSITY_AUX_1) / 2.0

D00 = 2.0 * P_CS * POT_AUX_0
D01 = Q_CS * POT_AUX_0 + R_CS * POT_AUX_1
D10 = R_CS * POT_AUX_1 + Q_CS * POT_AUX_0
D11 = 2.0 * S_CS * POT_AUX_1

DVH2_K0 = np.array(
    [[D00.real, 0.5 * (D01 + D10.conjugate()).real],
     [0.5 * (D01 + D10.conjugate()).real, D11.real]],
    dtype=np.complex128,
)
DVH2_K1 = np.array(
    [[D11.real, 0.5 * (D01 + D10.conjugate()).real],
     [0.5 * (D01 + D10.conjugate()).real, D00.real]],
    dtype=np.complex128,
)

OCC0 = {0: (0.5, 0.5), 1: (0.5, 0.5)}
OCC1 = {0: (2.5, 3.5), 1: (4.0, 5.0)}
TOTAL_CHARGE = 0.5 * 5.0 + 0.5 * 8.0


def _fmt(value: complex) -> str:
    return f"{value.real:.17e} {value.imag:.17e}"


def _write_fixture(
    root: Path,
    *,
    normalization: str = "legacy_extra_inverse_nk",
    hartree_k_scale: float = 1.0,
    hartree_k_perturb: complex = 0.0j,
    dvh1: complex = 0.0j,
    dr_hermitian: bool = True,
) -> dict[str, Path]:
    input_dir = root / "input"
    dump_call = root / "dump" / "call_002"
    input_dir.mkdir(parents=True)
    dump_call.mkdir(parents=True)
    trace_path = root / "qsgw_matrices.dat"

    (input_dir / "Cs_data_0.txt").write_text(
        "2 1\n"
        "1 1 0 0 0 1 1 1\n"
        f"{P_CS:.17e}\n"
        "1 2 0 0 0 1 1 1\n"
        f"{Q_CS:.17e}\n"
        "2 1 0 0 0 1 1 1\n"
        f"{R_CS:.17e}\n"
        "2 2 0 0 0 1 1 1\n"
        f"{S_CS:.17e}\n",
        encoding="utf-8",
        newline="\n",
    )
    (input_dir / "coulomb_cut_0.txt").write_text(
        "1\n"
        "2 1 2 1 2 1 1.0\n"
        f"{_fmt(V00_FILE)}\n"
        f"{_fmt(V01_FILE)}\n"
        f"{_fmt(V10_FILE_IGNORED)}\n"
        f"{_fmt(V11_FILE)}\n",
        encoding="utf-8",
        newline="\n",
    )
    (input_dir / "bz_sampling_out").write_text(
        "2 1 1\n"
        "2 2\n"
        "1 0.5 0.0 0.0 0.0 0.0 0.0 0.0 1 1\n"
        "2 0.5 0.5 0.0 0.0 3.141592653589793 0.0 0.0 2 2\n",
        encoding="utf-8",
        newline="\n",
    )
    (input_dir / "band_out").write_text(
        "2 1 2 2 0.0\n"
        "1 1\n"
        "1 0.5 0.0 0.0\n"
        "2 0.5 0.1 0.1\n"
        "2 1\n"
        "1 0.5 0.2 0.2\n"
        "2 0.5 0.3 0.3\n",
        encoding="utf-8",
        newline="\n",
    )
    (input_dir / "KS_eigenvector_1.dat").write_text(
        "1\n"
        "1.0 0.0\n"
        "0.0 0.0\n"
        "0.0 0.0\n"
        "1.0 0.0\n"
        "2\n"
        "0.0 0.0\n"
        "1.0 0.0\n"
        "1.0 0.0\n"
        "0.0 0.0\n",
        encoding="utf-8",
        newline="\n",
    )

    (dump_call / "manifest.txt").write_text(
        "schema=qsgw_hartree_pipeline_dump_v1\n"
        f"normalization={normalization}\n"
        "kpoint_count=2\n"
        "translation_count=2\n"
        "period=2 1 1\n"
        "atom_ao_sizes=0:1 1:1 \n"
        "density_delta_k_count=2\n"
        "hartree_k_atom_count=2\n"
        "hartree_r_atom_count=2\n"
        "density_delta_k_file=density_delta_k.txt\n"
        "density_delta_k_columns=kpoint row column real imag\n"
        "hartree_k_file=hartree_k.txt\n"
        "hartree_k_columns=atom_i atom_j kpoint row column real imag\n"
        "hartree_r_file=hartree_r.txt\n"
        "hartree_r_columns=atom_i atom_j R_x R_y R_z row column real imag\n",
        encoding="utf-8",
        newline="\n",
    )

    dr = [DR_K0.copy(), DR_K1.copy()]
    if not dr_hermitian:
        dr[0][0, 1] += 0.07j
    lines = ["# kpoint row column real imag"]
    for kpoint, matrix in enumerate(dr):
        for row in range(2):
            for column in range(2):
                lines.append(f"{kpoint} {row} {column} {_fmt(matrix[row, column])}")
    (dump_call / "density_delta_k.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8", newline="\n"
    )

    scale = hartree_k_scale
    norm_factor = 1.0 if normalization == "legacy_extra_inverse_nk" else 2.0
    d_values = {
        (0, 0): D00 * norm_factor * scale,
        (0, 1): D01 * norm_factor * scale,
        (1, 0): D10 * norm_factor * scale,
        (1, 1): D11 * norm_factor * scale,
    }
    d_values[(0, 1)] += hartree_k_perturb
    lines = ["# atom_i atom_j kpoint row column real imag"]
    for (atom_i, atom_j), value in d_values.items():
        for kpoint in range(2):
            lines.append(f"{atom_i} {atom_j} {kpoint} 0 0 {_fmt(value)}")
    (dump_call / "hartree_k.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8", newline="\n"
    )

    lines = ["# atom_i atom_j R_x R_y R_z row column real imag"]
    for (atom_i, atom_j), value in d_values.items():
        lines.append(f"{atom_i} {atom_j} -1 0 0 0 0 {_fmt(0.0j)}")
        lines.append(f"{atom_i} {atom_j} 0 0 0 0 0 {_fmt(value)}")
    (dump_call / "hartree_r.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8", newline="\n"
    )

    # Projected delta_vh picks up only half of a one-sided D01 perturbation
    # because of the 0.5*(P + P^dagger) Hermitian projection.
    dvh2 = {0: DVH2_K0 * norm_factor * scale, 1: DVH2_K1 * norm_factor * scale}
    dvh2[0] = dvh2[0].copy()
    dvh2[0][0, 1] += hartree_k_perturb / 2.0
    dvh2[0][1, 0] += hartree_k_perturb.conjugate() / 2.0
    dvh2[1] = dvh2[1].copy()
    dvh2[1][0, 1] += hartree_k_perturb / 2.0
    dvh2[1][1, 0] += hartree_k_perturb.conjugate() / 2.0

    lines = [
        "# qsgw_contract_version 5",
        "# iter channel component spin kpoint frequency_index frequency_Ha "
        "row column real_value imag_value",
    ]
    for iteration, occ in ((0, OCC0), (1, OCC1)):
        for kpoint in range(2):
            for band in range(2):
                lines.append(
                    f"{iteration} 0 occupation 0 {kpoint} -1 0.0 0 {band} "
                    f"{occ[kpoint][band]:.17e} 0.0"
                )
    for kpoint in range(2):
        for row in range(2):
            for column in range(2):
                value = dvh1 if row != column else 0.0j
                lines.append(
                    f"1 0 delta_vh 0 {kpoint} -1 0.0 {row} {column} {_fmt(value)}"
                )
    for kpoint in range(2):
        for row in range(2):
            for column in range(2):
                lines.append(
                    f"2 0 delta_vh 0 {kpoint} -1 0.0 {row} {column} "
                    f"{_fmt(dvh2[kpoint][row, column])}"
                )
    trace_path.write_text(
        "\n".join(lines) + "\n", encoding="utf-8", newline="\n"
    )

    return {"input": input_dir, "dump_call": dump_call, "trace": trace_path}


def _run_cli(fixture: dict[str, Path], *extra: str) -> tuple[int, dict]:
    output = fixture["dump_call"].parent.parent / "result.json"
    command = [
        sys.executable,
        str(MODULE),
        "--input-dir",
        str(fixture["input"]),
        "--dump-call",
        str(fixture["dump_call"]),
        "--trace",
        str(fixture["trace"]),
        "--iterations",
        "0:2",
        "--output",
        str(output),
    ] + list(extra)
    completed = subprocess.run(
        command, capture_output=True, text=True, cwd=fixture["input"].parent
    )
    payload = json.loads(output.read_text(encoding="utf-8"))
    return completed.returncode, payload


# ---------------------------------------------------------------------------
# Grid / phase conventions
# ---------------------------------------------------------------------------


class GridConventionTests(unittest.TestCase):
    def test_construct_r_grid_matches_cpp_order(self) -> None:
        self.assertEqual(construct_r_grid((1, 1, 1)), [(0, 0, 0)])
        self.assertEqual(
            construct_r_grid((2, 1, 1)), [(-1, 0, 0), (0, 0, 0)]
        )
        self.assertEqual(
            construct_r_grid((3, 1, 1)), [(-1, 0, 0), (0, 0, 0), (1, 0, 0)]
        )
        self.assertEqual(
            construct_r_grid((2, 2, 1)),
            [(-1, -1, 0), (-1, 0, 0), (0, -1, 0), (0, 0, 0)],
        )

    def test_fourier_phase_sign_convention(self) -> None:
        k = (0.25, 0.0, 0.0)
        self.assertAlmostEqual(
            fourier_phase(k, (1, 0, 0), 1.0), 1.0j, places=15
        )
        self.assertAlmostEqual(
            fourier_phase(k, (1, 0, 0), -1.0), -1.0j, places=15
        )
        self.assertAlmostEqual(
            fourier_phase(k, (-1, 0, 0), 1.0), -1.0j, places=15
        )
        self.assertAlmostEqual(
            fourier_phase((0.5, 0.0, 0.0), (-1, 0, 0), -1.0), -1.0, places=15
        )


# ---------------------------------------------------------------------------
# Frozen-input readers
# ---------------------------------------------------------------------------


class FrozenInputReaderTests(unittest.TestCase):
    def test_cs_text_roundtrip_and_ck_build(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "Cs_data_0.txt").write_text(
                "2 1\n"
                "1 1 0 0 0 1 1 2\n"
                "2.0 3.0\n"
                "1 1 1 0 0 1 1 2\n"
                "0.5 0.25\n",
                encoding="utf-8",
                newline="\n",
            )
            cs = read_cs_text(root)
            self.assertEqual(set(cs), {(0, 0)})
            self.assertEqual(set(cs[(0, 0)]), {(0, 0, 0), (1, 0, 0)})
            np.testing.assert_allclose(cs[(0, 0)][(0, 0, 0)], [[2.0, 3.0]])
            np.testing.assert_allclose(cs[(0, 0)][(1, 0, 0)], [[0.5, 0.25]])

            c_k = build_c_k(
                cs, {0: 1, 1: 1}, {0: 2, 1: 1}, [(0.0, 0.0, 0.0), (0.5, 0.0, 0.0)]
            )
            block = c_k[(0, 0)]
            self.assertEqual(block.shape, (2, 2, 1))
            # c_k(aux, orb) = sum_R exp(+2 pi i k.R) Cs(orb, aux)
            np.testing.assert_allclose(block[0, :, 0], [2.5, 3.25], atol=1e-14)
            np.testing.assert_allclose(block[1, :, 0], [1.5, 2.75], atol=1e-14)
            # pair (0,1) was not present in the file -> zero-filled like C++
            self.assertTrue(np.all(c_k[(0, 1)] == 0.0))

    def test_coulomb_gamma_and_hermitian_completion(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "coulomb_cut_0.txt").write_text(
                "2\n"
                "2 1 2 1 2 2 1.0\n"  # q_num=2, must be ignored
                "8.0 0.0 8.0 0.0 8.0 0.0 8.0 0.0\n"
                "2 1 2 1 2 1 1.0\n"  # q_num=1 == gamma
                f"{_fmt(V00_FILE)}\n"
                f"{_fmt(V01_FILE)}\n"
                f"{_fmt(V10_FILE_IGNORED)}\n"
                f"{_fmt(V11_FILE)}\n",
                encoding="utf-8",
                newline="\n",
            )
            v_full, zero_filled = read_coulomb_gamma_full(root, gamma_q_num=1)
            self.assertEqual(zero_filled, 0)
            v_q0 = build_v_q0(v_full, {0: 1, 1: 1})
            np.testing.assert_allclose(v_q0[(0, 0)], [[1.5 + 0.0j]], atol=1e-15)
            np.testing.assert_allclose(v_q0[(0, 1)], [[V01_FILE]], atol=1e-15)
            np.testing.assert_allclose(
                v_q0[(1, 0)], [[V01_FILE.conjugate()]], atol=1e-15
            )
            np.testing.assert_allclose(v_q0[(1, 1)], [[2.5 + 0.0j]], atol=1e-15)


# ---------------------------------------------------------------------------
# Check A: contraction
# ---------------------------------------------------------------------------


def _scalar_c_k() -> dict[tuple[int, int], np.ndarray]:
    values = {
        (0, 0): P_CS,
        (0, 1): Q_CS,
        (1, 0): R_CS,
        (1, 1): S_CS,
    }
    return {
        pair: np.full((2, 1, 1), value, dtype=np.complex128)
        for pair, value in values.items()
    }


def _scalar_density_blocks() -> dict[tuple[int, int], np.ndarray]:
    return {
        (0, 0): np.array([[[DR_K0[0, 0]]], [[DR_K1[0, 0]]]]),
        (0, 1): np.array([[[DR_K0[0, 1]]], [[DR_K1[0, 1]]]]),
        (1, 0): np.array([[[DR_K0[1, 0]]], [[DR_K1[1, 0]]]]),
        (1, 1): np.array([[[DR_K0[1, 1]]], [[DR_K1[1, 1]]]]),
    }


def _scalar_v_q0() -> dict[tuple[int, int], np.ndarray]:
    return {
        (0, 0): np.array([[1.5 + 0.0j]]),
        (0, 1): np.array([[V01_FILE]]),
        (1, 0): np.array([[V01_FILE.conjugate()]]),
        (1, 1): np.array([[2.5 + 0.0j]]),
    }


class ContractionTests(unittest.TestCase):
    def test_split_density_by_atom_blocks(self) -> None:
        m_k0 = np.array(
            [[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.complex128
        )
        m_k1 = m_k0 + 100.0
        density_k = np.stack([m_k0, m_k1])
        blocks = split_weighted_density_by_atom(density_k, {0: 1, 1: 2})
        self.assertEqual(blocks[(0, 0)].shape, (2, 1, 1))
        self.assertEqual(blocks[(0, 1)].shape, (2, 1, 2))
        self.assertEqual(blocks[(1, 1)].shape, (2, 2, 2))
        np.testing.assert_allclose(blocks[(0, 1)][0, 0, :], m_k0[0, 1:])
        np.testing.assert_allclose(blocks[(1, 0)][1, :, 0], m_k1[1:, 0])
        np.testing.assert_allclose(blocks[(1, 1)][0], m_k0[1:, 1:])

    def test_contraction_legacy_hand_computed(self) -> None:
        c_k = _scalar_c_k()
        density_blocks = _scalar_density_blocks()
        v_q0 = _scalar_v_q0()
        ao_sizes = {0: 1, 1: 1}
        aux_sizes = {0: 1, 1: 1}

        density_aux = compute_density_aux(
            c_k, density_blocks, ao_sizes, aux_sizes
        )
        np.testing.assert_allclose(density_aux[0], [DENSITY_AUX_0], atol=1e-12)
        np.testing.assert_allclose(density_aux[1], [DENSITY_AUX_1], atol=1e-12)

        potential_aux = compute_potential_aux(
            v_q0, density_aux, "legacy_extra_inverse_nk", 2
        )
        np.testing.assert_allclose(potential_aux[0], [POT_AUX_0], atol=1e-12)
        np.testing.assert_allclose(potential_aux[1], [POT_AUX_1], atol=1e-12)

        d_k = contract_hartree_full_grid(
            c_k, v_q0, density_blocks, ao_sizes, aux_sizes,
            "legacy_extra_inverse_nk",
        )
        for kpoint in range(2):
            np.testing.assert_allclose(d_k[(0, 0)][kpoint], [[D00]], atol=1e-10)
            np.testing.assert_allclose(d_k[(0, 1)][kpoint], [[D01]], atol=1e-10)
            np.testing.assert_allclose(d_k[(1, 0)][kpoint], [[D10]], atol=1e-10)
            np.testing.assert_allclose(d_k[(1, 1)][kpoint], [[D11]], atol=1e-10)

    def test_contraction_weighted_occupations_skips_inverse_nk(self) -> None:
        c_k = _scalar_c_k()
        density_blocks = _scalar_density_blocks()
        v_q0 = _scalar_v_q0()
        ao_sizes = {0: 1, 1: 1}
        aux_sizes = {0: 1, 1: 1}

        density_aux = compute_density_aux(
            c_k, density_blocks, ao_sizes, aux_sizes
        )
        potential_aux = compute_potential_aux(
            v_q0, density_aux, "weighted_occupations", 2
        )
        np.testing.assert_allclose(
            potential_aux[0], [2.0 * POT_AUX_0], atol=1e-12
        )
        np.testing.assert_allclose(
            potential_aux[1], [2.0 * POT_AUX_1], atol=1e-12
        )

        d_k = contract_hartree_full_grid(
            c_k, v_q0, density_blocks, ao_sizes, aux_sizes,
            "weighted_occupations",
        )
        np.testing.assert_allclose(d_k[(0, 0)][0], [[2.0 * D00]], atol=1e-10)
        np.testing.assert_allclose(d_k[(1, 1)][1], [[2.0 * D11]], atol=1e-10)

    def test_contraction_rejects_unknown_normalization(self) -> None:
        with self.assertRaises(RecomputeError):
            compute_potential_aux(
                _scalar_v_q0(),
                {0: np.array([1.0]), 1: np.array([1.0])},
                "plain",
                2,
            )


# ---------------------------------------------------------------------------
# Check B: inverse Fourier + BvK remap
# ---------------------------------------------------------------------------


class InverseFourierTests(unittest.TestCase):
    def test_inverse_fourier_hand_computed(self) -> None:
        # k0=(0,0,0), k1=(0.5,0,0); D(R) = (1/N_k) sum_k D_k exp(-2 pi i k.R)
        value_a = 2.0 + 1.0j
        value_b = 4.0 - 2.0j
        d_k = {
            (0, 0): np.array([[[value_a]], [[value_b]]], dtype=np.complex128)
        }
        result = inverse_fourier_hartree_operator(
            d_k,
            [(0.0, 0.0, 0.0), (0.5, 0.0, 0.0)],
            [(-1, 0, 0), (0, 0, 0)],
        )
        np.testing.assert_allclose(
            result[(0, 0, (0, 0, 0))], [[(value_a + value_b) / 2.0]], atol=1e-15
        )
        np.testing.assert_allclose(
            result[(0, 0, (-1, 0, 0))], [[(value_a - value_b) / 2.0]], atol=1e-15
        )

    def test_inverse_fourier_phase_sign_is_negative(self) -> None:
        # With k1=0.25: exp(-2 pi i * 0.25 * 1) = -i -> D(R=1) = (A - i B)/2
        value_a = 1.0 + 2.0j
        value_b = 3.0 - 1.0j
        d_k = {
            (0, 0): np.array([[[value_a]], [[value_b]]], dtype=np.complex128)
        }
        result = inverse_fourier_hartree_operator(
            d_k,
            [(0.0, 0.0, 0.0), (0.25, 0.0, 0.0)],
            [(1, 0, 0)],
        )
        np.testing.assert_allclose(
            result[(0, 0, (1, 0, 0))],
            [[(value_a - 1.0j * value_b) / 2.0]],
            atol=1e-15,
        )

    def test_inverse_fourier_bvk_remap_scatter(self) -> None:
        value_a = 2.0 + 0.0j
        value_b = 4.0 + 0.0j
        d_k = {
            (0, 0): np.array([[[value_a]], [[value_b]]], dtype=np.complex128)
        }
        remap = {(0, 0, (0, 0, 0)): [(0, 0, 0), (1, 0, 0)]}
        result = inverse_fourier_hartree_operator(
            d_k,
            [(0.0, 0.0, 0.0), (0.5, 0.0, 0.0)],
            [(0, 0, 0)],
            bvk_remap=remap,
        )
        expected = (value_a + value_b) / 2.0 / 2.0
        np.testing.assert_allclose(
            result[(0, 0, (0, 0, 0))], [[expected]], atol=1e-15
        )
        np.testing.assert_allclose(
            result[(0, 0, (1, 0, 0))], [[expected]], atol=1e-15
        )


# ---------------------------------------------------------------------------
# Check D: projection (standalone, hand-computed)
# ---------------------------------------------------------------------------


class ProjectionTests(unittest.TestCase):
    def test_projection_hand_computed(self) -> None:
        # Single non-zero R block at R=(0,0,0) -> operator_k = D (k-independent)
        hartree_r = {
            (0, 0, (0, 0, 0)): np.array([[D00]]),
            (0, 1, (0, 0, 0)): np.array([[D01]]),
            (1, 0, (0, 0, 0)): np.array([[D10]]),
            (1, 1, (0, 0, 0)): np.array([[D11]]),
            (0, 0, (-1, 0, 0)): np.array([[0.0j]]),
            (0, 1, (-1, 0, 0)): np.array([[0.0j]]),
            (1, 0, (-1, 0, 0)): np.array([[0.0j]]),
            (1, 1, (-1, 0, 0)): np.array([[0.0j]]),
        }
        wfc = {
            0: {
                0: np.eye(2, dtype=np.complex128),
                1: np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128),
            }
        }
        projected = project_periodic_operator_to_fixed_basis(
            hartree_r, {0: 1, 1: 1}, wfc, [(0.0, 0.0, 0.0), (0.5, 0.0, 0.0)]
        )
        np.testing.assert_allclose(projected[(0, 0)], DVH2_K0, atol=1e-10)
        np.testing.assert_allclose(projected[(0, 1)], DVH2_K1, atol=1e-10)


# ---------------------------------------------------------------------------
# Trace parsing
# ---------------------------------------------------------------------------


class TraceConventionTests(unittest.TestCase):
    def test_occupation_uses_row_zero_column_band(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trace.dat"
            path.write_text(
                "# header\n"
                "0 0 occupation 0 0 -1 0.0 0 0 0.5 0.0\n"
                "0 0 occupation 0 0 -1 0.0 0 1 0.25 0.0\n",
                encoding="utf-8",
                newline="\n",
            )
            occ = trace_occupations(path, {0})
            np.testing.assert_allclose(occ[(0, 0, 0)], [0.5, 0.25])

    def test_occupation_rejects_nonzero_row(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trace.dat"
            path.write_text(
                "0 0 occupation 0 0 -1 0.0 1 0 0.5 0.0\n",
                encoding="utf-8",
                newline="\n",
            )
            with self.assertRaises(RecomputeError):
                trace_occupations(path, {0})


# ---------------------------------------------------------------------------
# Manifest validation
# ---------------------------------------------------------------------------


class ManifestTests(unittest.TestCase):
    def test_manifest_rejects_bad_normalization(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            call = Path(tmp) / "call_001"
            call.mkdir()
            (call / "manifest.txt").write_text(
                "schema=qsgw_hartree_pipeline_dump_v1\n"
                "normalization=plain\n"
                "kpoint_count=1\n"
                "translation_count=1\n"
                "period=1 1 1\n"
                "atom_ao_sizes=0:1 \n"
                "density_delta_k_count=1\n"
                "hartree_k_atom_count=1\n"
                "hartree_r_atom_count=1\n",
                encoding="utf-8",
                newline="\n",
            )
            with self.assertRaisesRegex(RecomputeError, "normalization"):
                load_manifest(call)

    def test_manifest_rejects_wrong_schema(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            call = Path(tmp) / "call_001"
            call.mkdir()
            (call / "manifest.txt").write_text(
                "schema=other\n",
                encoding="utf-8",
                newline="\n",
            )
            with self.assertRaisesRegex(RecomputeError, "schema"):
                load_manifest(call)


# ---------------------------------------------------------------------------
# End-to-end CLI tests on the hand-computed fixture
# ---------------------------------------------------------------------------


class CliEndToEndTests(unittest.TestCase):
    def test_cli_end_to_end_passes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fixture = _write_fixture(Path(tmp))
            code, payload = _run_cli(fixture)
            self.assertEqual(code, 0, payload)
            self.assertTrue(payload["passed"])
            self.assertLess(payload["contraction"]["max_abs"], 1e-12)
            self.assertLess(payload["contraction"]["rel_frobenius"], 1e-12)
            self.assertLess(payload["inverse_fourier"]["max_abs"], 1e-12)
            self.assertLess(payload["inverse_fourier"]["rel_frobenius"], 1e-12)
            invariants = payload["density_invariants"]
            self.assertLess(invariants["hermiticity_max"], 1e-12)
            self.assertLess(invariants["per_k_trace_max_abs"], 1e-12)
            self.assertAlmostEqual(invariants["total_charge"], TOTAL_CHARGE, places=10)
            projection = payload["projection_end_to_end"]
            self.assertLess(projection["iteration2_max_abs"], 1e-12)
            self.assertLess(projection["iteration2_rel_frobenius"], 1e-12)
            self.assertLess(projection["iteration1_max_abs"], 1e-12)
            for section in (
                "contraction",
                "inverse_fourier",
                "density_invariants",
                "projection_end_to_end",
            ):
                self.assertTrue(payload[section]["passed"], section)

    def test_cli_weighted_occupations_fixture_passes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fixture = _write_fixture(
                Path(tmp), normalization="weighted_occupations"
            )
            code, payload = _run_cli(fixture)
            self.assertEqual(code, 0, payload)
            self.assertTrue(payload["passed"])

    def test_cli_contraction_mismatch_fails_exit2(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fixture = _write_fixture(Path(tmp), hartree_k_perturb=1.0e-5 + 0.0j)
            code, payload = _run_cli(fixture)
            self.assertEqual(code, 2, payload)
            self.assertFalse(payload["passed"])
            self.assertFalse(payload["contraction"]["passed"])
            # inverse Fourier reads the dump hartree_k independently: it still
            # passes because dump hartree_k and dump hartree_r stay consistent.
            self.assertTrue(payload["inverse_fourier"]["passed"])
            self.assertTrue(payload["projection_end_to_end"]["passed"])

    def test_cli_iteration_one_nonzero_delta_vh_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fixture = _write_fixture(Path(tmp), dvh1=1.0e-4 + 0.0j)
            code, payload = _run_cli(fixture)
            self.assertEqual(code, 2, payload)
            projection = payload["projection_end_to_end"]
            self.assertFalse(projection["passed"])
            self.assertGreater(projection["iteration1_max_abs"], 1e-5)
            self.assertLess(projection["iteration2_max_abs"], 1e-12)

    def test_cli_nonhermitian_density_fails_invariants(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fixture = _write_fixture(Path(tmp), dr_hermitian=False)
            code, payload = _run_cli(fixture)
            self.assertEqual(code, 2, payload)
            self.assertFalse(payload["density_invariants"]["passed"])
            self.assertGreater(
                payload["density_invariants"]["hermiticity_max"], 0.01
            )


class CliFailClosedTests(unittest.TestCase):
    def _expect_exit1(self, fixture: dict[str, Path]) -> dict:
        code, payload = _run_cli(fixture)
        self.assertEqual(code, 1, payload)
        self.assertFalse(payload["passed"])
        self.assertIn("error", payload)
        return payload

    def test_cli_missing_manifest_exit1(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fixture = _write_fixture(Path(tmp))
            (fixture["dump_call"] / "manifest.txt").unlink()
            self._expect_exit1(fixture)

    def test_cli_bad_normalization_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fixture = _write_fixture(Path(tmp), normalization="plain")
            self._expect_exit1(fixture)

    def test_cli_duplicate_density_row_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fixture = _write_fixture(Path(tmp))
            path = fixture["dump_call"] / "density_delta_k.txt"
            text = path.read_text(encoding="utf-8")
            path.write_text(
                text + "0 0 0 1.0 0.0\n", encoding="utf-8", newline="\n"
            )
            self._expect_exit1(fixture)

    def test_cli_missing_density_row_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fixture = _write_fixture(Path(tmp))
            path = fixture["dump_call"] / "density_delta_k.txt"
            lines = path.read_text(encoding="utf-8").splitlines()
            path.write_text(
                "\n".join(lines[:-1]) + "\n", encoding="utf-8", newline="\n"
            )
            self._expect_exit1(fixture)

    def test_cli_nonfinite_value_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fixture = _write_fixture(Path(tmp))
            path = fixture["dump_call"] / "hartree_k.txt"
            text = path.read_text(encoding="utf-8").splitlines()
            text[1] = "0 0 0 0 0 nan 0.0"
            path.write_text(
                "\n".join(text) + "\n", encoding="utf-8", newline="\n"
            )
            self._expect_exit1(fixture)

    def test_cli_missing_column_hartree_k_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fixture = _write_fixture(Path(tmp))
            path = fixture["dump_call"] / "hartree_k.txt"
            lines = path.read_text(encoding="utf-8").splitlines()
            lines[1] = "0 0 0 0 0 1.0"
            path.write_text(
                "\n".join(lines) + "\n", encoding="utf-8", newline="\n"
            )
            self._expect_exit1(fixture)

    def test_cli_malformed_trace_row_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fixture = _write_fixture(Path(tmp))
            path = fixture["trace"]
            path.write_text(
                path.read_text(encoding="utf-8")
                + "2 0 delta_vh 0 0 -1 0.0 0 0 1.0\n",
                encoding="utf-8",
                newline="\n",
            )
            self._expect_exit1(fixture)

    def test_cli_iterations_window_too_narrow_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fixture = _write_fixture(Path(tmp))
            output = fixture["dump_call"].parent.parent / "result.json"
            command = [
                sys.executable,
                str(MODULE),
                "--input-dir",
                str(fixture["input"]),
                "--dump-call",
                str(fixture["dump_call"]),
                "--trace",
                str(fixture["trace"]),
                "--iterations",
                "0:1",
                "--output",
                str(output),
            ]
            completed = subprocess.run(
                command, capture_output=True, text=True, cwd=fixture["input"].parent
            )
            self.assertEqual(completed.returncode, 1)
            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertFalse(payload["passed"])
            self.assertIn("error", payload)


if __name__ == "__main__":
    unittest.main()
