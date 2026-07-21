#!/usr/bin/env python3

from __future__ import annotations

import sys
import tempfile
import unittest
import os
from pathlib import Path

import numpy as np


REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "regression_tests" / "backend" / "comparisons"))

from validate_qsgw_band_v6 import BandValidationError, HA2EV, validate_band_run


MATRIX_HEADER = (
    "# iter channel component spin kpoint frequency_index frequency_Ha "
    "row column real_value imag_value"
)
EIGEN_HEADER = "# iter channel spin kpoint kx ky kz band energy_eV"
SUMMARY_HEADER = (
    "# iter max_delta_eV residual_l2_Ha residual_max_Ha efermi_eV gap_eV "
    "electron_count requested_mode applied_mode beta fallback rcond "
    "coefficient_l1 coefficient_count converged coefficients fallback_reason"
)


def contract(mode, hartree=False, mixer="none", beta=0.2):
    hartree_contract = (
        [
            "# hartree delta_density",
            "# hartree_coulomb full",
            "# hartree_normalization weighted_occupations",
        ]
        if hartree
        else ["# hartree disabled_stage1"]
    )
    return "\n".join(
        [
            "# qsgw_contract_version 6",
            "# fixed_basis immutable_mf0",
            "# live_update eigenvalues_wfc",
            "# velocity disabled_stage1",
            "# headwing disabled_stage1",
            "# symmetry exx_on_gw_on_rpa_on",
            *hartree_contract,
            "# band fixed_reference_operator_fourier_live",
            "# h_qsgw_cut band_postprocess",
            "# qsgw_band0_unoccupied_keep 0",
            "# qsgw_band0_cut_mode {}".format(mode),
            "# qsgw_band0_cut_shift_ha 20",
            "# qsgw_input_contract dataset/qsgw_band_input.contract",
            "# qsgw_input_contract_sha256 " + "a" * 64,
            "# qsgw_mixer {}".format(mixer),
            "# qsgw_mixing_beta {:.17g}".format(beta),
        ]
    )


def matrix_rows(iteration, channel, component, matrix, frequency_index=-1,
                frequency=0.0):
    rows = []
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix[row, column]
            rows.append(
                "{} {} {} 0 0 {} {:.17e} {} {} {:.17e} {:.17e}".format(
                    iteration,
                    channel,
                    component,
                    frequency_index,
                    frequency,
                    row,
                    column,
                    value.real,
                    value.imag,
                )
            )
    return rows


def table_row(occupations, energies):
    fields = ["    1", "      0.0000000", "      0.0000000", "      0.0000000"]
    for occupation, energy in zip(occupations, energies):
        fields.extend(["{:15.5f}".format(occupation), "{:15.5f}".format(energy)])
    return "".join(fields) + "\n"


def make_fixture(root, mode=0, diagnostic=0.0, corrupt_csr=False,
                 corrupt_wavefunction=False, hartree=False, mixer="none",
                 beta=0.2, legacy_mixing_order=False):
    identity = np.eye(2, dtype=np.complex128)
    occupation = np.asarray([[1.0, 0.0]], dtype=np.complex128)
    h0 = np.diag([-1.0, 1.0]).astype(np.complex128)
    vxc = np.diag([0.1, 0.2]).astype(np.complex128)
    exx = np.diag([-0.2, -0.1]).astype(np.complex128)
    vc = np.diag([0.05, 0.06]).astype(np.complex128)
    delta_vh = (
        np.diag([0.03, 0.04]).astype(np.complex128)
        if hartree
        else np.zeros_like(h0)
    )
    uncut = h0 - vxc + exx + vc + delta_vh
    raw = uncut.copy()
    initial = h0.copy()
    if mode in (1, 2):
        raw[0, 1] = raw[1, 0] = 0.0
        shift = 20.0 if mode == 2 else 0.0
        raw[1, 1] = h0[1, 1] + shift
        initial[1, 1] = h0[1, 1] + shift

    mixed = raw.copy()
    if mixer == "linear":
        mixed = initial + beta * (raw - initial)
        if legacy_mixing_order and mode == 2:
            legacy_initial = h0
            mixed = legacy_initial + beta * (raw - legacy_initial)
        elif mode in (1, 2):
            shift = 20.0 if mode == 2 else 0.0
            mixed[0, 1] = mixed[1, 0] = 0.0
            mixed[1, 1] = h0[1, 1] + shift

    matrix_lines = [contract(mode, hartree, mixer, beta), MATRIX_HEADER]
    for channel in (0, 1):
        matrix_lines += matrix_rows(0, channel, "h0", h0)
        matrix_lines += matrix_rows(0, channel, "vxc_dft", vxc)
        matrix_lines += matrix_rows(0, channel, "wfc_spinor0", identity)
        matrix_lines += matrix_rows(0, channel, "occupation", occupation)
    matrix_lines += matrix_rows(1, 0, "sigma_c_iw", vc, 0, 1.0)
    for channel in (0, 1):
        matrix_lines += matrix_rows(1, channel, "exx", exx)
        matrix_lines += matrix_rows(1, channel, "vc", vc)
        if hartree:
            matrix_lines += matrix_rows(1, channel, "delta_vh", delta_vh)
        matrix_lines += matrix_rows(1, channel, "raw_h", raw)
        matrix_lines += matrix_rows(1, channel, "mixed_h", mixed)
        matrix_lines += matrix_rows(1, channel, "rotation_u", identity)
        live_wavefunction = identity.copy()
        if corrupt_wavefunction and channel == 1:
            live_wavefunction[0, 0] += 0.01
        matrix_lines += matrix_rows(
            1, channel, "wfc_spinor0", live_wavefunction
        )
        matrix_lines += matrix_rows(1, channel, "occupation", occupation)
    scalar_values = {
        "basis_inverse_residual": diagnostic,
        "basis_condition_estimate": 1.0,
        "fourier_orthogonality_residual": 0.0,
        "source_roundtrip_relative_error": 0.0,
        "target_hermiticity_error": 0.0,
        "target_relative_hermiticity_error": 0.0,
        "repaired_target_hermiticity_error": 0.0,
    }
    for component, value in scalar_values.items():
        matrix_lines += matrix_rows(
            1, 1, component, np.asarray([[value]], dtype=np.complex128)
        )
    matrix_path = root / "qsgw_matrices.dat"
    matrix_path.write_text("\n".join(matrix_lines) + "\n", encoding="ascii")

    eigen_lines = [contract(mode, hartree, mixer, beta), EIGEN_HEADER]
    for iteration, matrix in ((0, h0), (1, mixed)):
        for channel in (0, 1):
            for band, energy in enumerate(np.diag(matrix).real * HA2EV):
                eigen_lines.append(
                    "{} {} 0 0 0 0 0 {} {:.17e}".format(
                        iteration, channel, band, energy
                    )
                )
    eigen_path = root / "qsgw_eigenvalues.dat"
    eigen_path.write_text("\n".join(eigen_lines) + "\n", encoding="ascii")

    residual = raw - initial
    residual_l2 = float(np.linalg.norm(residual))
    residual_maximum = float(np.max(np.abs(residual)))
    summary_lines = [
        contract(mode, hartree, mixer, beta),
        SUMMARY_HEADER,
        "0 0 0 0 0 2 1 -1 -1 1 0 0 0 0 0 none none",
        "1 1 {:.17e} {:.17e} 0 2 1 -1 -1 1 0 0 0 0 0 none none".format(
            residual_l2, residual_maximum
        ),
    ]
    summary_path = root / "qsgw_iterations.dat"
    summary_path.write_text("\n".join(summary_lines) + "\n", encoding="ascii")

    ks_energies = np.diag(h0).real * HA2EV
    exx_energies = np.diag(h0 - vxc + exx).real * HA2EV
    qsgw_energies = np.diag(mixed).real * HA2EV
    (root / "KS_band_spin_1_1.dat").write_text(
        table_row([1.0, 0.0], ks_energies), encoding="ascii"
    )
    (root / "EXX_band_spin_1_1.dat").write_text(
        table_row([1.0, 0.0], exx_energies), encoding="ascii"
    )
    (root / "QSGW_band_spin_1_1.dat").write_text(
        table_row([1.0, 0.0], qsgw_energies), encoding="ascii"
    )

    csr_matrix = mixed.copy()
    if corrupt_csr:
        csr_matrix[0, 0] += 0.01
    values = (2.0 * csr_matrix.real).reshape(-1)
    csr = "\n".join(
        [
            "STEP: 0",
            "Matrix Dimension of H(R): 2",
            "Matrix number of H(R): 1",
            "0 0 0 4",
            " " + " ".join("{:.16e}".format(value) for value in values),
            " 0 1 0 1",
            " 0 2 4",
        ]
    ) + "\n"
    (root / "hrs1_nao_qsgw_iter_0001.csr").write_text(csr, encoding="ascii")
    bz_sampling = root / "bz_sampling_out"
    bz_sampling.write_text("1 1 1 1 1\n", encoding="ascii")
    return matrix_path, eigen_path, summary_path, bz_sampling


class BandValidatorTests(unittest.TestCase):
    def run_fixture(self, mode=0, diagnostic=0.0, corrupt_csr=False,
                    corrupt_wavefunction=False, hartree=False, mixer="none",
                    beta=0.2, legacy_mixing_order=False):
        fixed_root = os.environ.get("LIBRPA_QSGW_TEST_TMP")
        if fixed_root:
            root = Path(fixed_root)
            if not root.is_dir():
                raise RuntimeError("LIBRPA_QSGW_TEST_TMP must already exist")
        else:
            temporary = tempfile.TemporaryDirectory()
            self.addCleanup(temporary.cleanup)
            root = Path(temporary.name)
        paths = make_fixture(
            root, mode, diagnostic, corrupt_csr, corrupt_wavefunction,
            hartree, mixer, beta, legacy_mixing_order
        )
        return validate_band_run(
            *paths[:3],
            root,
            paths[3],
            expected_iterations=1,
            expected_cut_mode=mode,
            expected_unoccupied_keep=0,
            expected_shift_ha=20.0,
        )

    def test_uncut_run_closes_through_band_tables_and_csr(self):
        report = self.run_fixture(mode=0)
        self.assertTrue(report["passed"])
        self.assertEqual(report["csr_exports"][0]["dimension"], 2)

    def test_shifted_cut_run_closes(self):
        for mode in (1, 2):
            with self.subTest(mode=mode):
                report = self.run_fixture(mode=mode)
                self.assertTrue(report["passed"])
                self.assertEqual(report["active_limits"]["1:1:0:0"], 1)

    def test_hartree_delta_is_included_in_grid_and_band_raw_closure(self):
        report = self.run_fixture(mode=0, hartree=True)
        self.assertTrue(report["passed"])
        self.assertEqual(report["contract"]["hartree"], "delta_density")
        self.assertLessEqual(report["raw_closure_max_abs_ha"], 1.0e-14)

    def test_linear_mixing_reapplies_shifted_cut(self):
        report = self.run_fixture(mode=2, mixer="linear", beta=0.2)
        self.assertTrue(report["passed"])
        self.assertLessEqual(report["mixing_closure_max_abs_ha"], 1.0e-14)

    def test_linear_mixing_rejects_legacy_uncut_initialization_order(self):
        with self.assertRaisesRegex(BandValidationError, "mix closure"):
            self.run_fixture(
                mode=2,
                mixer="linear",
                beta=0.2,
                legacy_mixing_order=True,
            )

    def test_fourier_diagnostic_failure_is_not_hidden(self):
        with self.assertRaisesRegex(BandValidationError, "basis_inverse_residual"):
            self.run_fixture(diagnostic=1.0e-5)

    def test_csr_roundtrip_failure_is_not_hidden(self):
        with self.assertRaisesRegex(BandValidationError, "CSR roundtrip"):
            self.run_fixture(corrupt_csr=True)

    def test_fixed_basis_wavefunction_failure_is_not_hidden(self):
        with self.assertRaisesRegex(
            BandValidationError, "fixed-basis wavefunction rotation"
        ):
            self.run_fixture(corrupt_wavefunction=True)


if __name__ == "__main__":
    unittest.main()
