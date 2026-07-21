#!/usr/bin/env python3
"""Focused tests for the QSGW Hamiltonian trace closure observer."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from validate_qsgw_trace_closure_v3 import (  # noqa: E402
    TraceClosureError,
    _materialize_upper_triangle_hermitian,
    validate_trace_text,
)


def _rows(iteration: int, component: str, matrix: np.ndarray) -> list[str]:
    result = []
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix[row, column]
            result.append(
                f"{iteration} 0 {component} 0 0 -1 0.0 {row} {column} "
                f"{value.real:.17e} {value.imag:.17e}"
            )
    return result


def _trace(
    raw_offset: complex = 0.0j,
    *,
    stale_effective_lower: bool = False,
) -> str:
    h0 = np.array([[1.0, 0.1 + 0.2j], [0.1 - 0.2j, 2.0]])
    vxc = np.array([[0.3, 0.04 - 0.03j], [0.04 + 0.03j, 0.5]])
    # Lower triangles deliberately differ. Scheme A reads only UPLO='U'.
    exx = np.array([[-0.2, 0.07 + 0.01j], [0.8 - 0.9j, -0.4]])
    vc = np.array([[0.02, -0.03 + 0.05j], [0.6 + 0.7j, 0.04]])
    unmaterialized = h0 - vxc + exx + vc
    raw = _materialize_upper_triangle_hermitian(unmaterialized)
    raw[0, 1] += raw_offset
    raw[1, 0] = (
        unmaterialized[1, 0]
        if stale_effective_lower
        else raw[0, 1].conjugate()
    )

    lines = [
        "# qsgw_contract_version 5",
        "# fixed_basis immutable_mf0",
        "# live_update eigenvalues_wfc",
        "# velocity disabled_stage1",
        "# headwing disabled_stage1",
        "# symmetry unsupported_full_bz_only",
        "# hartree disabled_stage1",
        "# band disabled_stage1",
        "# qsgw_input_contract synthetic.contract",
        f"# qsgw_input_contract_sha256 {'0' * 64}",
        "# qsgw_mixer none",
        "# qsgw_mixing_beta 2.00000000000000011e-01",
        "# iter channel component spin kpoint frequency_index frequency_Ha "
        "row column real_value imag_value",
    ]
    for component, matrix in (("h0", h0), ("vxc_dft", vxc)):
        lines.extend(_rows(0, component, matrix))
    for component, matrix in (
        ("exx", exx),
        ("vc", vc),
        ("raw_h", raw),
        ("mixed_h", raw),
    ):
        lines.extend(_rows(1, component, matrix))
    return "\n".join(lines) + "\n"


def _legacy_trace(raw_offset: complex = 0.0j) -> str:
    h0 = np.array([[1.0, 0.1 + 0.2j], [0.1 - 0.2j, 2.0]])
    vxc = np.array([[0.3, 0.04 - 0.03j], [0.04 + 0.03j, 0.5]])
    exx = np.array([[-0.2, 0.07 + 0.01j], [0.8 - 0.9j, -0.4]])
    vc = np.array([[0.02, -0.03 + 0.05j], [0.6 + 0.7j, 0.04]])
    raw = h0 - vxc + exx + vc
    raw[0, 1] += raw_offset

    lines = [
        "# qsgw_contract_version 4",
        "# task qsgw",
        "# fixed_basis immutable_reference",
        "# qsgw_mixer linear",
        "# qsgw_mixing_beta 1",
        "# qsgw_min_iter 1",
        "# qsgw_max_iter 1",
        "# starting_vxc dft_only",
        "# vxc_basis fixed_state",
        "# qsgw_update_hartree 0",
        "# qsgw_hartree_coulomb truncated",
        "# qsgw_hartree_normalization legacy_extra_inverse_nk",
        "# use_symmetry_gw 0",
        "# use_symmetry_exx 0",
        "# replace_w_head 0",
        "# option_dielect_func 0",
        "# nfreq 6",
        "# n_params_anacon 6",
        "# n_params_anacon_resample -1",
        "# anacon_nfreq -1",
        "# anacon_tfgrids_type -101",
        "# use_shrink_abfs 0",
        "# use_fullcoul_exx 0",
        "# use_fullcoul_eps 1",
        "# use_fullcoul_wc 0",
        "# constants_choice internal",
        "# ac_policy direct_pade",
        "# iter channel component spin kpoint frequency_index frequency_Ha "
        "row column real_value imag_value",
    ]
    for component, matrix in (("h0", h0), ("vxc_dft", vxc)):
        lines.extend(_rows(0, component, matrix))
    for component, matrix in (
        ("exx", exx),
        ("vc", vc),
        ("raw_h", raw),
        ("mixed_h", raw),
    ):
        lines.extend(_rows(1, component, matrix))
    return "\n".join(lines) + "\n"


class TraceClosureTests(unittest.TestCase):
    def test_upper_triangle_authoritative_trace_passes(self) -> None:
        report = validate_trace_text(_trace(), (0, 1))
        self.assertTrue(report["passed"])
        self.assertLess(report["effective_hamiltonian_max_abs_ha"], 1.0e-14)

    def test_tampered_raw_h_is_rejected(self) -> None:
        with self.assertRaisesRegex(
            TraceClosureError, "effective Hamiltonian closure failed"
        ):
            validate_trace_text(_trace(1.0e-4 + 2.0e-4j), (0, 1))

    def test_legacy_effective_hamiltonian_upper_triangle_passes(self) -> None:
        report = validate_trace_text(
            _legacy_trace(), (0, 1), require_current_contract=False
        )
        self.assertTrue(report["passed"])
        self.assertEqual(
            report["effective_hamiltonian_semantics"],
            "legacy_upper_triangle_authoritative",
        )

    def test_legacy_tampered_authoritative_upper_triangle_is_rejected(self):
        with self.assertRaisesRegex(
            TraceClosureError, "effective Hamiltonian closure failed"
        ):
            validate_trace_text(
                _legacy_trace(1.0e-4 + 2.0e-4j),
                (0, 1),
                require_current_contract=False,
            )

    def test_current_nonhermitian_effective_hamiltonian_is_rejected(self):
        with self.assertRaisesRegex(TraceClosureError, "not Hermitian"):
            validate_trace_text(
                _trace(stale_effective_lower=True), (0, 1)
            )


if __name__ == "__main__":
    unittest.main()
