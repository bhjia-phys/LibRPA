#!/usr/bin/env python3
"""Focused tests for the QSGW Hamiltonian trace closure observer v4.

Revision v4 accepts current (stage-one, contract version 5) traces that
carry an enabled Hartree delta-density update, instead of hard-rejecting
any current trace whose ``hartree`` header is not ``disabled_stage1``.
The Hartree closure logic itself (iteration-one zero reference,
post-reference response, delta_vh in the closure and Hermiticity checks)
already existed for legacy traces and is locked here for both contracts.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from validate_qsgw_trace_closure_v4 import (  # noqa: E402
    TraceClosureError,
    _materialize_upper_triangle_hermitian,
    validate_trace_text,
)


BETA = 0.2


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


H0 = np.array([[1.0, 0.1 + 0.2j], [0.1 - 0.2j, 2.0]])
VXC = np.array([[0.3, 0.04 - 0.03j], [0.04 + 0.03j, 0.5]])
EXX = np.array([[-0.2, 0.07 + 0.01j], [0.8 - 0.9j, -0.4]])
VC = np.array([[0.02, -0.03 + 0.05j], [0.6 + 0.7j, 0.04]])
DVH2 = np.array([[0.05, 0.01 + 0.02j], [0.01 - 0.02j, -0.03]])


def _current_hartree_trace(
    *,
    dvh1_offset: complex = 0.0j,
    drop_dvh_in_raw: bool = False,
    skip_dvh2_rows: bool = False,
    nonhermitian_dvh2: bool = False,
    hartree_mode: str = "delta_density",
    coulomb: str | None = "truncated",
    normalization: str | None = "legacy_extra_inverse_nk",
) -> str:
    dvh1 = np.zeros((2, 2), dtype=np.complex128)
    dvh1[0, 1] += dvh1_offset
    dvh1[1, 0] += dvh1_offset.conjugate()
    dvh2 = np.array(DVH2, dtype=np.complex128)
    if nonhermitian_dvh2:
        dvh2[1, 0] += 0.07 + 0.11j

    raw1 = _materialize_upper_triangle_hermitian(H0 - VXC + EXX + VC + dvh1)
    raw2_sum = H0 - VXC + EXX + VC + (0.0 * dvh2 if drop_dvh_in_raw else dvh2)
    raw2 = _materialize_upper_triangle_hermitian(raw2_sum)
    mixed1 = (1.0 - BETA) * H0 + BETA * raw1
    mixed2 = (1.0 - BETA) * mixed1 + BETA * raw2

    lines = [
        "# qsgw_contract_version 5",
        "# fixed_basis immutable_mf0",
        "# live_update eigenvalues_wfc",
        "# velocity disabled_stage1",
        "# headwing disabled_stage1",
        "# symmetry unsupported_full_bz_only",
        f"# hartree {hartree_mode}",
    ]
    if coulomb is not None:
        lines.append(f"# hartree_coulomb {coulomb}")
    if normalization is not None:
        lines.append(f"# hartree_normalization {normalization}")
    lines += [
        "# band disabled_stage1",
        "# qsgw_input_contract synthetic.contract",
        f"# qsgw_input_contract_sha256 {'0' * 64}",
        "# qsgw_mixer linear",
        "# qsgw_mixing_beta 2.00000000000000011e-01",
        "# iter channel component spin kpoint frequency_index frequency_Ha "
        "row column real_value imag_value",
    ]
    for component, matrix in (("h0", H0), ("vxc_dft", VXC)):
        lines.extend(_rows(0, component, matrix))
    for component, matrix in (
        ("exx", EXX),
        ("vc", VC),
        ("delta_vh", dvh1),
        ("raw_h", raw1),
        ("mixed_h", mixed1),
    ):
        lines.extend(_rows(1, component, matrix))
    for component, matrix in (
        ("exx", EXX),
        ("vc", VC),
        ("raw_h", raw2),
        ("mixed_h", mixed2),
    ):
        lines.extend(_rows(2, component, matrix))
    if not skip_dvh2_rows:
        lines.extend(_rows(2, "delta_vh", dvh2))
    return "\n".join(lines) + "\n"


def _legacy_hartree_trace(*, dvh1_offset: complex = 0.0j) -> str:
    dvh1 = np.zeros((2, 2), dtype=np.complex128)
    dvh1[0, 1] += dvh1_offset
    dvh1[1, 0] += dvh1_offset.conjugate()
    raw1 = H0 - VXC + EXX + VC + dvh1
    raw2 = H0 - VXC + EXX + VC + DVH2

    lines = [
        "# qsgw_contract_version 4",
        "# task qsgw",
        "# fixed_basis immutable_reference",
        "# qsgw_mixer linear",
        "# qsgw_mixing_beta 1",
        "# qsgw_min_iter 2",
        "# qsgw_max_iter 2",
        "# starting_vxc dft_only",
        "# vxc_basis fixed_state",
        "# qsgw_update_hartree 1",
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
    for component, matrix in (("h0", H0), ("vxc_dft", VXC)):
        lines.extend(_rows(0, component, matrix))
    for component, matrix in (
        ("exx", EXX),
        ("vc", VC),
        ("delta_vh", dvh1),
        ("raw_h", raw1),
        ("mixed_h", raw1),
    ):
        lines.extend(_rows(1, component, matrix))
    for component, matrix in (
        ("exx", EXX),
        ("vc", VC),
        ("delta_vh", DVH2),
        ("raw_h", raw2),
        ("mixed_h", raw2),
    ):
        lines.extend(_rows(2, component, matrix))
    return "\n".join(lines) + "\n"


class CurrentHartreeClosureTests(unittest.TestCase):
    def test_delta_density_trace_passes(self) -> None:
        report = validate_trace_text(
            _current_hartree_trace(),
            (0, 1, 2),
            hartree_response_minimum_ha=1.0e-3,
        )
        self.assertTrue(report["passed"])
        self.assertTrue(report["hartree_enabled"])
        self.assertEqual(report["hartree_iteration_one_max_abs_ha"], 0.0)
        self.assertGreater(report["hartree_post_reference_max_abs_ha"], 1.0e-3)
        self.assertLess(report["effective_hamiltonian_max_abs_ha"], 1.0e-14)
        self.assertLess(report["linear_mixing_max_abs_ha"], 1.0e-14)

    def test_iteration_one_nonzero_delta_vh_is_rejected(self) -> None:
        with self.assertRaisesRegex(
            TraceClosureError, "iteration-one Hartree delta does not vanish"
        ):
            validate_trace_text(
                _current_hartree_trace(dvh1_offset=0.01 + 0.02j), (0, 1, 2)
            )

    def test_response_below_minimum_is_rejected(self) -> None:
        with self.assertRaisesRegex(
            TraceClosureError, "below the required minimum"
        ):
            validate_trace_text(
                _current_hartree_trace(),
                (0, 1, 2),
                hartree_response_minimum_ha=10.0,
            )

    def test_missing_delta_vh_at_update_is_rejected(self) -> None:
        with self.assertRaisesRegex(
            TraceClosureError, "missing delta_vh at iteration 2"
        ):
            validate_trace_text(
                _current_hartree_trace(skip_dvh2_rows=True), (0, 1, 2)
            )

    def test_closure_requires_delta_vh_contribution(self) -> None:
        with self.assertRaisesRegex(
            TraceClosureError, "effective Hamiltonian closure failed"
        ):
            validate_trace_text(
                _current_hartree_trace(drop_dvh_in_raw=True), (0, 1, 2)
            )

    def test_nonhermitian_delta_vh_is_rejected(self) -> None:
        with self.assertRaisesRegex(TraceClosureError, "not Hermitian"):
            validate_trace_text(
                _current_hartree_trace(nonhermitian_dvh2=True), (0, 1, 2)
            )

    def test_missing_coulomb_contract_is_rejected(self) -> None:
        with self.assertRaisesRegex(TraceClosureError, "hartree_coulomb"):
            validate_trace_text(
                _current_hartree_trace(coulomb=None), (0, 1, 2)
            )

    def test_unsupported_coulomb_mode_is_rejected(self) -> None:
        with self.assertRaisesRegex(TraceClosureError, "hartree_coulomb"):
            validate_trace_text(
                _current_hartree_trace(coulomb="full"), (0, 1, 2)
            )

    def test_missing_normalization_contract_is_rejected(self) -> None:
        with self.assertRaisesRegex(TraceClosureError, "hartree_normalization"):
            validate_trace_text(
                _current_hartree_trace(normalization=None), (0, 1, 2)
            )

    def test_unsupported_normalization_is_rejected(self) -> None:
        with self.assertRaisesRegex(TraceClosureError, "hartree_normalization"):
            validate_trace_text(
                _current_hartree_trace(normalization="plain"), (0, 1, 2)
            )

    def test_disabled_stage1_with_delta_vh_is_rejected(self) -> None:
        with self.assertRaisesRegex(
            TraceClosureError, "unexpectedly contains delta_vh"
        ):
            validate_trace_text(
                _current_hartree_trace(
                    hartree_mode="disabled_stage1",
                    coulomb=None,
                    normalization=None,
                ),
                (0, 1, 2),
            )

    def test_disabled_stage1_with_hartree_options_is_rejected(self) -> None:
        with self.assertRaisesRegex(
            TraceClosureError, "unexpectedly declares"
        ):
            validate_trace_text(
                _current_hartree_trace(hartree_mode="disabled_stage1"),
                (0, 1, 2),
            )


class LegacyHartreeClosureTests(unittest.TestCase):
    def test_legacy_delta_density_trace_passes(self) -> None:
        report = validate_trace_text(
            _legacy_hartree_trace(),
            (0, 1, 2),
            require_current_contract=False,
            hartree_response_minimum_ha=1.0e-3,
        )
        self.assertTrue(report["passed"])
        self.assertTrue(report["hartree_enabled"])
        self.assertEqual(report["hartree_iteration_one_max_abs_ha"], 0.0)

    def test_legacy_iteration_one_nonzero_delta_vh_is_rejected(self) -> None:
        with self.assertRaisesRegex(
            TraceClosureError, "iteration-one Hartree delta does not vanish"
        ):
            validate_trace_text(
                _legacy_hartree_trace(dvh1_offset=0.01 + 0.02j),
                (0, 1, 2),
                require_current_contract=False,
            )


if __name__ == "__main__":
    unittest.main()
