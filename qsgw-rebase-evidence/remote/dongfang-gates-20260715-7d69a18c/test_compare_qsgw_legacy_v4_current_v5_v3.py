#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


HA2EV = 27.211386245988
HERE = Path(__file__).resolve().parent
MODULE_PATH = HERE / "compare_qsgw_component_traces_v3.py"
SPEC = importlib.util.spec_from_file_location("qsgw_trace_compare", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def row(
    iteration: int,
    component: str,
    value: float,
    *,
    frequency_index: int = -1,
    frequency_ha: float = 0.0,
) -> str:
    return (
        f"{iteration} 0 {component} 0 0 {frequency_index} "
        f"{frequency_ha:.17e} 0 0 {value:.17e} 0.0"
    )


def legacy_header(
    *,
    use_fullcoul_exx: int = 1,
    declared_final_iteration: int = 1,
    mixing_beta: float = 1.0,
    update_hartree: int = 0,
    use_shrink_abfs: int = 0,
) -> str:
    return "\n".join(
        (
            "# qsgw_contract_version 4",
            "# task qsgw",
            "# fixed_basis immutable_reference",
            "# qsgw_mixer linear",
            f"# qsgw_mixing_beta {mixing_beta:.17g}",
            f"# qsgw_min_iter {declared_final_iteration}",
            f"# qsgw_max_iter {declared_final_iteration}",
            "# starting_vxc dft_only",
            "# vxc_basis fixed_state",
            f"# qsgw_update_hartree {update_hartree}",
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
            f"# use_shrink_abfs {use_shrink_abfs}",
            f"# use_fullcoul_exx {use_fullcoul_exx}",
            "# use_fullcoul_eps 1",
            "# use_fullcoul_wc 0",
            "# constants_choice internal",
            "# ac_policy direct_pade",
        )
    )


def current_header(
    *,
    symmetry: str = "unsupported_full_bz_only",
    mixer: str = "none",
    mixing_beta: float = 0.2,
    hartree: bool = False,
    hartree_coulomb: str = "truncated",
    hartree_normalization: str = "legacy_extra_inverse_nk",
) -> str:
    lines = [
            "# qsgw_contract_version 5",
            "# fixed_basis immutable_mf0",
            "# live_update eigenvalues_wfc",
            "# velocity disabled_stage1",
            "# headwing disabled_stage1",
            f"# symmetry {symmetry}",
            f"# hartree {'delta_density' if hartree else 'disabled_stage1'}",
    ]
    if hartree:
        lines.extend(
            (
                f"# hartree_coulomb {hartree_coulomb}",
                f"# hartree_normalization {hartree_normalization}",
            )
        )
    lines.extend(
        (
            "# band disabled_stage1",
            "# qsgw_input_contract /frozen/qsgw_input.contract",
            f"# qsgw_input_contract_sha256 {'a' * 64}",
            f"# qsgw_mixer {mixer}",
            f"# qsgw_mixing_beta {mixing_beta:.17g}",
        )
    )
    return "\n".join(lines)


def legacy_trace(
    *,
    exx: float = 1.0e-12,
    use_fullcoul_exx: int = 1,
    declared_final_iteration: int = 1,
    mixing_beta: float = 1.0,
    hartree: bool = False,
    use_shrink_abfs: bool = False,
    include_delta_vh: bool = True,
) -> str:
    rows = [
        row(0, "h0", 1.0),
        row(0, "vxc_dft", 0.1),
        row(0, "wfc_spinor_0", 1.0),
        row(0, "velocity_x", 0.3),
        row(0, "velocity_y", 0.4),
        row(0, "velocity_z", 0.5),
        row(0, "occupation", 2.0),
        row(0, "fermi_energy_ha", 0.5),
        row(0, "electron_count", 2.0),
        row(0, "gap_ha", 0.2),
        row(1, "sigma_c_iw", 0.01, frequency_index=0, frequency_ha=0.25),
        row(1, "exx", exx),
        row(1, "vc", 0.3),
        row(1, "raw_h", 1.1),
        row(1, "mixed_h", 1.1),
        row(1, "rotation_u", 1.0),
        row(1, "wfc_spinor_0", 1.0),
        row(1, "velocity_x", 0.3),
        row(1, "velocity_y", 0.4),
        row(1, "velocity_z", 0.5),
        row(1, "occupation", 2.0),
        row(1, "fermi_energy_ha", 0.55),
        row(1, "electron_count", 2.0),
        row(1, "gap_ha", 0.25),
    ]
    if hartree and include_delta_vh:
        rows.insert(15, row(1, "delta_vh", 0.05))
    return (
        legacy_header(
            use_fullcoul_exx=use_fullcoul_exx,
            declared_final_iteration=declared_final_iteration,
            mixing_beta=mixing_beta,
            update_hartree=int(hartree),
            use_shrink_abfs=int(use_shrink_abfs),
        )
        + "\n"
        + "\n".join(rows)
        + "\n"
    )


def current_matrix_trace(
    *,
    exx: float = 1.0e-12,
    symmetry: str = "unsupported_full_bz_only",
    mixer: str = "none",
    mixing_beta: float = 0.2,
    hartree: bool = False,
    include_delta_vh: bool = True,
    hartree_normalization: str = "legacy_extra_inverse_nk",
) -> str:
    rows = [
        row(0, "h0", 1.0),
        row(0, "vxc_dft", 0.1),
        row(0, "wfc_spinor_0", 1.0),
        row(0, "occupation", 2.0),
        row(1, "sigma_c_iw", 0.01, frequency_index=0, frequency_ha=0.25),
        row(1, "exx", exx),
        row(1, "vc", 0.3),
        row(1, "raw_h", 1.1),
        row(1, "mixed_h", 1.1),
        row(1, "rotation_u", 1.0),
        row(1, "wfc_spinor_0", 1.0),
        row(1, "occupation", 2.0),
    ]
    if hartree and include_delta_vh:
        rows.insert(8, row(1, "delta_vh", 0.05))
    return (
        current_header(
            symmetry=symmetry,
            mixer=mixer,
            mixing_beta=mixing_beta,
            hartree=hartree,
            hartree_normalization=hartree_normalization,
        )
        + "\n"
        + "\n".join(rows)
        + "\n"
    )


def current_eigenvalue_trace(
    *,
    iteration_one_ha: float = 1.1,
    mixer: str = "none",
    mixing_beta: float = 0.2,
    hartree: bool = False,
    hartree_normalization: str = "legacy_extra_inverse_nk",
) -> str:
    rows = (
        f"0 0 0 0 0 0 0 0 {1.0 * HA2EV:.17e}",
        f"1 0 0 0 0 0 0 0 {iteration_one_ha * HA2EV:.17e}",
    )
    return (
        current_header(
            mixer=mixer,
            mixing_beta=mixing_beta,
            hartree=hartree,
            hartree_normalization=hartree_normalization,
        )
        + "\n"
        + "\n".join(rows)
        + "\n"
    )


def current_iteration_trace(
    *,
    iteration_one_gap_ev: float = 0.25 * HA2EV,
    mixer: str = "none",
    mixing_beta: float = 0.2,
    hartree: bool = False,
    hartree_normalization: str = "legacy_extra_inverse_nk",
) -> str:
    rows = (
        f"0 0 0 0 {0.5 * HA2EV:.17e} {0.2 * HA2EV:.17e} 2",
        f"1 0 0 0 {0.55 * HA2EV:.17e} {iteration_one_gap_ev:.17e} 2",
    )
    return (
        current_header(
            mixer=mixer,
            mixing_beta=mixing_beta,
            hartree=hartree,
            hartree_normalization=hartree_normalization,
        )
        + "\n"
        + "\n".join(rows)
        + "\n"
    )


class LegacyV4CurrentV5ComparisonTest(unittest.TestCase):
    def compare(self, **overrides):
        function = getattr(MODULE, "compare_legacy_v4_current_v5", None)
        self.assertIsNotNone(
            function,
            "compare_legacy_v4_current_v5 must implement the v4-to-v5 contract",
        )
        assert function is not None
        arguments = {
            "old_matrix_text": legacy_trace(),
            "current_matrix_text": current_matrix_trace(),
            "current_eigenvalue_text": current_eigenvalue_trace(),
            "current_iteration_text": current_iteration_trace(),
            "iterations": [0, 1],
            "matrix_max_abs_tolerance_ha": 1.0e-8,
            "matrix_relative_tolerance": 1.0e-8,
            "eigenvalue_tolerance_ha": 1.0e-6,
            "gap_tolerance_ev": 1.0e-5,
            "state_tolerance": 1.0e-10,
        }
        arguments.update(overrides)
        return function(**arguments)

    @staticmethod
    def hartree_arguments(**matrix_overrides):
        return {
            "old_matrix_text": legacy_trace(
                hartree=True, use_shrink_abfs=True
            ),
            "current_matrix_text": current_matrix_trace(
                hartree=True, **matrix_overrides
            ),
            "current_eigenvalue_text": current_eigenvalue_trace(
                hartree=True
            ),
            "current_iteration_text": current_iteration_trace(
                hartree=True
            ),
            "expected_hartree": True,
            "expected_use_shrink_abfs": True,
        }

    def test_linear_beta_point_two_contract_passes(self):
        report = self.compare(
            old_matrix_text=legacy_trace(mixing_beta=0.2),
            current_matrix_text=current_matrix_trace(
                mixer="linear", mixing_beta=0.2
            ),
            current_eigenvalue_text=current_eigenvalue_trace(
                mixer="linear", mixing_beta=0.2
            ),
            current_iteration_text=current_iteration_trace(
                mixer="linear", mixing_beta=0.2
            ),
            expected_mixing_beta=0.2,
            expected_current_mixer="linear",
        )
        self.assertTrue(report["passed"])
        self.assertEqual(
            report["contract"]["direct_update_mapping"],
            "legacy_linear_beta_0.2_to_current_linear_beta_0.2",
        )

    def test_equivalent_direct_update_artifacts_pass(self):
        report = self.compare()
        self.assertTrue(report["passed"])
        self.assertEqual(report["contract"]["old_version"], 4)
        self.assertEqual(report["contract"]["current_version"], 5)

    def test_hartree_enabled_contract_and_delta_vh_pass(self):
        report = self.compare(**self.hartree_arguments())
        self.assertTrue(report["passed"])
        self.assertTrue(report["contract"]["hartree_enabled"])
        self.assertTrue(report["contract"]["legacy_use_shrink_abfs"])
        self.assertIn("delta_vh", report["components"])

    def test_hartree_enabled_missing_delta_vh_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "delta_vh"):
            self.compare(
                **self.hartree_arguments(include_delta_vh=False)
            )

    def test_hartree_enabled_wrong_normalization_is_rejected(self):
        arguments = self.hartree_arguments(
            hartree_normalization="weighted_occupations"
        )
        arguments["current_eigenvalue_text"] = current_eigenvalue_trace(
            hartree=True,
            hartree_normalization="weighted_occupations",
        )
        arguments["current_iteration_text"] = current_iteration_trace(
            hartree=True,
            hartree_normalization="weighted_occupations",
        )
        with self.assertRaisesRegex(ValueError, "hartree_normalization"):
            self.compare(**arguments)

    def test_legacy_effective_hamiltonian_uses_upper_triangle(self):
        interpret = getattr(MODULE, "_legacy_matrix_for_comparison", None)
        self.assertIsNotNone(interpret)
        legacy = np.array(
            [[1.0, 0.2 + 0.3j], [4.0 + 5.0j, 2.0]],
            dtype=np.complex128,
        )
        expected = np.array(
            [[1.0, 0.2 + 0.3j], [0.2 - 0.3j, 2.0]],
            dtype=np.complex128,
        )
        np.testing.assert_allclose(interpret("raw_h", legacy), expected)
        np.testing.assert_allclose(interpret("mixed_h", legacy), expected)
        np.testing.assert_allclose(interpret("exx", legacy), legacy)

    def test_matrix_block_frequencies_are_indexed_once(self):
        index_frequencies = getattr(MODULE, "_matrix_block_frequencies", None)
        self.assertIsNotNone(index_frequencies)
        rows = {
            (1, 0, "sigma_c_iw", 0, 2, 3, 0, 0): (0.25, 1.0 + 0.0j),
            (1, 0, "sigma_c_iw", 0, 2, 3, 0, 1): (0.25, 2.0 + 0.0j),
            (1, 0, "sigma_c_iw", 0, 2, 4, 0, 0): (0.50, 3.0 + 0.0j),
        }
        self.assertEqual(
            index_frequencies(rows),
            {
                (1, "sigma_c_iw", 0, 2, 3): 0.25,
                (1, "sigma_c_iw", 0, 2, 4): 0.50,
            },
        )

    def test_inconsistent_matrix_block_frequency_is_rejected(self):
        index_frequencies = getattr(MODULE, "_matrix_block_frequencies", None)
        self.assertIsNotNone(index_frequencies)
        rows = {
            (1, 0, "sigma_c_iw", 0, 2, 3, 0, 0): (0.25, 1.0 + 0.0j),
            (1, 0, "sigma_c_iw", 0, 2, 3, 0, 1): (0.30, 2.0 + 0.0j),
        }
        with self.assertRaisesRegex(ValueError, "inconsistent frequencies"):
            index_frequencies(rows)

    def test_invalid_current_symmetry_contract_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "symmetry"):
            self.compare(
                current_matrix_text=current_matrix_trace(symmetry="reduced")
            )

    def test_legacy_non_direct_mixing_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "qsgw_mixing_beta"):
            self.compare(
                old_matrix_text=legacy_trace().replace(
                    "# qsgw_mixing_beta 1",
                    "# qsgw_mixing_beta 0.2",
                    1,
                )
            )

    def test_legacy_full_coulomb_contract_mismatch_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "use_fullcoul_exx"):
            self.compare(
                old_matrix_text=legacy_trace().replace(
                    "# use_fullcoul_exx 1",
                    "# use_fullcoul_exx 0",
                    1,
                )
            )

    def test_explicit_legacy_truncated_coulomb_contract_passes(self):
        report = self.compare(
            old_matrix_text=legacy_trace(use_fullcoul_exx=0),
            expected_legacy_use_fullcoul_exx=False,
        )
        self.assertTrue(report["passed"])
        self.assertFalse(report["contract"]["legacy_declared_use_fullcoul_exx"])

    def test_expected_legacy_truncated_coulomb_rejects_full_header(self):
        with self.assertRaisesRegex(ValueError, "use_fullcoul_exx"):
            self.compare(expected_legacy_use_fullcoul_exx=False)

    def test_relative_frobenius_threshold_is_enforced(self):
        report = self.compare(
            current_matrix_text=current_matrix_trace(exx=2.0e-12)
        )
        self.assertFalse(report["passed"])
        self.assertGreater(
            report["components"]["exx"]["max_relative_frobenius"],
            1.0e-8,
        )

    def test_current_eigenvalue_trace_mismatch_is_rejected(self):
        report = self.compare(
            current_eigenvalue_text=current_eigenvalue_trace(
                iteration_one_ha=1.100002
            )
        )
        self.assertFalse(report["passed"])
        self.assertGreater(
            report["eigenvalues"]["max_old_vs_current_abs_diff_ha"],
            1.0e-6,
        )

    def test_gap_mismatch_is_rejected(self):
        report = self.compare(
            current_iteration_text=current_iteration_trace(
                iteration_one_gap_ev=0.25 * HA2EV + 2.0e-5
            )
        )
        self.assertFalse(report["passed"])
        self.assertGreater(report["scalars"]["max_gap_abs_diff_ev"], 1.0e-5)

    def test_cli_writes_a_passing_json_report(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            old = root / "old.dat"
            current = root / "current.dat"
            eigenvalues = root / "eigenvalues.dat"
            iterations = root / "iterations.dat"
            output = root / "comparison.json"
            old.write_text(legacy_trace(), encoding="utf-8")
            current.write_text(current_matrix_trace(), encoding="utf-8")
            eigenvalues.write_text(current_eigenvalue_trace(), encoding="utf-8")
            iterations.write_text(current_iteration_trace(), encoding="utf-8")
            command = [
                sys.executable,
                "-B",
                str(MODULE_PATH),
                str(old),
                str(current),
                str(output),
                "--iterations",
                "0:1",
                "--contract-mode",
                "legacy_v4_to_current_v5",
                "--current-eigenvalue-trace",
                str(eigenvalues),
                "--current-iteration-trace",
                str(iterations),
                "--matrix-max-abs-tolerance-ha",
                "1e-8",
                "--matrix-relative-tolerance",
                "1e-8",
                "--eigenvalue-tolerance",
                "1e-6",
                "--gap-tolerance-ev",
                "1e-5",
                "--state-tolerance",
                "1e-10",
            ]
            completed = subprocess.run(
                command, check=False, capture_output=True, text=True
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertTrue(report["passed"])

    def test_cli_accepts_linear_beta_point_two_contract(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            old = root / "old.dat"
            current = root / "current.dat"
            eigenvalues = root / "eigenvalues.dat"
            iterations = root / "iterations.dat"
            output = root / "comparison.json"
            old.write_text(
                legacy_trace(mixing_beta=0.2), encoding="utf-8"
            )
            current.write_text(
                current_matrix_trace(mixer="linear", mixing_beta=0.2),
                encoding="utf-8",
            )
            eigenvalues.write_text(
                current_eigenvalue_trace(mixer="linear", mixing_beta=0.2),
                encoding="utf-8",
            )
            iterations.write_text(
                current_iteration_trace(mixer="linear", mixing_beta=0.2),
                encoding="utf-8",
            )
            command = [
                sys.executable,
                "-B",
                str(MODULE_PATH),
                str(old),
                str(current),
                str(output),
                "--iterations",
                "0:1",
                "--contract-mode",
                "legacy_v4_to_current_v5",
                "--current-eigenvalue-trace",
                str(eigenvalues),
                "--current-iteration-trace",
                str(iterations),
                "--expected-mixing-beta",
                "0.2",
                "--expected-current-mixer",
                "linear",
            ]
            completed = subprocess.run(
                command, check=False, capture_output=True, text=True
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertTrue(report["passed"])
            self.assertEqual(
                report["contract"]["direct_update_mapping"],
                "legacy_linear_beta_0.2_to_current_linear_beta_0.2",
            )

    def test_cli_accepts_hartree_enabled_contract(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            old = root / "old.dat"
            current = root / "current.dat"
            eigenvalues = root / "eigenvalues.dat"
            iterations = root / "iterations.dat"
            output = root / "comparison.json"
            old.write_text(
                legacy_trace(hartree=True, use_shrink_abfs=True),
                encoding="utf-8",
            )
            current.write_text(
                current_matrix_trace(hartree=True), encoding="utf-8"
            )
            eigenvalues.write_text(
                current_eigenvalue_trace(hartree=True), encoding="utf-8"
            )
            iterations.write_text(
                current_iteration_trace(hartree=True), encoding="utf-8"
            )
            command = [
                sys.executable,
                "-B",
                str(MODULE_PATH),
                str(old),
                str(current),
                str(output),
                "--iterations",
                "0:1",
                "--contract-mode",
                "legacy_v4_to_current_v5",
                "--current-eigenvalue-trace",
                str(eigenvalues),
                "--current-iteration-trace",
                str(iterations),
                "--expected-hartree",
                "1",
                "--expected-use-shrink-abfs",
                "1",
            ]
            completed = subprocess.run(
                command, check=False, capture_output=True, text=True
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertTrue(report["passed"])
            self.assertTrue(report["contract"]["hartree_enabled"])

    def test_cli_requires_opt_in_to_compare_a_longer_run_prefix(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            old = root / "old.dat"
            current = root / "current.dat"
            eigenvalues = root / "eigenvalues.dat"
            iterations = root / "iterations.dat"
            output = root / "comparison.json"
            old.write_text(
                legacy_trace(declared_final_iteration=2), encoding="utf-8"
            )
            current.write_text(current_matrix_trace(), encoding="utf-8")
            eigenvalues.write_text(current_eigenvalue_trace(), encoding="utf-8")
            iterations.write_text(current_iteration_trace(), encoding="utf-8")
            command = [
                sys.executable,
                "-B",
                str(MODULE_PATH),
                str(old),
                str(current),
                str(output),
                "--iterations",
                "0:1",
                "--allow-iteration-prefix",
                "--contract-mode",
                "legacy_v4_to_current_v5",
                "--current-eigenvalue-trace",
                str(eigenvalues),
                "--current-iteration-trace",
                str(iterations),
            ]
            completed = subprocess.run(
                command, check=False, capture_output=True, text=True
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertTrue(report["passed"])
            self.assertEqual(
                report["contract"]["iteration_selection_mode"], "prefix"
            )
            self.assertEqual(
                report["contract"]["declared_final_iteration"], 2
            )

    def test_cli_accepts_explicit_legacy_truncated_coulomb_contract(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            old = root / "old.dat"
            current = root / "current.dat"
            eigenvalues = root / "eigenvalues.dat"
            iterations = root / "iterations.dat"
            output = root / "comparison.json"
            old.write_text(
                legacy_trace(use_fullcoul_exx=0), encoding="utf-8"
            )
            current.write_text(current_matrix_trace(), encoding="utf-8")
            eigenvalues.write_text(current_eigenvalue_trace(), encoding="utf-8")
            iterations.write_text(current_iteration_trace(), encoding="utf-8")
            command = [
                sys.executable,
                "-B",
                str(MODULE_PATH),
                str(old),
                str(current),
                str(output),
                "--iterations",
                "0:1",
                "--contract-mode",
                "legacy_v4_to_current_v5",
                "--current-eigenvalue-trace",
                str(eigenvalues),
                "--current-iteration-trace",
                str(iterations),
                "--expected-legacy-use-fullcoul-exx",
                "0",
                "--matrix-max-abs-tolerance-ha",
                "1e-8",
                "--matrix-relative-tolerance",
                "1e-8",
                "--eigenvalue-tolerance",
                "1e-6",
                "--gap-tolerance-ev",
                "1e-5",
                "--state-tolerance",
                "1e-10",
            ]
            completed = subprocess.run(
                command, check=False, capture_output=True, text=True
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertTrue(report["passed"])
            self.assertFalse(
                report["contract"]["legacy_declared_use_fullcoul_exx"]
            )


if __name__ == "__main__":
    unittest.main()
