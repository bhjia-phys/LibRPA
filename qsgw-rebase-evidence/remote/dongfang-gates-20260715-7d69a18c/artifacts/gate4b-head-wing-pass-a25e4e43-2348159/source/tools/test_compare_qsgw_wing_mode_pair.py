#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import os
import sys
import unittest
from copy import deepcopy
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
MODULE_PATH = Path(
    os.environ.get(
        "QSGW_WING_MODE_COMPARATOR",
        str(HERE / "compare_qsgw_wing_mode_pair.py"),
    )
).resolve()
sys.path.insert(0, str(HERE))


class CompareQsgwWingModePairTest(unittest.TestCase):
    def load_module(self):
        self.assertTrue(
            MODULE_PATH.exists(),
            "compare_qsgw_wing_mode_pair.py must implement the controlled observer",
        )
        spec = importlib.util.spec_from_file_location(
            "compare_qsgw_wing_mode_pair", MODULE_PATH
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    @staticmethod
    def row(
        iteration: int,
        component: str,
        matrix: np.ndarray,
        *,
        frequency_index: int = -1,
        frequency: float = 0.0,
    ) -> list[str]:
        rows = []
        for row in range(matrix.shape[0]):
            for column in range(matrix.shape[1]):
                value = matrix[row, column]
                rows.append(
                    f"{iteration} 0 {component} 0 0 {frequency_index} "
                    f"{frequency:.17e} {row} {column} "
                    f"{value.real:.17e} {value.imag:.17e}"
                )
        return rows

    def make_trace(
        self,
        *,
        wing: bool,
        initial_shift: float = 0.0,
        exx_shift: float = 0.0,
        head_shift: float = 0.0,
        sigma_shift: float = 0.1,
    ) -> str:
        lines = [
            "# qsgw_contract_version 5",
            "# fixed_basis immutable_mf0",
            "# live_update eigenvalues_wfc",
            "# velocity fixed_basis_rotation",
            "# headwing scf_grid_analytic_live",
            "# symmetry unsupported_full_bz_only",
            "# hartree disabled_stage1",
            "# band disabled_stage1",
            "# qsgw_input_contract /tmp/qsgw_input.head-only.contract",
            f"# qsgw_input_contract_sha256 {'b' * 64}",
            "# qsgw_mixer none",
            "# qsgw_mixing_beta 0.20000000000000001",
            "# iter channel component spin kpoint frequency_index frequency_Ha row column real_value imag_value",
        ]
        h0 = np.diag([1.0 + initial_shift, 2.0]).astype(np.complex128)
        wfc = np.eye(2, dtype=np.complex128)
        exx = np.diag([-0.4 + exx_shift, -0.2]).astype(np.complex128)
        head = np.diag([1.2 + head_shift, 1.3, 1.4]).astype(np.complex128)
        sigma = np.diag([0.03, 0.05]).astype(np.complex128)
        if wing:
            sigma = sigma + np.eye(2) * sigma_shift
        lines.extend(self.row(0, "h0", h0))
        lines.extend(self.row(0, "wfc_spinor0", wfc))
        lines.extend(self.row(1, "exx", exx))
        lines.extend(
            self.row(
                1,
                "head_tensor",
                head,
                frequency_index=0,
                frequency=0.25,
            )
        )
        lines.extend(
            self.row(
                1,
                "sigma_c_iw",
                sigma,
                frequency_index=0,
                frequency=0.25,
            )
        )
        return "\n".join(lines) + "\n"

    @staticmethod
    def make_stdout(*, wing: bool, negative_gram: bool = False) -> str:
        option = 3 if wing else 4
        lines = [
            "replace_w_head = true",
            f"option_dielect_func = {option}",
        ]
        if wing:
            lines.extend(
                [
                    "* Success: calculate wing term.",
                    "* Success: calculate wing term.",
                    "* Success: calculate wing term.",
                    "Wing_mu diagnostics (iomega=0): max_abs_real= 1.00000000e+00 max_abs_imag= 2.00000000e-01 max_abs_value= 1.00000000e+00 real_over_abs= 1.00000000e+00",
                    "Wing_mu Gram (iomega=0, rows alpha, columns beta):",
                ]
            )
            diagonal = -1.0 if negative_gram else 1.0
            lines.extend(
                [
                    f"({diagonal:15.8e},{0.0:15.8e}) ({0.0:15.8e},{0.0:15.8e}) ({0.0:15.8e},{0.0:15.8e})",
                    f"({0.0:15.8e},{0.0:15.8e}) ({2.0:15.8e},{0.0:15.8e}) ({0.0:15.8e},{0.0:15.8e})",
                    f"({0.0:15.8e},{0.0:15.8e}) ({0.0:15.8e},{0.0:15.8e}) ({3.0:15.8e},{0.0:15.8e})",
                ]
            )
        return "\n".join(lines) + "\n"

    @staticmethod
    def pair_record() -> dict[str, object]:
        return {
            "schema": "qsgw-wing-mode-controlled-pair-v1",
            "changed_factor": "parameters.head_mode",
            "factor_role": "head_mode",
            "common": {
                "source_commit": "c" * 40,
                "executable_sha256": "d" * 64,
                "dataset_manifest_sha256": "e" * 64,
                "mpi_ranks": 4,
                "omp_threads": 12,
                "deterministic_reduction": True,
            },
            "side_a": {"parameters": {"head_mode": "head_only"}},
            "side_b": {"parameters": {"head_mode": "head_plus_wing"}},
            "variants": {
                "head_only": {
                    "replace_w_head": True,
                    "option_dielect_func": 4,
                    "qsgw_input_contract": "qsgw_input.head-only.contract",
                    "qsgw_input_contract_sha256": "b" * 64,
                    "librpa_input_sha256": "f" * 64,
                },
                "head_plus_wing": {
                    "replace_w_head": True,
                    "option_dielect_func": 3,
                    "qsgw_input_contract": "qsgw_input.head-only.contract",
                    "qsgw_input_contract_sha256": "b" * 64,
                    "librpa_input_sha256": "1" * 64,
                },
            },
        }

    def compare(self, module, **changes):
        arguments = {
            "head_only_text": self.make_trace(wing=False),
            "head_plus_wing_text": self.make_trace(wing=True),
            "head_only_stdout": self.make_stdout(wing=False),
            "head_plus_wing_stdout": self.make_stdout(wing=True),
            "pair": self.pair_record(),
            "initial_tolerance": 1.0e-12,
            "invariant_tolerance": 1.0e-12,
            "wing_effect_minimum": 1.0e-6,
        }
        arguments.update(changes)
        return module.compare(**arguments)

    def test_controlled_wing_mode_pair_passes(self):
        report = self.compare(self.load_module())
        self.assertTrue(report["passed"])
        self.assertLessEqual(report["initial_max_abs"], 1.0e-12)
        self.assertLessEqual(report["iteration1_exx_max_abs"], 1.0e-12)
        self.assertLessEqual(report["head_tensor_max_abs"], 1.0e-12)
        self.assertGreater(report["iteration1_sigc_max_abs_change"], 1.0e-6)
        self.assertGreaterEqual(report["wing_success_count"], 3)
        self.assertGreaterEqual(report["wing_gram_min_eigenvalue"], -1.0e-12)

    def test_initial_state_drift_is_rejected(self):
        module = self.load_module()
        report = self.compare(
            module,
            head_plus_wing_text=self.make_trace(wing=True, initial_shift=1.0e-3),
        )
        self.assertFalse(report["passed"])

    def test_exx_or_head_drift_is_rejected(self):
        module = self.load_module()
        exx_report = self.compare(
            module,
            head_plus_wing_text=self.make_trace(wing=True, exx_shift=1.0e-3),
        )
        head_report = self.compare(
            module,
            head_plus_wing_text=self.make_trace(wing=True, head_shift=1.0e-3),
        )
        self.assertFalse(exx_report["passed"])
        self.assertFalse(head_report["passed"])

    def test_missing_wing_effect_is_rejected(self):
        report = self.compare(
            self.load_module(),
            head_plus_wing_text=self.make_trace(wing=True, sigma_shift=0.0),
        )
        self.assertFalse(report["passed"])

    def test_non_positive_wing_gram_is_rejected(self):
        report = self.compare(
            self.load_module(),
            head_plus_wing_stdout=self.make_stdout(wing=True, negative_gram=True),
        )
        self.assertFalse(report["passed"])

    def test_second_parameter_difference_is_rejected(self):
        module = self.load_module()
        pair = deepcopy(self.pair_record())
        pair["side_b"]["parameters"]["mixing"] = "linear"
        with self.assertRaisesRegex(ValueError, "only head_mode"):
            self.compare(module, pair=pair)

    def test_stdout_mode_mismatch_is_rejected(self):
        module = self.load_module()
        with self.assertRaisesRegex(ValueError, "option_dielect_func"):
            self.compare(
                module,
                head_plus_wing_stdout=self.make_stdout(wing=False),
            )


if __name__ == "__main__":
    unittest.main()
