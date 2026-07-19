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
        "QSGW_HEAD_MODE_COMPARATOR",
        str(HERE / "compare_qsgw_head_mode_pair.py"),
    )
).resolve()
sys.path.insert(0, str(HERE))


class CompareQsgwHeadModePairTest(unittest.TestCase):
    def load_module(self):
        self.assertTrue(
            MODULE_PATH.exists(),
            "compare_qsgw_head_mode_pair.py must implement the controlled observer",
        )
        spec = importlib.util.spec_from_file_location(
            "compare_qsgw_head_mode_pair", MODULE_PATH
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
        head_only: bool,
        initial_shift: float = 0.0,
        exx_shift: float = 0.0,
        sigma_shift: float = 0.1,
        wrong_contract_path: bool = False,
    ) -> str:
        contract_sha = "b" * 64 if head_only else "a" * 64
        contract_name = (
            "qsgw_input.head-only.contract"
            if head_only
            else "qsgw_input.disabled.contract"
        )
        if wrong_contract_path:
            contract_name = "wrong.contract"
        lines = [
            "# qsgw_contract_version 5",
            "# fixed_basis immutable_mf0",
            "# live_update eigenvalues_wfc",
            "# velocity "
            + ("fixed_basis_rotation" if head_only else "disabled_stage1"),
            "# headwing "
            + ("scf_grid_analytic_live" if head_only else "disabled_stage1"),
            "# symmetry unsupported_full_bz_only",
            "# hartree disabled_stage1",
            "# band disabled_stage1",
            f"# qsgw_input_contract /tmp/{contract_name}",
            f"# qsgw_input_contract_sha256 {contract_sha}",
            "# qsgw_mixer none",
            "# qsgw_mixing_beta 0.20000000000000001",
            "# iter channel component spin kpoint frequency_index frequency_Ha row column real_value imag_value",
        ]
        h0 = np.diag([1.0 + initial_shift, 2.0]).astype(np.complex128)
        wfc = np.eye(2, dtype=np.complex128)
        exx = np.diag([-0.4 + exx_shift, -0.2]).astype(np.complex128)
        sigma = np.diag([0.03, 0.05]).astype(np.complex128)
        if head_only:
            sigma = sigma + np.eye(2) * sigma_shift
        lines.extend(self.row(0, "h0", h0))
        lines.extend(self.row(0, "wfc_spinor0", wfc))
        lines.extend(self.row(1, "exx", exx))
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
    def pair_record() -> dict[str, object]:
        return {
            "schema": "qsgw-head-mode-controlled-pair-v1",
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
            "side_a": {"parameters": {"head_mode": "disabled"}},
            "side_b": {"parameters": {"head_mode": "head_only"}},
            "variants": {
                "disabled": {
                    "replace_w_head": False,
                    "option_dielect_func": 0,
                    "qsgw_input_contract": "qsgw_input.disabled.contract",
                    "qsgw_input_contract_sha256": "a" * 64,
                    "librpa_input_sha256": "f" * 64,
                },
                "head_only": {
                    "replace_w_head": True,
                    "option_dielect_func": 4,
                    "qsgw_input_contract": "qsgw_input.head-only.contract",
                    "qsgw_input_contract_sha256": "b" * 64,
                    "librpa_input_sha256": "1" * 64,
                },
            },
        }

    def test_controlled_head_mode_pair_passes(self):
        module = self.load_module()
        report = module.compare(
            self.make_trace(head_only=False),
            self.make_trace(head_only=True),
            self.pair_record(),
            initial_tolerance=1.0e-12,
            exx_tolerance=1.0e-12,
            head_effect_minimum=1.0e-6,
        )
        self.assertTrue(report["passed"])
        self.assertLessEqual(report["initial_max_abs"], 1.0e-12)
        self.assertLessEqual(report["iteration1_exx_max_abs"], 1.0e-12)
        self.assertGreater(report["iteration1_sigc_max_abs_change"], 1.0e-6)

    def test_initial_state_drift_is_rejected(self):
        module = self.load_module()
        report = module.compare(
            self.make_trace(head_only=False),
            self.make_trace(head_only=True, initial_shift=1.0e-3),
            self.pair_record(),
            initial_tolerance=1.0e-12,
            exx_tolerance=1.0e-12,
            head_effect_minimum=1.0e-6,
        )
        self.assertFalse(report["passed"])

    def test_exx_drift_is_rejected(self):
        module = self.load_module()
        report = module.compare(
            self.make_trace(head_only=False),
            self.make_trace(head_only=True, exx_shift=1.0e-3),
            self.pair_record(),
            initial_tolerance=1.0e-12,
            exx_tolerance=1.0e-12,
            head_effect_minimum=1.0e-6,
        )
        self.assertFalse(report["passed"])

    def test_missing_head_effect_is_rejected(self):
        module = self.load_module()
        report = module.compare(
            self.make_trace(head_only=False),
            self.make_trace(head_only=True, sigma_shift=0.0),
            self.pair_record(),
            initial_tolerance=1.0e-12,
            exx_tolerance=1.0e-12,
            head_effect_minimum=1.0e-6,
        )
        self.assertFalse(report["passed"])

    def test_second_parameter_difference_is_rejected(self):
        module = self.load_module()
        pair = deepcopy(self.pair_record())
        pair["side_b"]["parameters"]["mixing"] = "linear"
        with self.assertRaisesRegex(ValueError, "only head_mode"):
            module.compare(
                self.make_trace(head_only=False),
                self.make_trace(head_only=True),
                pair,
                initial_tolerance=1.0e-12,
                exx_tolerance=1.0e-12,
                head_effect_minimum=1.0e-6,
            )

    def test_trace_contract_path_mismatch_is_rejected(self):
        module = self.load_module()
        with self.assertRaisesRegex(ValueError, "contract path"):
            module.compare(
                self.make_trace(head_only=False),
                self.make_trace(head_only=True, wrong_contract_path=True),
                self.pair_record(),
                initial_tolerance=1.0e-12,
                exx_tolerance=1.0e-12,
                head_effect_minimum=1.0e-6,
            )


if __name__ == "__main__":
    unittest.main()
