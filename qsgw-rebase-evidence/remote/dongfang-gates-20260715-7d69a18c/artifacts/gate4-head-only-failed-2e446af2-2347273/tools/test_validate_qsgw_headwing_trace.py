#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import os
import sys
import unittest
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
MODULE_PATH = Path(
    os.environ.get(
        "QSGW_HEADWING_VALIDATOR",
        str(HERE / "validate_qsgw_headwing_trace.py"),
    )
).resolve()
sys.path.insert(0, str(HERE))


class ValidateQsgwHeadwingTraceTest(unittest.TestCase):
    def load_module(self):
        self.assertTrue(
            MODULE_PATH.exists(),
            "validate_qsgw_headwing_trace.py must implement the structural observer",
        )
        spec = importlib.util.spec_from_file_location(
            "validate_qsgw_headwing_trace", MODULE_PATH
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
        frozen_velocity: bool = False,
        nonhermitian_head: bool = False,
        initial_head_drift: bool = False,
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
            "# qsgw_input_contract /tmp/qsgw_input.contract",
            "# qsgw_input_contract_sha256 " + "a" * 64,
            "# qsgw_mixer none",
            "# qsgw_mixing_beta 0.20000000000000001",
            "# iter channel component spin kpoint frequency_index frequency_Ha row column real_value imag_value",
        ]
        identity = np.eye(2, dtype=np.complex128)
        angle = 0.37
        phase = 0.41
        rotation = np.asarray(
            [
                [np.cos(angle), -np.exp(1.0j * phase) * np.sin(angle)],
                [np.exp(-1.0j * phase) * np.sin(angle), np.cos(angle)],
            ],
            dtype=np.complex128,
        )
        initial_velocity = {
            "velocity_x": np.diag([1.0, 2.0]).astype(np.complex128),
            "velocity_y": np.asarray([[0.0, 1.0j], [-1.0j, 0.0]]),
            "velocity_z": np.diag([3.0, 4.0]).astype(np.complex128),
        }
        for iteration, unitary in ((0, identity), (1, identity), (2, rotation)):
            if iteration > 0:
                lines.extend(self.row(iteration, "rotation_u", unitary))
            for component, reference in initial_velocity.items():
                live = unitary.conj().T @ reference @ unitary
                if frozen_velocity and iteration == 2:
                    live = reference
                lines.extend(self.row(iteration, component, live))
            for frequency_index, frequency in enumerate((0.25, 0.75)):
                head = np.diag(
                    [2.0 + frequency, 3.0 + frequency, 4.0 + frequency]
                ).astype(np.complex128)
                if initial_head_drift and iteration == 1:
                    head += np.eye(3) * 0.05
                if iteration == 2:
                    head += np.eye(3) * 0.1
                if nonhermitian_head and iteration == 2:
                    head[0, 1] = 0.2j
                lines.extend(
                    self.row(
                        iteration,
                        "head_tensor",
                        head,
                        frequency_index=frequency_index,
                        frequency=frequency,
                    )
                )
        return "\n".join(lines) + "\n"

    def test_live_velocity_and_head_tensor_invariants_pass(self):
        module = self.load_module()
        report = module.validate(
            self.make_trace(),
            [0, 1, 2],
            channel=0,
            invariant_tolerance=1.0e-12,
            head_live_change_minimum=1.0e-6,
        )
        self.assertTrue(report["passed"])
        self.assertLessEqual(report["velocity_rotation_relative_residual"], 1.0e-12)
        self.assertEqual(report["head_frequency_count"], 2)
        self.assertGreater(report["head_post_update_max_abs_change"], 1.0e-6)

    def test_frozen_velocity_is_rejected(self):
        module = self.load_module()
        report = module.validate(
            self.make_trace(frozen_velocity=True),
            [0, 1, 2],
            channel=0,
            invariant_tolerance=1.0e-12,
            head_live_change_minimum=1.0e-6,
        )
        self.assertFalse(report["passed"])
        self.assertGreater(report["velocity_rotation_relative_residual"], 1.0e-12)

    def test_nonhermitian_head_is_rejected(self):
        module = self.load_module()
        report = module.validate(
            self.make_trace(nonhermitian_head=True),
            [0, 1, 2],
            channel=0,
            invariant_tolerance=1.0e-12,
            head_live_change_minimum=1.0e-6,
        )
        self.assertFalse(report["passed"])
        self.assertGreater(report["head_hermiticity_max_abs"], 1.0e-12)

    def test_initial_head_replay_drift_is_rejected(self):
        module = self.load_module()
        report = module.validate(
            self.make_trace(initial_head_drift=True),
            [0, 1, 2],
            channel=0,
            invariant_tolerance=1.0e-12,
            initial_replay_tolerance=1.0e-12,
            head_live_change_minimum=1.0e-6,
        )
        self.assertFalse(report["passed"])
        self.assertGreater(
            report["head_iteration0_to1_max_abs_change"], 1.0e-12
        )


if __name__ == "__main__":
    unittest.main()
