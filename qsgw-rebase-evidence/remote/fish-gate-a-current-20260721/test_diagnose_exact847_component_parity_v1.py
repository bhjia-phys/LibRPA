#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
MODULE_PATH = HERE / "diagnose_exact847_component_parity_v1.py"
SPEC = importlib.util.spec_from_file_location("diagnostic_under_test", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def trace_row(iteration: int, component: str, row: int, column: int, value: complex) -> str:
    return (
        f"{iteration} 0 {component} 0 0 -1 0.0 {row} {column} "
        f"{value.real:.17e} {value.imag:.17e}"
    )


class Exact847ComponentDiagnosticTests(unittest.TestCase):
    def test_parses_complete_static_trace(self):
        lines = ["# qsgw_contract_version 6"]
        for component, iteration in MODULE.MATRIX_COMPONENTS.items():
            for row in range(2):
                for column in range(2):
                    lines.append(
                        trace_row(
                            iteration,
                            component,
                            row,
                            column,
                            complex(row + column + 1, row - column),
                        )
                    )
        parsed = MODULE.parse_static_components_text(
            "\n".join(lines) + "\n", "fixture", 1, 1, 2
        )
        self.assertEqual(set(parsed), set(MODULE.MATRIX_COMPONENTS))
        self.assertEqual(parsed["vc"][0, 0, 1, 0], complex(2, 1))

    def test_rejects_missing_component_rows(self):
        with self.assertRaisesRegex(MODULE.DiagnosticError, "h0 has 0 rows"):
            MODULE.parse_static_components_text(
                "# qsgw_contract_version 6\n", "fixture", 1, 1, 2
            )

    def test_parses_first_complete_legacy_state_table(self):
        text = """
Final Quasi-Particle Energy after QSGW Iterations [unit: eV]
spin  1, k-point    1: (0.0, 0.0, 0.0)
State e_mf v_xc v_exx1 v_exx2 ReSigc ImSigc e_qp
1 1.0 2.0 3.0 4.0 0.0 0.0
2 5.0 6.0 7.0 8.0 0.0 0.0
"""
        parsed = MODULE.parse_legacy_state_table_text(text, 1, 1, 2)
        self.assertEqual(parsed["vxc_ev"][0, 0, 1], 6.0)
        self.assertEqual(parsed["exx_matrix_ev"][0, 0, 0], 4.0)

    def test_analysis_localizes_vc_difference(self):
        h0 = np.diag([1.0, 2.0]).reshape(1, 1, 2, 2).astype(complex)
        zero = np.zeros_like(h0)
        vxc = np.diag([0.2, 0.3]).reshape(1, 1, 2, 2).astype(complex)
        exx = np.diag([-0.4, -0.5]).reshape(1, 1, 2, 2).astype(complex)
        candidate_vc = np.diag([0.1, 0.2]).reshape(1, 1, 2, 2).astype(complex)
        legacy_vc = 2.0 * candidate_vc
        base = h0 - vxc + exx
        legacy_h = base + legacy_vc
        candidate_h = base + candidate_vc
        components = {
            "h0": h0,
            "vxc_dft": vxc,
            "exx": exx,
            "vc": candidate_vc,
            "raw_h": candidate_h,
            "mixed_h": candidate_h,
        }
        legacy_table = {
            "e_mf_ev": np.linalg.eigvalsh(legacy_h) * MODULE.HA2EV,
            "vxc_ev": MODULE.diagonal(vxc) * MODULE.HA2EV,
            "exx_energy_ev": MODULE.diagonal(exx) * MODULE.HA2EV,
            "exx_matrix_ev": MODULE.diagonal(exx) * MODULE.HA2EV,
            "resigc_ev": MODULE.diagonal(zero).real,
            "imsigc_ev": MODULE.diagonal(zero).real,
        }
        report = MODULE.analyze(legacy_h, components, legacy_table)
        self.assertTrue(report["diagnostic_complete"])
        self.assertAlmostEqual(
            report["candidate_raw_closure"]["max_abs_ha"], 0.0
        )
        self.assertAlmostEqual(
            report["candidate_vs_legacy"]["best_real_vc_scale"], 2.0
        )
        self.assertAlmostEqual(
            report["candidate_vs_legacy"]["best_scaled_vc_hamiltonian"]["max_abs_ha"],
            0.0,
        )


if __name__ == "__main__":
    unittest.main()
