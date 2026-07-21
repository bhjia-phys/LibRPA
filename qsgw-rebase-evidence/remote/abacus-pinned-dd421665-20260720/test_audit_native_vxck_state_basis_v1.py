#!/usr/bin/env python3
"""Tests for the ABACUS native Vxc state-basis source audit."""

from __future__ import annotations

import unittest
from pathlib import Path

import audit_native_vxck_state_basis_v1 as audit


class NativeVxcStateBasisAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.repo = Path(__file__).resolve().parents[3]
        cls.workspace = cls.repo.parents[3]
        cls.abacus_repo = (
            cls.workspace
            / "research/librpa/analysis/si8_newlat_origin_k999_g0w0_20260625"
            / "upstream_input_contract_study_20260720/abacus-develop-master_ghj"
        )
        cls.legacy_source = (
            cls.repo
            / "qsgw-rebase-evidence/remote/fish-gate-a-symmetry-20260720"
            / "oracle-source-audit-v1/exact847/task_qsgw_band_0.cpp"
        )
        cls.generator = Path(__file__).with_name(
            "prepare_abacus_qsgw_ibz_contract_v3.py"
        )

    def test_live_sources_prove_native_vxck_is_fixed_state_basis(self) -> None:
        report = audit.build_report(
            self.repo,
            self.abacus_repo,
            self.legacy_source,
            self.generator,
        )

        self.assertTrue(report["source_contract_passed"])
        self.assertEqual(
            report["producer"]["commit"],
            "dd4216653386d32f79e3219f3ea5dd2d229c1c5a",
        )
        self.assertEqual(report["producer"]["matrix_basis"], "ks_state")
        self.assertEqual(report["producer"]["matrix_shape"], "n_bands_x_n_bands")
        self.assertTrue(report["producer"]["nao_suffix_is_not_basis_label"])
        self.assertEqual(report["legacy"]["basis_transform"], "none")
        self.assertEqual(report["candidate"]["basis_transform"], "none")
        self.assertTrue(report["candidate"]["true_nao_projection_path_retained"])
        self.assertEqual(report["generator"]["manifest_basis"], "state")
        self.assertEqual(report["generator"]["manifest_gauge"], "mf0_state")
        self.assertEqual(report["generator"]["dimension_binding"], "n_bands")


if __name__ == "__main__":
    unittest.main()
