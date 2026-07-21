#!/usr/bin/env python3
"""Tests for the native ABACUS Vxc reader compatibility observer."""

from __future__ import annotations

import unittest
from pathlib import Path

import audit_native_vxck_reader_compatibility_v1 as audit


class NativeVxckReaderCompatibilityTests(unittest.TestCase):
    def test_extract_block_uses_exact_boundaries(self) -> None:
        source = "prefix\nSTART\nreader body\nEND\nsuffix\n"
        self.assertEqual(
            audit.extract_block(source, "START", "END"),
            "START\nreader body\n",
        )

    def test_real_legacy_and_candidate_readers_share_native_contract(self) -> None:
        evidence_dir = Path(__file__).resolve().parent
        repo = evidence_dir.parents[2]
        oracle = (
            repo
            / "qsgw-rebase-evidence/remote/fish-gate-a-symmetry-20260720"
            / "oracle-source-audit-v1/exact847/task_qsgw_band_0.cpp"
        )
        scheme_a = (
            repo
            / "qsgw-rebase-evidence/remote/fish-gate-a-symmetry-20260720"
            / "patch-work-scheme-a-v3/task_qsgw_band_0.cpp"
        )

        report = audit.build_report(repo, oracle, scheme_a)

        self.assertTrue(report["source_contract_passed"])
        self.assertTrue(report["legacy_parser_unchanged_by_scheme_a"])
        self.assertTrue(report["both_readers_accept_native_comment_row_schema"])
        self.assertTrue(report["both_readers_convert_rydberg_to_hartree"])
        self.assertTrue(report["numerical_runtime_parity_pending"])


if __name__ == "__main__":
    unittest.main()
