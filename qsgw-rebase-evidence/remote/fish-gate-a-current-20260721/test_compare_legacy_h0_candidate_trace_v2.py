#!/usr/bin/env python3
from __future__ import annotations

import unittest
from pathlib import Path
from unittest import mock

from compare_legacy_h0_candidate_trace_v2 import (
    legacy,
    require_candidate_contract,
)


class ContractV6Tests(unittest.TestCase):
    def test_contract_v6_is_accepted(self) -> None:
        path = Path("trace.dat")
        with mock.patch.object(
            Path, "read_text", return_value="# qsgw_contract_version 6\n"
        ):
            require_candidate_contract(path)

    def test_contract_v5_is_rejected(self) -> None:
        path = Path("trace.dat")
        with mock.patch.object(
            Path, "read_text", return_value="# qsgw_contract_version 5\n"
        ):
            with self.assertRaisesRegex(
                legacy.ComparisonError, "missing contract-v6 header"
            ):
                require_candidate_contract(path)

    def test_missing_contract_is_rejected(self) -> None:
        path = Path("trace.dat")
        with mock.patch.object(
            Path, "read_text", return_value="# fixed_basis immutable_mf0\n"
        ):
            with self.assertRaises(legacy.ComparisonError):
                require_candidate_contract(path)


if __name__ == "__main__":
    unittest.main()
