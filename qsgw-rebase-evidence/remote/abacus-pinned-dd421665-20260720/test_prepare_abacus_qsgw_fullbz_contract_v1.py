#!/usr/bin/env python3
"""Unit tests for the state-basis ABACUS full-BZ QSGW contract generator."""

from __future__ import annotations

import os
import unittest
from pathlib import Path

import prepare_abacus_qsgw_fullbz_contract_v1 as contract


def full_bz_text(*, counts: tuple[int, int] = (64, 64), duplicate_last: bool = False) -> str:
    rows = ["4 4 4", f"{counts[0]} {counts[1]}"]
    for index in range(64):
        ix = index // 16
        iy = (index // 4) % 4
        iz = index % 4
        if duplicate_last and index == 63:
            ix = iy = iz = 0
        rows.append(
            f"{index + 1} {1.0 / 64.0:.17g} {ix / 4:.17g} "
            f"{iy / 4:.17g} {iz / 4:.17g} 0 0 0 {index + 1} {index + 1}"
        )
    return "\n".join(rows) + "\n"


class FullBzContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        fixed = os.environ.get("LIBRPA_QSGW_FULLBZ_CONTRACT_TEST_TMP")
        if not fixed:
            raise RuntimeError("LIBRPA_QSGW_FULLBZ_CONTRACT_TEST_TMP is required")
        cls.root = Path(fixed)
        cls.root.mkdir(parents=True, exist_ok=True)

    def write(self, name: str, text: str) -> Path:
        path = self.root / name
        path.write_text(text, encoding="ascii")
        return path

    def test_reads_exact_k444_full_bz(self) -> None:
        grid, kpoints, multiplicities = contract.read_full_bz(
            self.write("contract-good.out", full_bz_text()), 64
        )
        self.assertEqual(grid, (4, 4, 4))
        self.assertEqual(len(kpoints), 64)
        self.assertEqual(multiplicities, [1] * 64)

    def test_rejects_reduced_counts(self) -> None:
        with self.assertRaisesRegex(ValueError, "full-BZ counts"):
            contract.read_full_bz(
                self.write("contract-reduced.out", full_bz_text(counts=(8, 8))),
                64,
            )

    def test_rejects_periodic_duplicate(self) -> None:
        with self.assertRaisesRegex(ValueError, "periodic k-points are not unique"):
            contract.read_full_bz(
                self.write(
                    "contract-duplicate.out",
                    full_bz_text(duplicate_last=True),
                ),
                64,
            )

    def test_source_reuses_state_basis_writer(self) -> None:
        source = Path(contract.__file__).read_text(encoding="utf-8")
        self.assertIn("shared.write_vxc_manifest", source)
        self.assertIn("shared.write_contract", source)
        self.assertIn('"sampling_mode": "full_bz"', source)


if __name__ == "__main__":
    unittest.main()
