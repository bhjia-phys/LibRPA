#!/usr/bin/env python3
"""Unit tests for the pinned ABACUS Si k444 full-BZ output observer."""

from __future__ import annotations

import os
import unittest
from pathlib import Path

import validate_abacus_si_k444_fullbz_output_v1 as observer


def full_bz_text(*, duplicate_last: bool = False, first_weight: float | None = None) -> str:
    rows = ["4 4 4", "64 64"]
    for index in range(64):
        ix = index // 16
        iy = (index // 4) % 4
        iz = index % 4
        if duplicate_last and index == 63:
            ix = iy = iz = 0
        weight = 1.0 / 64.0 if index or first_weight is None else first_weight
        rows.append(
            f"{index + 1} {weight:.17g} {ix / 4:.17g} {iy / 4:.17g} "
            f"{iz / 4:.17g} 0 0 0 {index + 1} {index + 1}"
        )
    return "\n".join(rows) + "\n"


def identity_structure_text() -> str:
    fields = ["1"] * 18
    fields.extend(["2", "0", "0", "0", "1", "0.25", "0.25", "0.25", "1"])
    fields.extend(["1", "row", "1", "0", "0", "0", "1", "0", "0", "0", "1", "0", "0", "0"])
    return " ".join(fields) + "\n"


def structure_without_symmetry_text() -> str:
    fields = ["1"] * 18
    fields.extend(["2", "0", "0", "0", "1", "0.25", "0.25", "0.25", "1"])
    return " ".join(fields) + "\n"


class FullBzObserverTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        fixed = os.environ.get("LIBRPA_QSGW_FULLBZ_OBSERVER_TEST_TMP")
        if not fixed:
            raise RuntimeError("LIBRPA_QSGW_FULLBZ_OBSERVER_TEST_TMP is required")
        cls.root = Path(fixed)
        cls.root.mkdir(parents=True, exist_ok=True)

    def write(self, name: str, text: str) -> Path:
        path = self.root / name
        path.write_text(text, encoding="ascii")
        return path

    def test_accepts_exact_64_point_full_bz(self) -> None:
        summary = observer.parse_bz_sampling(
            self.write("bz-good.out", full_bz_text())
        )
        self.assertEqual(summary["grid"], [4, 4, 4])
        self.assertEqual(summary["n_scf"], 64)
        self.assertEqual(summary["n_ibz"], 64)
        self.assertTrue(summary["full_bz"])

    def test_rejects_nonuniform_full_bz_weight(self) -> None:
        with self.assertRaisesRegex(SystemExit, "uniform 1/64"):
            observer.parse_bz_sampling(
                self.write("bz-weight.out", full_bz_text(first_weight=0.02))
            )

    def test_rejects_periodic_duplicate(self) -> None:
        with self.assertRaisesRegex(SystemExit, "periodic k-points are not unique"):
            observer.parse_bz_sampling(
                self.write("bz-duplicate.out", full_bz_text(duplicate_last=True))
            )

    def test_accepts_absent_structure_symmetry_metadata(self) -> None:
        summary = observer.parse_stru(
            self.write("stru-no-symmetry.out", structure_without_symmetry_text())
        )
        self.assertEqual(summary["n_symops"], 0)
        self.assertEqual(summary["convention"], "absent")
        self.assertFalse(summary["identity_present"])

    def test_rejects_unexpected_structure_symmetry_metadata(self) -> None:
        with self.assertRaisesRegex(SystemExit, "unexpected symmetry-operation metadata"):
            observer.parse_stru(
                self.write("stru-identity.out", identity_structure_text())
            )


if __name__ == "__main__":
    unittest.main()
