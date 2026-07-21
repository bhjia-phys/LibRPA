#!/usr/bin/env python3
"""Static acceptance checks for the corrected legacy Hartree reader patch."""

from __future__ import annotations

import unittest
from pathlib import Path


PATCH = (
    Path(__file__).resolve().parent
    / "legacy-qsgw-hartree-isolated-full-reader-v1.patch"
)


class LegacyHartreeIsolatedReaderPatchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = PATCH.read_text(encoding="utf-8")

    def test_only_legacy_qsgw_driver_is_modified(self) -> None:
        self.assertEqual(self.source.count("diff --git "), 1)
        self.assertIn(
            "diff --git a/driver/task_qsgw.cpp b/driver/task_qsgw.cpp",
            self.source,
        )

    def test_normal_gw_coulomb_map_is_moved_and_restored(self) -> None:
        required = (
            "+#include <utility>",
            "auto gw_vq_cut = std::move(Vq_cut);",
            "Vq_cut.clear();",
            "hartree_full_vq_cut = std::move(Vq_cut);",
            "Vq_cut = std::move(gw_vq_cut);",
            "gw_vq_restored=true",
        )
        for value in required:
            with self.subTest(value=value):
                self.assertIn(value, self.source)
        self.assertEqual(
            self.source.count("Vq_cut = std::move(gw_vq_cut);"), 2
        )

    def test_all_reader_side_state_is_restored_on_success_and_failure(self) -> None:
        for value in (
            "n_irk_points = saved_n_irk_points;",
            "irk_points = saved_irk_points;",
            "irk_weight = saved_irk_weight;",
        ):
            with self.subTest(value=value):
                self.assertEqual(self.source.count(value), 2)

    def test_hartree_uses_only_dedicated_full_map(self) -> None:
        self.assertIn(
            "hartree_full_vq_cut, meanfield.get_n_kpoints(), Rlist,",
            self.source,
        )
        self.assertNotIn(
            "+                    Vq_cut, meanfield.get_n_kpoints(), Rlist, true",
            self.source,
        )

    def test_does_not_force_normal_reader_to_full_mode(self) -> None:
        self.assertNotIn(
            'oracle_env_bool("QSGW_ORACLE_UPDATE_HARTREE", false))\n     {',
            self.source,
        )


if __name__ == "__main__":
    unittest.main()
