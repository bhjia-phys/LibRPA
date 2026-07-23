#!/usr/bin/env python3

from __future__ import annotations

import unittest
from pathlib import Path
import re


RUNNER = Path(__file__).with_name("run_fish_si_k444_head_band_v1.sh")
REPOSITORY = Path(__file__).resolve().parents[3]


def parse_input_block(text: str, side: str) -> dict[str, str]:
    match = re.search(
        rf'cat >"\${side}/librpa\.in" <<EOF\n(.*?)\nEOF',
        text,
        re.DOTALL,
    )
    if match is None:
        raise AssertionError(f"missing {side} librpa.in block")
    result = {}
    for raw in match.group(1).splitlines():
        key, value = (field.strip() for field in raw.split("=", 1))
        if key in result:
            raise AssertionError(f"duplicate {key} in {side} input")
        result[key] = value
    return result


class FishRunnerContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.text = RUNNER.read_text(encoding="ascii")

    def test_enforces_requested_physics_scope(self) -> None:
        self.assertIn("replace_w_head = true", self.text)
        self.assertIn("option_dielect_func = 4", self.text)
        self.assertIn("qsgw_update_hartree = false", self.text)
        self.assertIn("qsgw_mixer = none", self.text)
        self.assertNotIn("option_dielect_func = 3", self.text)
        self.assertNotIn("qsgw_mixer = linear", self.text)
        self.assertNotIn("qsgw_mixer = pulay", self.text)

    def test_requires_full_bz_k444_head_and_band_contract(self) -> None:
        self.assertGreaterEqual(
            self.text.count("grep -Fqx 'n_scf_kpoints 64'"), 3
        )
        self.assertIn("grep -Fqx 'n_headwing_kpoints 64'", self.text)
        self.assertIn("grep -Fqx 'n_band_kpoints 201'", self.text)
        self.assertIn("headwing_grid scf", self.text)
        self.assertIn("headwing_update fixed_basis_rotation", self.text)
        self.assertIn("band_update fixed_basis_rotation", self.text)
        self.assertNotIn("band_update operator_fourier", self.text)

    def test_acceptance_is_only_multiround_bands_and_gap(self) -> None:
        self.assertIn("compare_qsgw_band_iterations_v1.py", self.text)
        self.assertIn("--occupied-bands 4", self.text)
        self.assertIn("--energy-tolerance-ev 1e-4", self.text)
        self.assertIn("--gap-tolerance-ev 2e-4", self.text)
        self.assertNotIn("residual_tolerance", self.text)
        self.assertNotIn("mixing_coefficient", self.text)

    def test_builds_hash_checked_legacy_band_vxc_view(self) -> None:
        self.assertIn("build_legacy_band_vxc_view_v1.py", self.text)
        self.assertIn("test_build_legacy_band_vxc_view_v1.py", self.text)
        self.assertIn("qsgw_vxc_band.v2.manifest", self.text)
        self.assertIn("LEGACY_BAND_VXC_VIEW.json", self.text)
        self.assertIn(
            'find "$overlay" -maxdepth 1 -type f', self.text
        )
        self.assertIn(
            "-name 'band_vxcs1k*_nao.txt'", self.text
        )
        self.assertIn(
            'test ! -s "$legacy/librpa.stderr" ||', self.text
        )
        self.assertIn(
            "! grep -Fq 'VXC_band file not found'", self.text
        )

    def test_binds_legacy_shared_libraries_only_for_legacy_run(self) -> None:
        self.assertIn(
            ': "${LEGACY_BUILD_DIR:?legacy build directory is required}"',
            self.text,
        )
        self.assertIn(
            'test -f "$LEGACY_BUILD_DIR/qsgw/libqsgw.so.0.3.0"',
            self.text,
        )
        self.assertIn(
            'test -f "$LEGACY_BUILD_DIR/src/librpa.so.0.3.0"',
            self.text,
        )
        self.assertIn(
            'export LD_LIBRARY_PATH="$LEGACY_BUILD_DIR/qsgw:'
            '$LEGACY_BUILD_DIR/src${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"',
            self.text,
        )
        legacy_block = self.text.split('cd "$legacy"', 1)[1].split(
            'cd "$candidate"', 1
        )[0]
        candidate_block = self.text.split('cd "$candidate"', 1)[1]
        self.assertIn("export LD_LIBRARY_PATH=", legacy_block)
        self.assertNotIn("export LD_LIBRARY_PATH=", candidate_block)

    def test_legacy_and_candidate_numerical_inputs_are_identical(self) -> None:
        legacy = parse_input_block(self.text, "legacy")
        candidate = parse_input_block(self.text, "candidate")
        legacy_controls = {"max_iter"}
        candidate_controls = {
            "qsgw_input_contract",
            "qsgw_mixer",
            "qsgw_min_iter",
            "qsgw_max_iter",
            "qsgw_write_iteration_matrices",
            "qsgw_update_hartree",
        }
        self.assertEqual(set(legacy) - legacy_controls,
                         set(candidate) - candidate_controls)
        for key in set(legacy) - legacy_controls:
            self.assertEqual(legacy[key], candidate[key], key)
        self.assertEqual(legacy["max_iter"], "$iterations")
        self.assertEqual(candidate["qsgw_min_iter"], "$iterations")
        self.assertEqual(candidate["qsgw_max_iter"], "$iterations")
        self.assertEqual(candidate["qsgw_mixer"], "none")
        self.assertEqual(candidate["qsgw_update_hartree"], "false")

    def test_none_mode_is_a_direct_raw_hamiltonian_update(self) -> None:
        source = (REPOSITORY / "driver" / "tasks" / "qsgw.cpp").read_text(
            encoding="utf-8"
        )
        self.assertIn('if (driver_params.qsgw_mixer == "linear")', source)
        self.assertIn("if (mixer)", source)
        self.assertIn("mixed_hamiltonian = raw;", source)
        self.assertIn(
            "if (compute_band) mixed_band_hamiltonian = raw_band;", source
        )


if __name__ == "__main__":
    unittest.main()
