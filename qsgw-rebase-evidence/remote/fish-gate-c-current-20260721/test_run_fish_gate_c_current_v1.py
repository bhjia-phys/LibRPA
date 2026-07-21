#!/usr/bin/env python3

from __future__ import annotations

import unittest
from pathlib import Path


RUNNER = Path(__file__).resolve().parent / "run_fish_gate_c_current_v1.sh"


class FishGateCRunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = RUNNER.read_text(encoding="utf-8")

    def test_binds_gate0_bundle_executable_and_contract_hashes(self) -> None:
        required = (
            "CANDIDATE_GATE0_PROVENANCE_SHA256",
            "CANDIDATE_EXE_SHA256",
            "HARTREE_BUNDLE_PROVENANCE_SHA256",
            "HARTREE_BUNDLE_OUTPUT_SHA256",
            "HARTREE_DATASET_MANIFEST_SHA256",
            "GRID_BASE_CONTRACT_SHA256",
            "GRID_HARTREE_FULL_CONTRACT_SHA256",
            "BAND_HARTREE_FULL_CONTRACT_SHA256",
            "RUNNER_SHA256",
            "sha256sum --check --quiet OUTPUT_SHA256SUMS.txt",
            "sha256sum --check --quiet DATASET_SHA256SUMS.txt",
        )
        for value in required:
            with self.subTest(value=value):
                self.assertIn(value, self.source)

    def test_runs_exact_two_round_grid_and_band_hartree_cases(self) -> None:
        required = (
            "run_case grid qsgw qsgw_input.hartree-full.contract",
            "run_case band qsgw_band qsgw_band_input.hartree-full.contract",
            'write_librpa_input "$mode_root" "$task" "$contract" 2 true',
            "run_hartree_off_control",
            'write_librpa_input "$mode_root" qsgw qsgw_input.contract 1 false',
            "qsgw_min_iter = $iterations",
            "qsgw_max_iter = $iterations",
            "qsgw_update_hartree = $update_hartree",
            "qsgw_hartree_coulomb = full",
            "qsgw_hartree_normalization = weighted_occupations",
            "qsgw_mixer = linear",
            "qsgw_mixing_beta = 0.2",
        )
        for value in required:
            with self.subTest(value=value):
                self.assertIn(value, self.source)

    def test_symmetry_on_and_headwing_off_are_explicit(self) -> None:
        required = (
            "replace_w_head = false",
            "option_dielect_func = 0",
            "use_abacus_exx_symmetry = true",
            "use_abacus_gw_symmetry = true",
            "use_symmetry_exx = true",
            "use_symmetry_gw = true",
            "use_symmetry_rpa = true",
            "# headwing disabled_stage1",
            "# symmetry exx_on_gw_on_rpa_on",
        )
        for value in required:
            with self.subTest(value=value):
                self.assertIn(value, self.source)

    def test_uses_all_hartree_and_band_observers(self) -> None:
        required = (
            "validate_qsgw_hartree_dump_v2.py",
            "validate_qsgw_hartree_contraction_v1.py",
            "validate_qsgw_hartree_trace_v6.py",
            "validate_qsgw_band_v6.py",
            "compare_qsgw_grid_channels_v1.py",
            "compare_qsgw_hartree_null_delta_v1.py",
            "LIBRPA_QSGW_HARTREE_DUMP_DIR",
            "iteration1_null_delta_side_effect_guard=true",
            "independent_hartree_contraction_tasks=qsgw,qsgw_band",
            '--dump-call "$dump_root/call_002"',
            "--coulomb-prefix coulomb_mat_",
            "legacy_same_dataset_acceptance=false",
        )
        for value in required:
            with self.subTest(value=value):
                self.assertIn(value, self.source)

    def test_runner_does_not_open_remote_connections_or_submit_jobs(self) -> None:
        for forbidden in ("ssh ", "scp ", "sbatch ", "srun "):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, self.source)


if __name__ == "__main__":
    unittest.main()
