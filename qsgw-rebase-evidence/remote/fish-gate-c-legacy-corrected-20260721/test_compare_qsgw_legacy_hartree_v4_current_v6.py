#!/usr/bin/env python3
"""Contract tests for the legacy-v4/current-v6 Hartree adapter."""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
ADAPTER_PATH = HERE / "compare_qsgw_legacy_hartree_v4_current_v6.py"
BASE_PATH = (
    ROOT
    / "qsgw-rebase-evidence"
    / "remote"
    / "dongfang-gates-20260715-7d69a18c"
    / "compare_qsgw_component_traces_v3.py"
)
CURRENT_PATH = ROOT / "regression_tests" / "backend" / "comparisons" / "cmp_qsgw.py"


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


adapter = load(ADAPTER_PATH, "hartree_adapter_under_test")
base = load(BASE_PATH, "hartree_base_under_test")
current = load(CURRENT_PATH, "hartree_current_under_test")


def legacy_header(
    *, beta: str = "0.2", hartree: str = "1", normalization: str = "legacy_extra_inverse_nk"
) -> str:
    return """# qsgw_contract_version 4
# oracle_kind legacy_scheme_a
# oracle_source_commit e08f4a13
# task qsgw
# fixed_basis immutable_reference
# qsgw_mixer linear
# qsgw_mixing_beta {beta}
# qsgw_min_iter 2
# qsgw_max_iter 2
# starting_vxc dft_only
# vxc_basis fixed_state
# qsgw_update_hartree {hartree}
# qsgw_hartree_coulomb truncated
# qsgw_hartree_normalization {normalization}
# use_symmetry_gw 0
# use_symmetry_exx 0
# replace_w_head 0
# option_dielect_func 0
# nfreq 6
# n_params_anacon -1
# n_params_anacon_resample -1
# anacon_nfreq -1
# anacon_tfgrids_type -101
# use_shrink_abfs 0
# use_fullcoul_exx 0
# use_fullcoul_eps 1
# use_fullcoul_wc 0
# constants_choice internal
# ac_policy direct_pade
# iter channel component spin kpoint frequency_index frequency_Ha row column real_value imag_value
""".format(beta=beta, hartree=hartree, normalization=normalization)


def current_header(
    *, normalization: str = "legacy_extra_inverse_nk", symmetry: str = "exx_off_gw_off_rpa_off"
) -> str:
    return """# qsgw_contract_version 6
# fixed_basis immutable_mf0
# live_update eigenvalues_wfc
# velocity disabled_stage1
# headwing disabled_stage1
# symmetry {symmetry}
# hartree delta_density
# hartree_coulomb truncated
# hartree_normalization {normalization}
# band disabled_stage1
# h_qsgw_cut disabled_non_band
# qsgw_input_contract qsgw_input.hartree-truncated.contract
# qsgw_input_contract_sha256 aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
# qsgw_mixer linear
# qsgw_mixing_beta 0.2
""".format(normalization=normalization, symmetry=symmetry)


class HartreeAdapterContractTests(unittest.TestCase):
    def validate(self, legacy: str, now: str):
        return adapter._validate_actual_contracts(
            base_module=base,
            current_module=current,
            legacy_text=legacy,
            current_texts=(("matrix", now), ("eigen", now), ("iteration", now)),
            final_iteration=2,
            expected_mode="linear",
            expected_legacy_beta=0.2,
            expected_current_beta=0.2,
        )

    def test_hartree_parity_contract_passes(self) -> None:
        result = self.validate(legacy_header(), current_header())
        self.assertTrue(result["passed"])
        self.assertEqual(result["hartree_normalization"], "legacy_extra_inverse_nk")
        self.assertEqual(result["symmetry"], "full_bz_no_symmetry")

    def test_legacy_hartree_off_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "contract mismatch"):
            self.validate(legacy_header(hartree="0"), current_header())

    def test_current_weighted_default_is_rejected_in_parity_lane(self) -> None:
        with self.assertRaisesRegex(ValueError, "contract mismatch"):
            self.validate(
                legacy_header(), current_header(normalization="weighted_occupations")
            )

    def test_symmetry_on_current_trace_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "contract mismatch"):
            self.validate(
                legacy_header(),
                current_header(symmetry="exx_on_gw_on_rpa_on"),
            )

    def test_alignment_normalization_changes_headers_only(self) -> None:
        row = "0 0 h0 0 0 0 0.0 0 0 1.0 0.0\n"
        legacy, current_traces = adapter._normalized_for_frozen_aligner(
            legacy_header() + row,
            (current_header() + row,) * 3,
        )
        self.assertIn("# n_params_anacon 6\n", legacy)
        self.assertNotIn("# n_params_anacon -1\n", legacy)
        self.assertIn(row, legacy)
        self.assertIn("# qsgw_contract_version 5\n", current_traces[0])
        self.assertIn("# symmetry unsupported_full_bz_only\n", current_traces[0])
        self.assertIn("# hartree delta_density\n", current_traces[0])
        self.assertIn("# qsgw_mixer linear\n", current_traces[0])
        self.assertIn(row, current_traces[0])
        base._validate_legacy_current_contract(
            legacy,
            (
                ("matrix", current_traces[0]),
                ("eigen", current_traces[1]),
                ("iteration", current_traces[2]),
            ),
            2,
            False,
            False,
            0.2,
            "linear",
            True,
            False,
        )

    def test_self_validator_normalization_preserves_hartree_contract(self) -> None:
        row = "0 0 h0 0 0 0 0.0 0 0 1.0 0.0\n"
        normalized = adapter._normalized_for_v5_self_validators(
            current_header() + row
        )
        self.assertIn("# qsgw_contract_version 5\n", normalized)
        self.assertIn("# symmetry unsupported_full_bz_only\n", normalized)
        self.assertIn("# hartree delta_density\n", normalized)
        self.assertIn("# hartree_normalization legacy_extra_inverse_nk\n", normalized)
        self.assertIn("# qsgw_mixer linear\n", normalized)
        self.assertIn(row, normalized)


if __name__ == "__main__":
    unittest.main()
