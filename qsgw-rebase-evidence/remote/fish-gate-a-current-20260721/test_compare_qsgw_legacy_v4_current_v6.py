#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
ADAPTER_PATH = HERE / "compare_qsgw_legacy_v4_current_v6.py"
BASE_PATH = (
    ROOT
    / "qsgw-rebase-evidence"
    / "remote"
    / "fish-gate-a-symmetry-20260720"
    / "compare_qsgw_component_traces-v4-c3daf072.py"
)
CURRENT_PATH = ROOT / "regression_tests" / "backend" / "comparisons" / "cmp_qsgw.py"


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


adapter = load(ADAPTER_PATH, "adapter_under_test")
base = load(BASE_PATH, "base_under_test")
current = load(CURRENT_PATH, "current_under_test")


def legacy_header(beta: str = "1", symmetry: str = "1") -> str:
    return """# qsgw_contract_version 4
# oracle_kind legacy_scheme_a
# oracle_source_commit 7a7ff17f
# task qsgw
# fixed_basis immutable_reference
# qsgw_mixer linear
# qsgw_mixing_beta {beta}
# qsgw_min_iter 2
# qsgw_max_iter 2
# starting_vxc dft_only
# vxc_basis fixed_state
# qsgw_update_hartree 0
# qsgw_hartree_coulomb truncated
# qsgw_hartree_normalization legacy_extra_inverse_nk
# use_symmetry_gw {symmetry}
# use_symmetry_exx {symmetry}
# replace_w_head 0
# option_dielect_func 0
# nfreq 6
# n_params_anacon -1
# n_params_anacon_resample -1
# anacon_nfreq -1
# anacon_tfgrids_type -101
# use_shrink_abfs 1
# use_fullcoul_exx 0
# use_fullcoul_eps 1
# use_fullcoul_wc 0
# constants_choice internal
# ac_policy direct_pade
# iter channel component spin kpoint frequency_index frequency_Ha row column real_value imag_value
""".format(beta=beta, symmetry=symmetry)


def current_header(mode: str = "none", beta: str = "0.2") -> str:
    return """# qsgw_contract_version 6
# fixed_basis immutable_mf0
# live_update eigenvalues_wfc
# velocity disabled_stage1
# headwing disabled_stage1
# symmetry exx_on_gw_on_rpa_on
# hartree disabled_stage1
# band disabled_stage1
# h_qsgw_cut disabled_non_band
# qsgw_input_contract qsgw_input.contract
# qsgw_input_contract_sha256 aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
# qsgw_mixer {mode}
# qsgw_mixing_beta {beta}
""".format(mode=mode, beta=beta)


class AdapterContractTests(unittest.TestCase):
    def validate(
        self, legacy: str, now: str, mode: str,
        legacy_beta: float, current_beta: float,
    ):
        return adapter._validate_actual_contracts(
            base_module=base,
            current_module=current,
            legacy_text=legacy,
            current_texts=(("matrix", now), ("eigen", now), ("iteration", now)),
            final_iteration=2,
            expected_mode=mode,
            expected_legacy_beta=legacy_beta,
            expected_current_beta=current_beta,
        )

    def test_none_contract_passes(self):
        result = self.validate(
            legacy_header(), current_header(), "none", 1.0, 0.2
        )
        self.assertTrue(result["passed"])
        self.assertEqual(result["current_version"], 6)

    def test_linear_contract_passes(self):
        result = self.validate(
            legacy_header("0.2"), current_header("linear", "0.2"),
            "linear", 0.2, 0.2,
        )
        self.assertTrue(result["passed"])

    def test_wrong_legacy_symmetry_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "contract mismatch"):
            self.validate(
                legacy_header(symmetry="0"), current_header(),
                "none", 1.0, 0.2,
            )

    def test_wrong_current_mode_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "contract mismatch"):
            self.validate(
                legacy_header(), current_header("linear"),
                "none", 1.0, 0.2,
            )

    def test_normalization_changes_headers_only(self):
        data = "0 0 h0 0 0 0 0.0 0 0 1.0 0.0\n"
        legacy, current_traces = adapter._normalized_for_frozen_aligner(
            legacy_header() + data,
            (current_header() + data,) * 3,
        )
        self.assertIn("# qsgw_contract_version 5\n", current_traces[0])
        self.assertIn("# symmetry unsupported_full_bz_only\n", current_traces[0])
        self.assertIn(data, legacy)
        self.assertIn(data, current_traces[0])
        base._validate_legacy_current_contract(
            legacy,
            (("matrix", current_traces[0]), ("eigen", current_traces[1]),
             ("iteration", current_traces[2])),
            2,
            False,
            False,
        )

    def test_self_validator_normalization_preserves_linear_mixer(self):
        data = "0 0 h0 0 0 0 0.0 0 0 1.0 0.0\n"
        normalized = adapter._normalized_for_v5_self_validators(
            current_header("linear", "0.2") + data
        )
        self.assertIn("# qsgw_contract_version 5\n", normalized)
        self.assertIn("# symmetry input_kstar_live\n", normalized)
        self.assertIn("# qsgw_mixer linear\n", normalized)
        self.assertIn(data, normalized)


if __name__ == "__main__":
    unittest.main()
