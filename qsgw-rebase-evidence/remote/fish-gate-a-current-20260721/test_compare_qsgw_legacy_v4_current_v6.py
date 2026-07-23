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


def legacy_header(
    beta: str = "1", symmetry: str = "1", n_params: str = "-1",
    final_iteration: str = "2",
    use_shrink_abfs: str = "1",
    head: bool = False,
) -> str:
    return """# qsgw_contract_version 4
# oracle_kind legacy_scheme_a
# oracle_source_commit 7a7ff17f
# task qsgw
# fixed_basis immutable_reference
# qsgw_mixer linear
# qsgw_mixing_beta {beta}
# qsgw_min_iter {final_iteration}
# qsgw_max_iter {final_iteration}
# starting_vxc dft_only
# vxc_basis fixed_state
# qsgw_update_hartree 0
# qsgw_hartree_coulomb truncated
# qsgw_hartree_normalization legacy_extra_inverse_nk
# use_symmetry_gw {symmetry}
# use_symmetry_exx {symmetry}
# replace_w_head {replace_w_head}
# option_dielect_func {option_dielect_func}
# nfreq 6
# n_params_anacon {n_params}
# n_params_anacon_resample -1
# anacon_nfreq -1
# anacon_tfgrids_type -101
# use_shrink_abfs {use_shrink_abfs}
# use_fullcoul_exx 0
# use_fullcoul_eps 1
# use_fullcoul_wc 0
# constants_choice internal
# ac_policy direct_pade
# iter channel component spin kpoint frequency_index frequency_Ha row column real_value imag_value
""".format(
        beta=beta,
        symmetry=symmetry,
        n_params=n_params,
        final_iteration=final_iteration,
        use_shrink_abfs=use_shrink_abfs,
        replace_w_head="1" if head else "0",
        option_dielect_func="4" if head else "0",
    )


def current_header(
    mode: str = "none",
    beta: str = "0.2",
    symmetry: str = "exx_on_gw_on_rpa_on",
    head: bool = False,
) -> str:
    velocity = "fixed_basis_rotation" if head else "disabled_stage1"
    head_contract = "scf_grid_analytic_live" if head else "disabled_stage1"
    return """# qsgw_contract_version 6
# fixed_basis immutable_mf0
# live_update eigenvalues_wfc
# velocity {velocity}
# head {head_contract}
# wing disabled_stage1
# symmetry {symmetry}
# hartree disabled_stage1
# band disabled_stage1
# h_qsgw_cut disabled_non_band
# qsgw_input_contract qsgw_input.contract
# qsgw_input_contract_sha256 aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
# qsgw_mixer {mode}
# qsgw_mixing_beta {beta}
""".format(
        mode=mode,
        beta=beta,
        symmetry=symmetry,
        velocity=velocity,
        head_contract=head_contract,
    )


class AdapterContractTests(unittest.TestCase):
    def validate(
        self, legacy: str, now: str, mode: str,
        legacy_beta: float, current_beta: float,
        final_iteration: int = 2,
        allow_legacy_iteration_prefix: bool = False,
        expected_legacy_use_shrink_abfs: bool = True,
        expected_legacy_symmetry: str = "on",
        expected_current_symmetry: str = "on",
        expected_legacy_head: str = "off",
        expected_current_head: str = "off",
    ):
        return adapter._validate_actual_contracts(
            base_module=base,
            current_module=current,
            legacy_text=legacy,
            current_texts=(("matrix", now), ("eigen", now), ("iteration", now)),
            final_iteration=final_iteration,
            expected_mode=mode,
            expected_legacy_beta=legacy_beta,
            expected_current_beta=current_beta,
            expected_legacy_symmetry=expected_legacy_symmetry,
            expected_current_symmetry=expected_current_symmetry,
            expected_legacy_head=expected_legacy_head,
            expected_current_head=expected_current_head,
            allow_legacy_iteration_prefix=allow_legacy_iteration_prefix,
            expected_legacy_use_shrink_abfs=expected_legacy_use_shrink_abfs,
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

    def test_full_bz_contract_passes(self):
        result = self.validate(
            legacy_header(symmetry="0"),
            current_header(symmetry="exx_off_gw_off_rpa_off"),
            "none",
            1.0,
            0.2,
            expected_legacy_symmetry="off",
            expected_current_symmetry="off",
        )
        self.assertEqual(result["symmetry_mapping"], "legacy_off_to_current_off")

    def test_head_on_contract_passes(self):
        result = self.validate(
            legacy_header(head=True),
            current_header(head=True),
            "none",
            1.0,
            0.2,
            expected_legacy_head="on",
            expected_current_head="on",
        )
        self.assertEqual(result["head_mapping"], "legacy_on_to_current_on")

    def test_legacy_full_bz_to_current_symmetry_contract_passes(self):
        result = self.validate(
            legacy_header(symmetry="0"),
            current_header(symmetry="exx_on_gw_on_rpa_on"),
            "none",
            1.0,
            0.2,
            expected_legacy_symmetry="off",
            expected_current_symmetry="on",
        )
        self.assertEqual(result["legacy_symmetry"], "off")
        self.assertEqual(result["current_symmetry"], "on")

    def test_legacy_literal_nfreq_is_effectively_all_points(self):
        result = self.validate(
            legacy_header(n_params="6"), current_header(),
            "none", 1.0, 0.2,
        )
        self.assertEqual(result["legacy_declared_n_params_anacon"], 6)
        self.assertEqual(result["legacy_effective_n_params_anacon"], 6)

    def test_legacy_iteration_prefix_is_explicit(self):
        result = self.validate(
            legacy_header(final_iteration="2"), current_header(),
            "none", 1.0, 0.2,
            final_iteration=1,
            allow_legacy_iteration_prefix=True,
        )
        self.assertEqual(result["legacy_iteration_selection"], "prefix")

    def test_legacy_iteration_prefix_requires_opt_in(self):
        with self.assertRaisesRegex(ValueError, "iteration bound differs"):
            self.validate(
                legacy_header(final_iteration="2"), current_header(),
                "none", 1.0, 0.2,
                final_iteration=1,
            )

    def test_non_all_point_legacy_pade_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "does not use all"):
            self.validate(
                legacy_header(n_params="4"), current_header(),
                "none", 1.0, 0.2,
            )

    def test_wrong_legacy_symmetry_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "contract mismatch"):
            self.validate(
                legacy_header(symmetry="0"), current_header(),
                "none", 1.0, 0.2,
            )

    def test_wrong_current_symmetry_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "contract mismatch"):
            self.validate(
                legacy_header(symmetry="0"), current_header(),
                "none", 1.0, 0.2,
                expected_legacy_symmetry="off",
                expected_current_symmetry="off",
            )

    def test_full_abf_legacy_contract_requires_explicit_opt_in(self):
        with self.assertRaisesRegex(ValueError, "contract mismatch"):
            self.validate(
                legacy_header(use_shrink_abfs="0"), current_header(),
                "none", 1.0, 0.2,
            )
        result = self.validate(
            legacy_header(use_shrink_abfs="0"), current_header(),
            "none", 1.0, 0.2,
            expected_legacy_use_shrink_abfs=False,
        )
        self.assertFalse(result["legacy_use_shrink_abfs"])

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
        self.assertIn("# headwing disabled_stage1\n", current_traces[0])
        self.assertNotIn("# head ", current_traces[0])
        self.assertNotIn("# wing ", current_traces[0])
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
        self.assertIn("# headwing disabled_stage1\n", normalized)
        self.assertNotIn("# head ", normalized)
        self.assertNotIn("# wing ", normalized)
        self.assertIn("# qsgw_mixer linear\n", normalized)
        self.assertIn(data, normalized)

    def test_self_validator_normalization_preserves_live_head(self):
        normalized = adapter._normalized_for_v5_self_validators(
            current_header(head=True)
        )
        self.assertIn("# velocity fixed_basis_rotation\n", normalized)
        self.assertIn("# headwing scf_grid_analytic_live\n", normalized)

    def test_aims_ev_trace_is_rescaled_to_internal_constant(self):
        aims_ha2ev = 27.2113845
        value_ha = -65.0
        trace = (
            "# header\n"
            "0 0 0 0 0 0 0 0 {:.17e}\n".format(value_ha * aims_ha2ev)
        )
        normalized = adapter._rescale_trace_ev_columns(
            trace,
            source_ha2ev=aims_ha2ev,
            columns=(8,),
        )
        converted_ev = float(normalized.splitlines()[1].split()[8])
        self.assertAlmostEqual(
            converted_ev / adapter.INTERNAL_HA2EV,
            value_ha,
            places=13,
        )

    def test_invalid_ha2ev_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "current_ha2ev"):
            adapter._rescale_trace_ev_columns(
                "0 0\n",
                source_ha2ev=0.0,
                columns=(1,),
            )


if __name__ == "__main__":
    unittest.main()
