#!/usr/bin/env python3

from __future__ import annotations

import importlib
import shutil
import sys
import unittest
import uuid
from pathlib import Path


CURRENT_HEADER = """# qsgw_contract_version 6
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
# qsgw_mixer none
# qsgw_mixing_beta 0.2
"""


class CurrentV6ClosureAdapterTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.root = Path.cwd()
        cls.tool_dir = cls.root / "__test_tmp" / uuid.uuid4().hex
        cls.tool_dir.mkdir(parents=True)
        sources = {
            "compare_qsgw_component_traces.py": (
                cls.root / "compare_qsgw_component_traces_v6_adapter.py"
            ),
            "compare_qsgw_component_traces_v4.py": (
                cls.root.parent
                / "fish-gate-a-symmetry-20260720"
                / "observer-tools-v1"
                / "compare_qsgw_component_traces.py"
            ),
            "cmp_qsgw_v6.py": (
                cls.root.parent.parent.parent
                / "regression_tests"
                / "backend"
                / "comparisons"
                / "cmp_qsgw.py"
            ),
            "validate_qsgw_trace_closure.py": (
                cls.root.parent
                / "fish-gate-a-symmetry-20260720"
                / "observer-tools-v1"
                / "validate_qsgw_trace_closure-v3-4a5de94e.py"
            ),
            "validate_qsgw_fixed_basis.py": (
                cls.root.parent
                / "fish-gate-a-symmetry-20260720"
                / "observer-tools-v1"
                / "validate_qsgw_fixed_basis.py"
            ),
        }
        for name, source in sources.items():
            shutil.copyfile(source, cls.tool_dir / name)
        sys.path.insert(0, str(cls.tool_dir))
        cls.adapter = importlib.import_module("compare_qsgw_component_traces")
        cls.closure = importlib.import_module("validate_qsgw_trace_closure")
        cls.fixed_basis = importlib.import_module("validate_qsgw_fixed_basis")

    @classmethod
    def tearDownClass(cls) -> None:
        sys.path.remove(str(cls.tool_dir))
        for name in (
            "compare_qsgw_component_traces",
            "compare_qsgw_component_traces_v4",
            "cmp_qsgw_v6",
            "validate_qsgw_trace_closure",
            "validate_qsgw_fixed_basis",
        ):
            sys.modules.pop(name, None)
        shutil.rmtree(cls.tool_dir)

    def test_maps_compact_v6_contract_for_closure_observer(self) -> None:
        contract = self.adapter.parse_contract(
            CURRENT_HEADER, "current trace", require_current=True
        )
        self.assertEqual(contract["qsgw_contract_version"], 6)
        self.assertEqual(contract["task"], "qsgw")
        self.assertEqual(contract["qsgw_update_hartree"], "0")
        self.assertEqual(contract["symmetry"], "exx_on_gw_on_rpa_on")

        closure_contract = self.closure._parse_closure_contract(
            CURRENT_HEADER, True
        )
        self.assertEqual(closure_contract["qsgw_contract_version"], 6)
        self.assertEqual(closure_contract["task"], "qsgw")
        self.assertIs(
            self.fixed_basis._matrix_groups,
            self.adapter._matrix_groups,
        )

    def test_preserves_hartree_and_band_semantics(self) -> None:
        text = CURRENT_HEADER.replace(
            "# hartree disabled_stage1",
            "# hartree delta_density\n# hartree_coulomb full\n"
            "# hartree_normalization weighted_occupations",
        ).replace(
            "# band disabled_stage1\n# h_qsgw_cut disabled_non_band",
            "# band fixed_reference_operator_fourier_live\n"
            "# h_qsgw_cut band_postprocess\n"
            "# qsgw_band0_unoccupied_keep 10\n"
            "# qsgw_band0_cut_mode 0\n"
            "# qsgw_band0_cut_shift_ha 20",
        )
        contract = self.adapter.parse_contract(text, "current band trace")
        self.assertEqual(contract["task"], "qsgw_band")
        self.assertEqual(contract["qsgw_update_hartree"], "1")

    def test_rejects_incomplete_v6_contract(self) -> None:
        with self.assertRaisesRegex(ValueError, "missing QSGW contract keys"):
            self.adapter.parse_contract(
                CURRENT_HEADER.replace("# fixed_basis immutable_mf0\n", ""),
                "current trace",
            )


if __name__ == "__main__":
    unittest.main()
