#!/usr/bin/env python3
"""Tests for header-only normalization of native QSGW v6 traces."""

from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path


SOURCE = Path(__file__).with_name("normalize_qsgw_v6_self_traces_v1.py")


def load_module():
    spec = importlib.util.spec_from_file_location("qsgw_v6_normalizer", SOURCE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {SOURCE}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def trace(kind: str, numeric: str, *, symmetry: str = "exx_on_gw_on_rpa_on") -> str:
    return "\n".join(
        (
            "# qsgw_contract_version 6",
            "# fixed_basis immutable_mf0",
            "# live_update eigenvalues_wfc",
            "# velocity disabled_stage1",
            "# headwing disabled_stage1",
            f"# symmetry {symmetry}",
            "# hartree disabled_stage1",
            "# band disabled_stage1",
            "# qsgw_input_contract /frozen/qsgw_input.contract",
            "# qsgw_input_contract_sha256 " + "a" * 64,
            "# qsgw_mixer linear",
            "# qsgw_mixing_beta 0.2",
            f"# trace_kind {kind}",
            numeric,
            "",
        )
    )


class NormalizeQsgwV6SelfTracesTests(unittest.TestCase):
    def test_rewrites_only_contract_version_and_symmetry(self) -> None:
        module = load_module()
        source = trace("matrix", "0 0 h0 0 0 -1 0 0 1.25 0.0")

        normalized = module.normalize_trace(source)

        self.assertIn("# qsgw_contract_version 5\n", normalized)
        self.assertIn("# symmetry input_kstar_live\n", normalized)
        self.assertNotIn("# qsgw_contract_version 6\n", normalized)
        self.assertNotIn("# symmetry exx_on_gw_on_rpa_on\n", normalized)
        source_numeric = [line for line in source.splitlines() if not line.startswith("#")]
        normalized_numeric = [
            line for line in normalized.splitlines() if not line.startswith("#")
        ]
        self.assertEqual(source_numeric, normalized_numeric)
        for header in (
            "# fixed_basis immutable_mf0",
            "# qsgw_mixer linear",
            "# qsgw_mixing_beta 0.2",
        ):
            self.assertIn(header, normalized)

    def test_rejects_non_v6_or_non_symmetry_trace(self) -> None:
        module = load_module()
        with self.assertRaisesRegex(ValueError, "contract version 6"):
            module.normalize_trace(trace("matrix", "0 row").replace("version 6", "version 5"))
        with self.assertRaisesRegex(ValueError, "symmetry-on"):
            module.normalize_trace(
                trace("matrix", "0 row", symmetry="unsupported_full_bz_only")
            )

    def test_cli_normalizes_all_three_traces_and_writes_audit(self) -> None:
        module = load_module()
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temporary:
            root = Path(temporary)
            inputs = []
            outputs = []
            for name, numeric in (
                ("matrix", "0 0 h0 0 0 -1 0 0 1.25 0.0"),
                ("eigenvalues", "0 0 0 0 0 0 0 1 -2.0"),
                ("iterations", "0 0 0 0 0 0 8 none none 0.2 0 0 0 0 0 [] -"),
            ):
                source = root / f"{name}.dat"
                destination = root / f"{name}-v5.dat"
                source.write_text(trace(name, numeric), encoding="utf-8")
                inputs.append(source)
                outputs.append(destination)
            report = root / "normalization.json"

            result = module.normalize_files(
                matrix_input=inputs[0],
                eigenvalue_input=inputs[1],
                iteration_input=inputs[2],
                matrix_output=outputs[0],
                eigenvalue_output=outputs[1],
                iteration_output=outputs[2],
                report_output=report,
            )

            self.assertTrue(result["passed"])
            self.assertEqual(result["scope"], "contract_headers_only_numeric_rows_unchanged")
            self.assertEqual(result["trace_count"], 3)
            self.assertTrue(report.is_file())
            for output in outputs:
                self.assertIn("# qsgw_contract_version 5", output.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
