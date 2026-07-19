#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent
MODULE_PATH = HERE / "plot_qsgw_linear_convergence_v1.py"
SPEC = importlib.util.spec_from_file_location("convergence_plot", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def convergence_csv() -> str:
    return "\n".join(
        (
            "iteration,gap_ev,residual_l2_ha,old_residual_l2_ha,reported_residual_l2_ha,beta",
            "0,1.0,0,0,0,0.2",
            "1,1.5,100,100.0000000001,100,0.2",
            "2,1.8,80,80.0000000001,80,0.2",
            "3,2.0,64,64.0000000001,64,0.2",
        )
    ) + "\n"


class ConvergencePlotTest(unittest.TestCase):
    def test_summary_reports_monotonic_decay(self):
        rows = MODULE.load_rows(convergence_csv())
        report = MODULE.summarize(rows)
        self.assertTrue(report["passed"])
        self.assertTrue(report["residual_monotonic_decrease"])
        self.assertAlmostEqual(
            report["geometric_mean_residual_ratio"], 0.8
        )

    def test_svg_contains_both_panels_and_curves(self):
        rows = MODULE.load_rows(convergence_csv())
        svg = MODULE.render_svg(rows, MODULE.summarize(rows))
        self.assertIn("Hamiltonian residual L2", svg)
        self.assertIn("Gap (eV)", svg)
        self.assertGreaterEqual(svg.count("<polyline"), 3)

    def test_cli_writes_svg_and_json(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "trajectory.csv"
            svg = root / "curve.svg"
            report = root / "curve.json"
            source.write_text(convergence_csv(), encoding="utf-8")
            completed = subprocess.run(
                [
                    sys.executable, "-B", str(MODULE_PATH),
                    str(source), str(svg), str(report),
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertTrue(svg.read_text(encoding="utf-8").startswith("<svg"))
            self.assertTrue(
                json.loads(report.read_text(encoding="utf-8"))["passed"]
            )


if __name__ == "__main__":
    unittest.main()
