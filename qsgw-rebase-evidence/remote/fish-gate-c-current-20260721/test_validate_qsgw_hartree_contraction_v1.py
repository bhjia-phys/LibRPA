#!/usr/bin/env python3
"""Synthetic tests for the independent QSGW Hartree contraction observer."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


MODULE = Path(__file__).resolve().parent / "validate_qsgw_hartree_contraction_v1.py"


class _FixedDirectory:
    def __init__(self, path: Path):
        self.path = path

    def __enter__(self) -> str:
        return str(self.path)

    def __exit__(self, *_args) -> None:
        return None


def _write_fixture(
    root: Path,
    *,
    normalization: str = "weighted_occupations",
    hartree_perturbation: float = 0.0,
    duplicate_gamma: bool = False,
) -> dict[str, Path]:
    input_dir = root / "input"
    call_dir = root / "dump" / "call_002"
    input_dir.mkdir(parents=True, exist_ok=True)
    call_dir.mkdir(parents=True, exist_ok=True)

    (input_dir / "Cs_data_0.txt").write_text(
        "1 1\n"
        "1 1 0 0 0 1 1 1\n"
        "2.00000000000000000e+00\n",
        encoding="utf-8",
        newline="\n",
    )
    (input_dir / "coulomb_cut_0.txt").write_text(
        "1\n"
        "1 1 1 1 1 1 1.0\n"
        "1.50000000000000000e+00 0.00000000000000000e+00\n",
        encoding="utf-8",
        newline="\n",
    )

    manifest = (
        "schema=qsgw_hartree_pipeline_dump_v1\n"
        f"normalization={normalization}\n"
        "kpoint_count=2\n"
        "translation_count=2\n"
        "period=2 1 1\n"
        "atom_ao_sizes=0:1 \n"
        "density_delta_k_count=2\n"
        "hartree_k_atom_count=1\n"
        "hartree_r_atom_count=1\n"
        "density_delta_k_file=density_delta_k.txt\n"
        "density_delta_k_columns=kpoint row column real imag\n"
        "hartree_k_file=hartree_k.txt\n"
        "hartree_k_columns=atom_i atom_j kpoint row column real imag\n"
        "hartree_r_file=hartree_r.txt\n"
        "hartree_r_columns=atom_i atom_j R_x R_y R_z row column real imag\n"
        "full_kpoints_file=full_kpoints.txt\n"
        "full_kpoints_columns=index kx ky kz\n"
        "translations_file=translations.txt\n"
        "translations_columns=index R_x R_y R_z\n"
        "bvk_remap_file=bvk_remap.txt\n"
        "bvk_remap_columns=atom_i atom_j source_R_x source_R_y source_R_z "
        "target_index target_count target_R_x target_R_y target_R_z\n"
        "bvk_remap_source_count=0\n"
    )
    (call_dir / "manifest.txt").write_text(
        manifest, encoding="utf-8", newline="\n"
    )
    second_k = "1.0" if duplicate_gamma else "0.5"
    (call_dir / "full_kpoints.txt").write_text(
        "# index kx ky kz\n"
        "0 0.0 0.0 0.0\n"
        f"1 {second_k} 0.0 0.0\n",
        encoding="utf-8",
        newline="\n",
    )
    (call_dir / "translations.txt").write_text(
        "# index R_x R_y R_z\n0 -1 0 0\n1 0 0 0\n",
        encoding="utf-8",
        newline="\n",
    )
    (call_dir / "bvk_remap.txt").write_text(
        "# atom_i atom_j source_R_x source_R_y source_R_z target_index "
        "target_count target_R_x target_R_y target_R_z\n",
        encoding="utf-8",
        newline="\n",
    )
    (call_dir / "density_delta_k.txt").write_text(
        "# kpoint row column real imag\n"
        "0 0 0 1.00000000000000000e+00 0.0\n"
        "1 0 0 3.00000000000000000e+00 0.0\n",
        encoding="utf-8",
        newline="\n",
    )
    expected = 96.0 if normalization == "weighted_occupations" else 48.0
    expected += hartree_perturbation
    (call_dir / "hartree_k.txt").write_text(
        "# atom_i atom_j kpoint row column real imag\n"
        f"0 0 0 0 0 {expected:.17e} 0.0\n"
        f"0 0 1 0 0 {expected:.17e} 0.0\n",
        encoding="utf-8",
        newline="\n",
    )
    (call_dir / "hartree_r.txt").write_text(
        "# atom_i atom_j R_x R_y R_z row column real imag\n"
        "0 0 -1 0 0 0 0 0.0 0.0\n"
        f"0 0 0 0 0 0 0 {expected:.17e} 0.0\n",
        encoding="utf-8",
        newline="\n",
    )
    return {"input": input_dir, "call": call_dir}


def _run(
    fixture: dict[str, Path], expected_normalization: str
) -> tuple[int, dict]:
    output = fixture["call"].parent.parent / "result.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(MODULE),
            "--input-dir",
            str(fixture["input"]),
            "--dump-call",
            str(fixture["call"]),
            "--expected-normalization",
            expected_normalization,
            "--coulomb-prefix",
            "coulomb_cut_",
            "--output",
            str(output),
        ],
        capture_output=True,
        text=True,
        cwd=fixture["input"].parent,
    )
    return completed.returncode, json.loads(output.read_text(encoding="utf-8"))


class HartreeContractionObserverTests(unittest.TestCase):
    def fixture_root(self):
        fixed_root = os.environ.get("LIBRPA_QSGW_HARTREE_CONTRACTION_TEST_TMP")
        if fixed_root:
            root = Path(fixed_root)
            if not root.is_dir():
                raise RuntimeError(
                    "LIBRPA_QSGW_HARTREE_CONTRACTION_TEST_TMP must already exist"
                )
            return _FixedDirectory(root)
        return tempfile.TemporaryDirectory(ignore_cleanup_errors=True)

    def test_weighted_contraction_passes(self) -> None:
        with self.fixture_root() as temporary:
            fixture = _write_fixture(Path(temporary) / "weighted-pass")
            returncode, report = _run(fixture, "weighted_occupations")
        self.assertEqual(returncode, 0, report)
        self.assertTrue(report["passed"])
        self.assertEqual(report["comparison"]["max_abs_ha"], 0.0)

    def test_legacy_normalization_passes_when_requested(self) -> None:
        with self.fixture_root() as temporary:
            fixture = _write_fixture(
                Path(temporary) / "legacy-pass",
                normalization="legacy_extra_inverse_nk",
            )
            returncode, report = _run(fixture, "legacy_extra_inverse_nk")
        self.assertEqual(returncode, 0, report)
        self.assertTrue(report["passed"])

    def test_hartree_perturbation_fails(self) -> None:
        with self.fixture_root() as temporary:
            fixture = _write_fixture(
                Path(temporary) / "perturbed", hartree_perturbation=1.0e-4
            )
            returncode, report = _run(fixture, "weighted_occupations")
        self.assertEqual(returncode, 1)
        self.assertFalse(report["passed"])
        self.assertGreater(report["comparison"]["max_abs_ha"], 1.0e-8)

    def test_normalization_contract_mismatch_fails(self) -> None:
        with self.fixture_root() as temporary:
            fixture = _write_fixture(Path(temporary) / "wrong-normalization")
            returncode, report = _run(fixture, "legacy_extra_inverse_nk")
        self.assertEqual(returncode, 1)
        self.assertFalse(report["passed"])
        self.assertIn("differs from expected", report["error"])

    def test_duplicate_periodic_gamma_fails(self) -> None:
        with self.fixture_root() as temporary:
            fixture = _write_fixture(
                Path(temporary) / "duplicate-gamma", duplicate_gamma=True
            )
            returncode, report = _run(fixture, "weighted_occupations")
        self.assertEqual(returncode, 1)
        self.assertFalse(report["passed"])
        self.assertIn("2 periodic Gamma", report["error"])


if __name__ == "__main__":
    unittest.main()
