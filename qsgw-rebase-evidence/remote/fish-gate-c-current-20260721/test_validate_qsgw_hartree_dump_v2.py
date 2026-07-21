#!/usr/bin/env python3
"""Synthetic regression tests for the order-explicit Hartree dump observer."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


MODULE = Path(__file__).resolve().parent / "validate_qsgw_hartree_dump_v2.py"
CONTRACT = (
    "# qsgw_contract_version 6\n"
    "# fixed_basis immutable_mf0\n"
    "# live_update eigenvalues_wfc\n"
    "# velocity disabled_stage1\n"
    "# headwing disabled_stage1\n"
    "# symmetry exx_on_gw_on_rpa_on\n"
    "# hartree delta_density\n"
    "# hartree_coulomb full\n"
    "# hartree_normalization weighted_occupations\n"
    "# band disabled_stage1\n"
    "# h_qsgw_cut disabled_non_band\n"
    "# qsgw_input_contract qsgw_input.hartree-full.contract\n"
    f"# qsgw_input_contract_sha256 {'0' * 64}\n"
    "# qsgw_mixer none\n"
    "# qsgw_mixing_beta 0.2\n"
)


def _matrix_row(iteration: int, component: str, value: complex, frequency=-1) -> str:
    frequency_ha = 0.0 if frequency == -1 else 0.1
    return (
        f"{iteration} 0 {component} 0 0 {frequency} {frequency_ha:.17e} "
        f"0 0 {value.real:.17e} {value.imag:.17e}"
    )


def _write_trace(root: Path, delta_iteration2: float = 2.0) -> tuple[Path, Path]:
    matrix = root / "qsgw_matrices.dat"
    rows = [
        _matrix_row(0, "h0", 0.0j),
        _matrix_row(0, "vxc_dft", 0.0j),
        _matrix_row(0, "occupation", 1.0 + 0.0j),
        _matrix_row(0, "wfc_spinor0", 1.0 + 0.0j),
    ]
    for iteration, delta in ((1, 0.0), (2, delta_iteration2)):
        rows.extend(
            [
                _matrix_row(iteration, "sigma_c_iw", 0.0j, frequency=0),
                _matrix_row(iteration, "exx", 0.0j),
                _matrix_row(iteration, "vc", 0.0j),
                _matrix_row(iteration, "delta_vh", delta + 0.0j),
                _matrix_row(iteration, "raw_h", delta + 0.0j),
                _matrix_row(iteration, "mixed_h", delta + 0.0j),
                _matrix_row(iteration, "rotation_u", 1.0 + 0.0j),
                _matrix_row(iteration, "occupation", 1.0 + 0.0j),
                _matrix_row(iteration, "wfc_spinor0", 1.0 + 0.0j),
            ]
        )
    matrix.write_text(
        CONTRACT
        + "# iter channel component spin kpoint frequency_index frequency_Ha "
        "row column real_value imag_value\n"
        + "\n".join(rows)
        + "\n",
        encoding="utf-8",
        newline="\n",
    )

    eigenvalue = root / "qsgw_eigenvalues.dat"
    eigenvalue.write_text(
        CONTRACT
        + "# iter channel spin kpoint kx ky kz band energy_eV\n"
        + "\n".join(
            f"{iteration} 0 0 0 0.0 0.0 0.0 0 {float(iteration):.17e}"
            for iteration in range(3)
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return matrix, eigenvalue


def _write_call(call_dir: Path, iteration: int) -> None:
    call_dir.mkdir(parents=True, exist_ok=True)
    call_dir.joinpath("manifest.txt").write_text(
        "schema=qsgw_hartree_pipeline_dump_v1\n"
        "normalization=weighted_occupations\n"
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
        "bvk_remap_source_count=1\n"
        "bvk_remap_file=bvk_remap.txt\n"
        "bvk_remap_columns=atom_i atom_j source_R_x source_R_y source_R_z "
        "target_index target_count target_R_x target_R_y target_R_z\n",
        encoding="utf-8",
        newline="\n",
    )
    call_dir.joinpath("full_kpoints.txt").write_text(
        "# index kx ky kz\n0 0 0 0\n1 0.5 0 0\n",
        encoding="utf-8",
        newline="\n",
    )
    call_dir.joinpath("translations.txt").write_text(
        "# index R_x R_y R_z\n0 0 0 0\n1 1 0 0\n",
        encoding="utf-8",
        newline="\n",
    )
    call_dir.joinpath("bvk_remap.txt").write_text(
        "# atom_i atom_j source_R_x source_R_y source_R_z target_index "
        "target_count target_R_x target_R_y target_R_z\n"
        "0 0 1 0 0 0 1 -1 0 0\n",
        encoding="utf-8",
        newline="\n",
    )
    if iteration == 1:
        density = (0.0, 0.0)
        hartree_k = (0.0, 0.0)
        hartree_r = (0.0, 0.0)
    else:
        density = (1.0, -1.0)
        hartree_k = (2.0, 0.0)
        hartree_r = (1.0, 1.0)
    call_dir.joinpath("density_delta_k.txt").write_text(
        "# kpoint row column real imag\n"
        f"0 0 0 {density[0]:.17e} 0\n"
        f"1 0 0 {density[1]:.17e} 0\n",
        encoding="utf-8",
        newline="\n",
    )
    call_dir.joinpath("hartree_k.txt").write_text(
        "# atom_i atom_j kpoint row column real imag\n"
        f"0 0 0 0 0 {hartree_k[0]:.17e} 0\n"
        f"0 0 1 0 0 {hartree_k[1]:.17e} 0\n",
        encoding="utf-8",
        newline="\n",
    )
    call_dir.joinpath("hartree_r.txt").write_text(
        "# atom_i atom_j R_x R_y R_z row column real imag\n"
        f"0 0 0 0 0 0 0 {hartree_r[0]:.17e} 0\n"
        f"0 0 -1 0 0 0 0 {hartree_r[1]:.17e} 0\n",
        encoding="utf-8",
        newline="\n",
    )


def _write_fixture(root: Path, delta_iteration2: float = 2.0) -> dict[str, Path]:
    dump = root / "dump"
    _write_call(dump / "call_001", 1)
    _write_call(dump / "call_002", 2)
    matrix, eigenvalue = _write_trace(root, delta_iteration2)
    return {
        "dump": dump,
        "matrix": matrix,
        "eigenvalue": eigenvalue,
        "report": root / "report.json",
    }


def _run(paths: dict[str, Path]) -> tuple[subprocess.CompletedProcess[str], dict]:
    result = subprocess.run(
        [
            sys.executable,
            "-B",
            str(MODULE),
            str(paths["dump"]),
            str(paths["matrix"]),
            str(paths["eigenvalue"]),
            str(paths["report"]),
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    report = json.loads(paths["report"].read_text(encoding="utf-8"))
    return result, report


class HartreeDumpObserverTests(unittest.TestCase):
    def fixture_root(self):
        fixed_root = os.environ.get("LIBRPA_QSGW_HARTREE_OBSERVER_TEST_TMP")
        if fixed_root:
            root = Path(fixed_root)
            if not root.is_dir():
                raise RuntimeError(
                    "LIBRPA_QSGW_HARTREE_OBSERVER_TEST_TMP must already exist"
                )
            return _FixedDirectory(root)
        return tempfile.TemporaryDirectory(ignore_cleanup_errors=True)

    def test_symmetry_reduced_trace_and_order_explicit_full_grid_pass(self) -> None:
        with self.fixture_root() as temporary:
            paths = _write_fixture(Path(temporary))
            result, report = _run(paths)
            self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
            self.assertTrue(report["passed"])
            self.assertEqual(report["calls"][1]["fixed_basis_projection"]["channels"], [0])
            self.assertEqual(report["calls"][1]["bvk_remap"]["source_count"], 1)

    def test_full_kpoint_order_is_used_in_fourier_transform(self) -> None:
        with self.fixture_root() as temporary:
            paths = _write_fixture(Path(temporary))
            path = paths["dump"] / "call_002" / "full_kpoints.txt"
            path.write_text(
                "# index kx ky kz\n0 0.5 0 0\n1 0 0 0\n",
                encoding="utf-8",
                newline="\n",
            )
            result, report = _run(paths)
            self.assertEqual(result.returncode, 2)
            self.assertFalse(report["calls"][1]["inverse_fourier"]["passed"])

    def test_bvk_remap_is_required_for_fourier_match(self) -> None:
        with self.fixture_root() as temporary:
            paths = _write_fixture(Path(temporary))
            path = paths["dump"] / "call_002" / "bvk_remap.txt"
            path.write_text(
                "# atom_i atom_j source_R_x source_R_y source_R_z target_index "
                "target_count target_R_x target_R_y target_R_z\n"
                "0 0 1 0 0 0 1 -2 0 0\n",
                encoding="utf-8",
                newline="\n",
            )
            result, report = _run(paths)
            self.assertEqual(result.returncode, 1)
            self.assertFalse(report["passed"])
            self.assertIn("block-key mismatch", report["error"])

    def test_projection_mismatch_fails_numerically(self) -> None:
        with self.fixture_root() as temporary:
            paths = _write_fixture(Path(temporary), delta_iteration2=2.1)
            result, report = _run(paths)
            self.assertEqual(result.returncode, 2)
            self.assertFalse(report["calls"][1]["fixed_basis_projection"]["passed"])

    def test_missing_order_metadata_fails_closed(self) -> None:
        with self.fixture_root() as temporary:
            paths = _write_fixture(Path(temporary))
            path = paths["dump"] / "call_001" / "manifest.txt"
            text = path.read_text(encoding="utf-8")
            path.write_text(
                text.replace("full_kpoints_file=full_kpoints.txt\n", ""),
                encoding="utf-8",
                newline="\n",
            )
            result, report = _run(paths)
            self.assertEqual(result.returncode, 1)
            self.assertFalse(report["passed"])
            self.assertIn("full_kpoints_file", report["error"])


class _FixedDirectory:
    def __init__(self, path: Path):
        self.path = path

    def __enter__(self) -> str:
        return str(self.path)

    def __exit__(self, _type, _value, _traceback) -> None:
        return None


if __name__ == "__main__":
    unittest.main()
