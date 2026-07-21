#!/usr/bin/env python3
"""Synthetic tests for the corrected-legacy null-delta side-effect guard."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


MODULE = Path(__file__).resolve().parent / "compare_legacy_hartree_null_delta_v1.py"


class _FixedDirectory:
    def __init__(self, path: Path):
        self.path = path

    def __enter__(self) -> str:
        return str(self.path)

    def __exit__(self, *_args) -> None:
        return None


def _header(hartree: bool, *, nfreq: int = 6) -> str:
    return """# qsgw_contract_version 4
# oracle_kind legacy_scheme_a
# oracle_source_commit e08f4a13
# task qsgw
# fixed_basis immutable_reference
# qsgw_mixer linear
# qsgw_mixing_beta 0.2
# qsgw_min_iter 1
# qsgw_max_iter 1
# starting_vxc dft_only
# vxc_basis fixed_state
# qsgw_update_hartree {hartree}
# qsgw_hartree_coulomb truncated
# qsgw_hartree_normalization legacy_extra_inverse_nk
# use_symmetry_gw 0
# use_symmetry_exx 0
# replace_w_head 0
# option_dielect_func 0
# nfreq {nfreq}
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
""".format(hartree=1 if hartree else 0, nfreq=nfreq)


def _row(iteration: int, component: str, value: float, frequency: int = -1) -> str:
    frequency_ha = 0.1 if frequency >= 0 else 0.0
    return (
        f"{iteration} 0 {component} 0 0 {frequency} {frequency_ha:.17e} "
        f"0 0 {value:.17e} 0.0"
    )


def _write_fixture(
    root: Path,
    *,
    delta: float = 0.0,
    exx_side_effect: float = 0.0,
    contract_difference: bool = False,
) -> dict[str, Path]:
    root.mkdir(parents=True, exist_ok=True)
    common = [
        _row(0, "h0", 1.0),
        _row(0, "vxc_dft", 0.2),
        _row(0, "occupation", 1.0),
        _row(0, "fermi_energy_ha", 0.1),
        _row(0, "electron_count", 2.0),
        _row(0, "gap_ha", 0.3),
        _row(0, "wfc_spinor0", 1.0),
        _row(1, "sigma_c_iw", 0.4, frequency=0),
        _row(1, "exx", 0.5),
        _row(1, "vc", 0.6),
        _row(1, "raw_h", 1.7),
        _row(1, "mixed_h", 1.14),
        _row(1, "rotation_u", 1.0),
        _row(1, "occupation", 1.0),
        _row(1, "fermi_energy_ha", 0.1),
        _row(1, "electron_count", 2.0),
        _row(1, "gap_ha", 0.3),
        _row(1, "wfc_spinor0", 1.0),
    ]
    disabled = root / "disabled.dat"
    enabled = root / "enabled.dat"
    disabled.write_text(
        _header(False) + "\n".join(common) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    enabled_rows = list(common)
    enabled_rows[8] = _row(1, "exx", 0.5 + exx_side_effect)
    enabled_rows.append(_row(1, "delta_vh", delta))
    enabled.write_text(
        _header(True, nfreq=7 if contract_difference else 6)
        + "\n".join(enabled_rows)
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    enabled_summary = root / "enabled-summary.dat"
    disabled_summary = root / "disabled-summary.dat"
    summary = "0 -1.0 1.0 0.0\n1 -0.9 1.1 0.1\n"
    enabled_summary.write_text(summary, encoding="utf-8", newline="\n")
    disabled_summary.write_text(summary, encoding="utf-8", newline="\n")
    return {
        "enabled": enabled,
        "disabled": disabled,
        "enabled_summary": enabled_summary,
        "disabled_summary": disabled_summary,
    }


def _run(fixture: dict[str, Path]) -> tuple[int, dict]:
    output = fixture["enabled"].parent / "result.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(MODULE),
            str(fixture["enabled"]),
            str(fixture["disabled"]),
            str(fixture["enabled_summary"]),
            str(fixture["disabled_summary"]),
            str(output),
        ],
        capture_output=True,
        text=True,
        cwd=fixture["enabled"].parent,
    )
    return completed.returncode, json.loads(output.read_text(encoding="utf-8"))


class LegacyHartreeNullDeltaTests(unittest.TestCase):
    def fixture_root(self):
        fixed = os.environ.get("LIBRPA_QSGW_LEGACY_HARTREE_NULL_TEST_TMP")
        if fixed:
            root = Path(fixed)
            if not root.is_dir():
                raise RuntimeError(
                    "LIBRPA_QSGW_LEGACY_HARTREE_NULL_TEST_TMP must already exist"
                )
            return _FixedDirectory(root)
        return tempfile.TemporaryDirectory(ignore_cleanup_errors=True)

    def test_zero_delta_and_identical_control_pass(self) -> None:
        with self.fixture_root() as temporary:
            fixture = _write_fixture(Path(temporary) / "pass")
            returncode, report = _run(fixture)
        self.assertEqual(returncode, 0, report)
        self.assertTrue(report["passed"])

    def test_nonzero_delta_fails(self) -> None:
        with self.fixture_root() as temporary:
            fixture = _write_fixture(Path(temporary) / "delta", delta=1.0e-4)
            returncode, report = _run(fixture)
        self.assertEqual(returncode, 1)
        self.assertFalse(report["delta_vh"]["passed"])

    def test_exx_reader_side_effect_fails(self) -> None:
        with self.fixture_root() as temporary:
            fixture = _write_fixture(
                Path(temporary) / "exx", exx_side_effect=1.0e-4
            )
            returncode, report = _run(fixture)
        self.assertEqual(returncode, 1)
        self.assertFalse(report["components"]["exx"]["passed"])

    def test_non_hartree_contract_difference_fails(self) -> None:
        with self.fixture_root() as temporary:
            fixture = _write_fixture(
                Path(temporary) / "contract", contract_difference=True
            )
            returncode, report = _run(fixture)
        self.assertEqual(returncode, 1)
        self.assertIn("outside Hartree enablement", report["error"])


if __name__ == "__main__":
    unittest.main()
