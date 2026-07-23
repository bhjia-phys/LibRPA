#!/usr/bin/env python3
"""Tests for prepare_qsgw_band_contract_v1.

The generator turns a disabled (head/hartree/band off) QSGW input
contract plus band assets into a ``task=qsgw_band`` contract and a v2
band Vxc manifest, without touching any existing file. These tests use a
synthetic two-k-point dataset and lock: the exact append-only diff
against the baseline, the v2 manifest layout accepted by
``src/qsgw/vxc_io.cpp``, fail-closed validation, and SHA256 bindings.
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from prepare_qsgw_band_contract_v1 import (  # noqa: E402
    ContractGenerationError,
    generate_report,
    main,
)


N_BANDS = 26
N_BASIS = 30
EIGENVECTOR_BYTES = N_BANDS * N_BASIS * 16
KPOINTS = [("0", "0", "0"), ("0.025", "0.1", "-0.3333333333333333")]
BAND_CONTRACT_NAME = "qsgw_input.band.contract"
BAND_MANIFEST_NAME = "qsgw_vxc_band.v2.manifest"


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _baseline_text(extra_roles: tuple = ()) -> str:
    lines = [
        "# librpa-qsgw-input-contract-v1",
        "producer abacus",
        "internal_energy_units hartree",
        "mf0_basis state_coefficients_in_nao",
        "mf0_gauge producer_state",
        "n_spins 1",
        f"n_bands {N_BANDS}",
        f"n_aos {N_BASIS}",
        "n_scf_kpoints 64",
        "n_headwing_kpoints 0",
        "n_band_kpoints 0",
        "headwing_grid disabled",
        "headwing_update none",
        "hartree_update off",
        "band_update off",
        "role sha256 file",
        f"mf0_eigenvalues {'1' * 64} band_out",
        f"mf0_wavefunctions {'2' * 64} KS_eigenvector_0.dat",
    ]
    lines.extend(extra_roles)
    return "\n".join(lines) + "\n"


def _build_dataset(
    root: Path,
    *,
    extra_roles: tuple = (),
    eigenvector_size: int = EIGENVECTOR_BYTES,
    skip_vxc: frozenset = frozenset(),
) -> None:
    (root / "vxc_band").mkdir(parents=True, exist_ok=True)
    (root / "qsgw_input.disabled.contract").write_text(
        _baseline_text(extra_roles), encoding="utf-8", newline="\n"
    )
    kpath_lines = [f"{N_BANDS} {N_BASIS} 1 {len(KPOINTS)}"] + [
        " ".join(kpoint) for kpoint in KPOINTS
    ]
    (root / "band_kpath_info").write_text(
        "\n".join(kpath_lines) + "\n", encoding="utf-8", newline="\n"
    )
    for kpoint in (1, 2):
        (root / f"band_KS_eigenvalue_k_{kpoint:05d}.txt").write_text(
            f"# eigenvalues k={kpoint}\n0.1 0.2 0.3\n",
            encoding="utf-8",
            newline="\n",
        )
        (root / f"band_KS_eigenvector_k_{kpoint:05d}.txt").write_bytes(
            bytes((index * 7 + kpoint) % 256 for index in range(eigenvector_size))
        )
        if kpoint not in skip_vxc:
            (root / "vxc_band" / f"vxck{kpoint}s1_nao.txt").write_text(
                f"# vxc k={kpoint}\n1.0 0.0\n",
                encoding="utf-8",
                newline="\n",
            )


def _run_cli(*args: str) -> tuple[int, dict | None]:
    stdout = io.StringIO()
    with contextlib.redirect_stdout(stdout):
        exit_code = main(list(args))
    printed = stdout.getvalue().strip()
    return exit_code, json.loads(printed) if printed else None


class GenerateHappyPathTests(unittest.TestCase):
    def test_happy_path_outputs_and_diff(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _build_dataset(root)
            baseline_bytes = (
                root / "qsgw_input.disabled.contract"
            ).read_bytes()

            report = generate_report(root)

            self.assertEqual(report["n_band_kpoints"], 2)
            self.assertEqual(report["band_roles_added"], 6)
            self.assertEqual(
                report["metadata_changes"],
                [
                    {"key": "n_band_kpoints", "old": "0", "new": "2"},
                    {
                        "key": "band_update",
                        "old": "off",
                        "new": "fixed_basis_rotation",
                    },
                ],
            )
            self.assertEqual(
                report["diff_summary"],
                {"metadata_lines_changed": 2, "role_lines_appended": 6},
            )
            self.assertFalse(report["dry_run"])

            # Baseline untouched.
            self.assertEqual(
                (root / "qsgw_input.disabled.contract").read_bytes(),
                baseline_bytes,
            )

            contract_path = root / BAND_CONTRACT_NAME
            manifest_path = root / BAND_MANIFEST_NAME
            self.assertEqual(Path(report["band_contract"]), contract_path)
            self.assertEqual(Path(report["vxc_manifest"]), manifest_path)

            old_lines = _baseline_text().splitlines()
            new_lines = contract_path.read_text(
                encoding="utf-8"
            ).splitlines()
            self.assertEqual(len(new_lines), len(old_lines) + 6)
            changed = [
                index
                for index, (old, new) in enumerate(
                    zip(old_lines, new_lines[: len(old_lines)])
                )
                if old != new
            ]
            self.assertEqual(len(changed), 2)
            self.assertEqual(new_lines[changed[0]], "n_band_kpoints 2")
            self.assertEqual(
                new_lines[changed[1]],
                "band_update fixed_basis_rotation",
            )

            appended = new_lines[len(old_lines):]
            roles = [line.split(None, 1)[0] for line in appended]
            self.assertEqual(
                roles,
                [
                    "band_kpoints",
                    "band_mf0_eigenvalues",
                    "band_mf0_eigenvalues",
                    "band_mf0_wavefunctions",
                    "band_mf0_wavefunctions",
                    "vxc_band_manifest",
                ],
            )
            files = [line.split()[2] for line in appended]
            self.assertEqual(
                files,
                [
                    "band_kpath_info",
                    "band_KS_eigenvalue_k_00001.txt",
                    "band_KS_eigenvalue_k_00002.txt",
                    "band_KS_eigenvector_k_00001.txt",
                    "band_KS_eigenvector_k_00002.txt",
                    BAND_MANIFEST_NAME,
                ],
            )

            manifest_lines = manifest_path.read_text(
                encoding="utf-8"
            ).splitlines()
            self.assertEqual(
                manifest_lines[0], "# librpa-qsgw-vxc-manifest-v2"
            )
            metadata = {}
            for line in manifest_lines[1:6]:
                key, value = line.split()
                metadata[key] = value
            self.assertEqual(
                metadata,
                {
                    "kind": "band",
                    "producer": "abacus",
                    "units": "Ry",
                    "basis": "state",
                    "gauge": "mf0_state",
                },
            )
            self.assertEqual(
                manifest_lines[6],
                "spin k_index kx ky kz rows columns sha256 file",
            )
            entries = manifest_lines[7:]
            self.assertEqual(len(entries), 2)
            first = entries[0].split()
            self.assertEqual(first[:2], ["1", "1"])
            self.assertEqual(first[2:5], ["0", "0", "0"])
            self.assertEqual(first[5:7], ["26", "26"])
            self.assertEqual(first[8], "vxc_band/vxck1s1_nao.txt")
            second = entries[1].split()
            self.assertEqual(second[:2], ["1", "2"])
            # Coordinate text precision preserved verbatim.
            self.assertEqual(second[2:5], list(KPOINTS[1]))
            self.assertEqual(second[8], "vxc_band/vxck2s1_nao.txt")

    def test_dry_run_writes_nothing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _build_dataset(root)
            report = generate_report(root, dry_run=True)
            self.assertTrue(report["dry_run"])
            self.assertEqual(report["n_band_kpoints"], 2)
            self.assertEqual(report["band_roles_added"], 6)
            self.assertFalse((root / BAND_CONTRACT_NAME).exists())
            self.assertFalse((root / BAND_MANIFEST_NAME).exists())

    def test_cli_happy_path_exit_zero(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _build_dataset(root)
            exit_code, report = _run_cli("--dataset", str(root))
            self.assertEqual(exit_code, 0)
            self.assertEqual(report["n_band_kpoints"], 2)


class FailureTests(unittest.TestCase):
    def test_missing_vxc_file_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _build_dataset(root, skip_vxc=frozenset({2}))
            with self.assertRaisesRegex(ContractGenerationError, "vxc"):
                generate_report(root)
            exit_code, report = _run_cli("--dataset", str(root))
            self.assertEqual(exit_code, 1)
            self.assertIn("error", report)

    def test_eigenvector_wrong_size_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _build_dataset(root, eigenvector_size=1000)
            with self.assertRaisesRegex(
                ContractGenerationError, "eigenvector.*size"
            ):
                generate_report(root)
            exit_code, _report = _run_cli("--dataset", str(root))
            self.assertEqual(exit_code, 1)

    def test_duplicate_role_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _build_dataset(
                root,
                extra_roles=(f"band_kpoints {'3' * 64} band_kpath_info",),
            )
            with self.assertRaisesRegex(ContractGenerationError, "duplicate"):
                generate_report(root)
            exit_code, _report = _run_cli("--dataset", str(root))
            self.assertEqual(exit_code, 1)


class IntegrityTests(unittest.TestCase):
    def test_baseline_not_modified(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _build_dataset(root)
            baseline = root / "qsgw_input.disabled.contract"
            before_content = baseline.read_bytes()
            before_mtime_ns = baseline.stat().st_mtime_ns
            generate_report(root)
            self.assertEqual(baseline.read_bytes(), before_content)
            self.assertEqual(baseline.stat().st_mtime_ns, before_mtime_ns)

    def test_sha256_correctness(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _build_dataset(root)
            report = generate_report(root)

            contract_bytes = (root / BAND_CONTRACT_NAME).read_bytes()
            manifest_bytes = (root / BAND_MANIFEST_NAME).read_bytes()
            self.assertEqual(
                report["sha256"]["band_contract"], _sha256_bytes(contract_bytes)
            )
            self.assertEqual(
                report["sha256"]["vxc_manifest"],
                _sha256_bytes(manifest_bytes),
            )

            lines = contract_bytes.decode("utf-8").splitlines()
            role_records = {}
            for line in lines:
                fields = line.split()
                if len(fields) == 3 and fields[0] != "role":
                    role_records[(fields[0], fields[2])] = fields[1]

            for kpoint in (1, 2):
                for role, name in (
                    (
                        "band_mf0_eigenvalues",
                        f"band_KS_eigenvalue_k_{kpoint:05d}.txt",
                    ),
                    (
                        "band_mf0_wavefunctions",
                        f"band_KS_eigenvector_k_{kpoint:05d}.txt",
                    ),
                ):
                    expected = _sha256_bytes((root / name).read_bytes())
                    self.assertEqual(role_records[(role, name)], expected)
            self.assertEqual(
                role_records[("band_kpoints", "band_kpath_info")],
                _sha256_bytes((root / "band_kpath_info").read_bytes()),
            )
            self.assertEqual(
                role_records[("vxc_band_manifest", BAND_MANIFEST_NAME)],
                _sha256_bytes(manifest_bytes),
            )

            manifest_entries = (
                manifest_bytes.decode("utf-8").splitlines()[7:]
            )
            for kpoint, entry in zip((1, 2), manifest_entries):
                fields = entry.split()
                expected = _sha256_bytes(
                    (root / "vxc_band" / f"vxck{kpoint}s1_nao.txt").read_bytes()
                )
                self.assertEqual(fields[7], expected)


if __name__ == "__main__":
    unittest.main()
