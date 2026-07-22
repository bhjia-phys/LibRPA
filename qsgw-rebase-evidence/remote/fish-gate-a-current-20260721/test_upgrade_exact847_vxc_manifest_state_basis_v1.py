#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import shutil
import unittest
import uuid
from pathlib import Path

from upgrade_exact847_vxc_manifest_state_basis_v1 import run_upgrade


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class UpgradeExact847VxcManifestTest(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path.cwd() / "__test_tmp" / uuid.uuid4().hex
        self.source = self.root / "source"
        self.output = self.root / "output"
        self.source.mkdir(parents=True)
        self.output.mkdir()
        self.matrix1 = self.source / "vxcs1k1_nao.txt"
        self.matrix2 = self.source / "vxcs1k2_nao.txt"
        self.static = self.source / "band_out"
        self.matrix1.write_text("matrix one\n", encoding="ascii")
        self.matrix2.write_text("matrix two\n", encoding="ascii")
        self.static.write_text("1 2 3 4\n", encoding="ascii")
        self.manifest = self.source / "qsgw_vxc_scf.manifest"
        self.manifest.write_text(
            "\n".join(
                [
                    "# librpa-qsgw-vxc-manifest-v2",
                    "kind scf",
                    "producer abacus",
                    "units Ry",
                    "basis nao",
                    "gauge ao_bloch",
                    "spin k_index kx ky kz rows columns sha256 file",
                    f"1 1 0 0 0 2 2 {sha256_file(self.matrix1)} {self.matrix1.name}",
                    f"1 2 0.5 0 0 2 2 {sha256_file(self.matrix2)} {self.matrix2.name}",
                ]
            )
            + "\n",
            encoding="ascii",
        )
        self.contract = self.source / "qsgw_input.contract"
        self.contract.write_text(
            "\n".join(
                [
                    "# librpa-qsgw-input-contract-v1",
                    "producer abacus",
                    "role sha256 file",
                    f"mf0_eigenvalues {sha256_file(self.static)} {self.static.name}",
                    f"vxc_scf_manifest {sha256_file(self.manifest)} {self.manifest.name}",
                ]
            )
            + "\n",
            encoding="ascii",
        )

    def tearDown(self) -> None:
        shutil.rmtree(self.root)

    def run_valid_upgrade(self) -> tuple[Path, Path, Path, dict[str, object]]:
        output_contract = self.output / self.contract.name
        output_manifest = self.output / self.manifest.name
        report_path = self.output / "report.json"
        report = run_upgrade(
            self.contract,
            self.manifest,
            output_contract,
            output_manifest,
            report_path,
        )
        return output_contract, output_manifest, report_path, report

    def test_corrects_only_basis_metadata_and_contract_hash(self) -> None:
        output_contract, output_manifest, report_path, report = self.run_valid_upgrade()

        expected_manifest = self.manifest.read_text(encoding="ascii").replace(
            "basis nao\ngauge ao_bloch", "basis state\ngauge mf0_state"
        )
        self.assertEqual(output_manifest.read_text(encoding="ascii"), expected_manifest)
        self.assertIn(
            f"vxc_scf_manifest {sha256_file(output_manifest)} {output_manifest.name}",
            output_contract.read_text(encoding="ascii"),
        )
        self.assertEqual(report["manifest"]["matrix_count"], 2)
        self.assertEqual(report["contract"]["verified_role_count"], 2)
        self.assertTrue(report["passed"])
        self.assertTrue(report_path.is_file())

    def test_rejects_matrix_hash_mismatch(self) -> None:
        self.matrix1.write_text("tampered\n", encoding="ascii")
        with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
            self.run_valid_upgrade()

    def test_rejects_nonlegacy_native_filename(self) -> None:
        text = self.manifest.read_text(encoding="ascii").replace(
            self.matrix1.name, "vxck1_nao.txt"
        )
        self.manifest.write_text(text, encoding="ascii")
        with self.assertRaisesRegex(ValueError, "expected exact847"):
            self.run_valid_upgrade()

    def test_rejects_already_state_basis_manifest(self) -> None:
        text = self.manifest.read_text(encoding="ascii").replace(
            "basis nao\ngauge ao_bloch", "basis state\ngauge mf0_state"
        )
        self.manifest.write_text(text, encoding="ascii")
        with self.assertRaisesRegex(ValueError, "expected 'basis nao'"):
            self.run_valid_upgrade()

    def test_rejects_contract_not_bound_to_manifest(self) -> None:
        text = self.contract.read_text(encoding="ascii").replace(
            sha256_file(self.manifest), "0" * 64
        )
        self.contract.write_text(text, encoding="ascii")
        with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
            self.run_valid_upgrade()


if __name__ == "__main__":
    unittest.main()
