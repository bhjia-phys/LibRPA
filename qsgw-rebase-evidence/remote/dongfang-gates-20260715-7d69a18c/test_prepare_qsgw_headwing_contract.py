#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import importlib.util
import os
import tempfile
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent
MODULE_PATH = Path(
    os.environ.get(
        "QSGW_HEADWING_PREPARER",
        str(HERE / "prepare_qsgw_headwing_contract.py"),
    )
).resolve()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class PrepareQsgwHeadwingContractTest(unittest.TestCase):
    def load_module(self):
        self.assertTrue(
            MODULE_PATH.exists(),
            "prepare_qsgw_headwing_contract.py must implement the contract transition",
        )
        spec = importlib.util.spec_from_file_location(
            "prepare_qsgw_headwing_contract", MODULE_PATH
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def make_dataset(self, root: Path) -> tuple[Path, Path]:
        dataset = root / "dataset"
        dataset.mkdir()
        for name, payload in (
            ("band_out", b"bands\n"),
            ("KS_eigenvector_0.dat", b"wfc\n"),
            ("bz_sampling_out", b"kpoints\n"),
            ("qsgw_vxc_scf.manifest", b"vxc\n"),
            ("stru_out", b"structure\n"),
            ("velocity_matrix", b"velocity\n"),
        ):
            (dataset / name).write_bytes(payload)
        contract = dataset / "qsgw_input.contract"
        contract.write_text(
            "# librpa-qsgw-input-contract-v1\n"
            "producer abacus\n"
            "internal_energy_units hartree\n"
            "mf0_basis state_coefficients_in_nao\n"
            "mf0_gauge producer_state\n"
            "n_spins 1\n"
            "n_bands 2\n"
            "n_aos 2\n"
            "n_scf_kpoints 1\n"
            "n_headwing_kpoints 0\n"
            "n_band_kpoints 0\n"
            "headwing_grid disabled\n"
            "headwing_update none\n"
            "hartree_update off\n"
            "band_update off\n"
            "role sha256 file\n"
            f"mf0_eigenvalues {sha256(dataset / 'band_out')} band_out\n"
            f"mf0_wavefunctions {sha256(dataset / 'KS_eigenvector_0.dat')} KS_eigenvector_0.dat\n"
            f"scf_kpoints {sha256(dataset / 'bz_sampling_out')} bz_sampling_out\n"
            f"vxc_scf_manifest {sha256(dataset / 'qsgw_vxc_scf.manifest')} qsgw_vxc_scf.manifest\n"
            f"reader_static {sha256(dataset / 'stru_out')} stru_out\n",
            encoding="ascii",
        )
        return dataset, contract

    def test_disabled_contract_becomes_same_grid_live_velocity_contract(self):
        module = self.load_module()
        with tempfile.TemporaryDirectory() as temporary:
            dataset, contract = self.make_dataset(Path(temporary))
            original_sha = sha256(contract)
            report = module.activate_same_grid_headwing(
                dataset,
                expected_source_contract_sha256=original_sha,
                velocity_filename="velocity_matrix",
            )
            text = contract.read_text(encoding="ascii")
            self.assertIn("n_headwing_kpoints 1\n", text)
            self.assertIn("headwing_grid scf\n", text)
            self.assertIn("headwing_update fixed_basis_rotation\n", text)
            self.assertIn(
                f"velocity_mf0 {sha256(dataset / 'velocity_matrix')} velocity_matrix\n",
                text,
            )
            self.assertEqual(report["source_contract_sha256"], original_sha)
            self.assertEqual(report["new_contract_sha256"], sha256(contract))
            self.assertEqual(
                report["changed_metadata"],
                {
                    "headwing_grid": ["disabled", "scf"],
                    "headwing_update": ["none", "fixed_basis_rotation"],
                    "n_headwing_kpoints": ["0", "1"],
                },
            )
            self.assertEqual(report["added_roles"], ["velocity_mf0"])

    def test_source_payload_hash_mismatch_is_rejected(self):
        module = self.load_module()
        with tempfile.TemporaryDirectory() as temporary:
            dataset, contract = self.make_dataset(Path(temporary))
            source_sha = sha256(contract)
            (dataset / "band_out").write_text("changed\n", encoding="ascii")
            with self.assertRaisesRegex(ValueError, "payload SHA256 mismatch"):
                module.activate_same_grid_headwing(
                    dataset,
                    expected_source_contract_sha256=source_sha,
                    velocity_filename="velocity_matrix",
                )

    def test_abacus_pyatb_reader_set_is_declared_exactly(self):
        module = self.load_module()
        with tempfile.TemporaryDirectory() as temporary:
            dataset, contract = self.make_dataset(Path(temporary))
            pyatb = dataset / "pyatb_librpa_df"
            pyatb.mkdir()
            for name, payload in (
                ("k_path_info", b"path\n"),
                ("band_out", b"head bands\n"),
                ("KS_eigenvector_0.dat", b"head wfc\n"),
                ("velocity_matrix", b"head velocity\n"),
            ):
                (pyatb / name).write_bytes(payload)
            report = module.activate_same_grid_headwing(
                dataset,
                expected_source_contract_sha256=sha256(contract),
            )
            expected = [
                "pyatb_librpa_df/k_path_info",
                "pyatb_librpa_df/band_out",
                "pyatb_librpa_df/KS_eigenvector_0.dat",
                "pyatb_librpa_df/velocity_matrix",
            ]
            self.assertEqual(report["velocity_files"], expected)
            records = [
                line.split()[2]
                for line in contract.read_text(encoding="ascii").splitlines()
                if line.startswith("velocity_mf0 ")
            ]
            self.assertEqual(records, expected)
            self.assertNotIn("velocity_matrix", records)

    def test_already_enabled_contract_is_rejected(self):
        module = self.load_module()
        with tempfile.TemporaryDirectory() as temporary:
            dataset, contract = self.make_dataset(Path(temporary))
            source_sha = sha256(contract)
            module.activate_same_grid_headwing(
                dataset,
                expected_source_contract_sha256=source_sha,
                velocity_filename="velocity_matrix",
            )
            with self.assertRaisesRegex(ValueError, "disabled source contract"):
                module.activate_same_grid_headwing(
                    dataset,
                    expected_source_contract_sha256=sha256(contract),
                    velocity_filename="velocity_matrix",
                )


if __name__ == "__main__":
    unittest.main()
