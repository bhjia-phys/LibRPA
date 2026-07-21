#!/usr/bin/env python3
"""End-to-end synthetic test for the ABACUS QSGW band input pipeline."""

from __future__ import annotations

import contextlib
import json
import os
import tempfile
import unittest
from pathlib import Path

import prepare_abacus_qsgw_band_contract_v2 as band_contract
import preprocess_abacus_band_for_librpa_v2 as preprocess
import validate_abacus_si_band_nscf_output_v1 as validator


def wfc_text(k_index: int) -> str:
    return "\n".join(
        [
            f"{k_index} (index of k points)",
            "0.0 0.0 0.0",
            "2 (number of bands)",
            "3 (number of orbitals)",
            "1 (band)",
            "-1.0 (Ry)",
            "2.0 (Occupations)",
            "1.0 0.0 0.0 1.0 -0.5 0.25",
            "2 (band)",
            "0.5 (Ry)",
            "0.0 (Occupations)",
            "0.1 -0.2 0.3 -0.4 0.5 -0.6",
            "",
        ]
    )


def native_vxc_text(scale: int) -> str:
    return "\n".join(
        [
            "# rows 2",
            "# columns 2",
            "Row 1",
            f" ({2 * scale}.0,0.0) ({scale}.0,0.5)",
            "Row 2",
            f" ({4 * scale}.0,0.0)",
            "",
        ]
    )


class AbacusBandPipelineIntegrationTests(unittest.TestCase):
    def test_validator_preprocessor_and_contract_share_one_file_set(self) -> None:
        external_root = os.environ.get("LIBRPA_BAND_PIPELINE_TEST_ROOT")
        if external_root:
            external = Path(external_root).resolve(strict=True)
            if not external.is_dir():
                raise ValueError("external band-pipeline test root is not a directory")
            temporary_context = contextlib.nullcontext(external)
        else:
            temporary_context = tempfile.TemporaryDirectory(
                prefix="band-pipeline-v1-"
            )
        with temporary_context as temporary:
            root = Path(temporary)
            run = root / "run"
            output = run / "OUT.ABACUS"
            wfc_dir = output / "WFC"
            wfc_dir.mkdir(parents=True)
            (run / "INPUT").write_text(
                "\n".join(
                    [
                        "INPUT_PARAMETERS",
                        "calculation nscf",
                        "nbands 2",
                        "basis_type lcao",
                        "symmetry -1",
                        "init_chg file",
                        "out_app_flag 0",
                        "out_mat_xc 1",
                        "out_mat_xc2 1",
                        "out_wfc_lcao 1",
                        "",
                    ]
                ),
                encoding="ascii",
            )
            (run / "KPT").write_text("K_POINTS\n", encoding="ascii")
            (run / "STRU").write_text("ATOMIC_SPECIES\n", encoding="ascii")
            (output / "running_nscf.log").write_text(
                " NONSELF-CONSISTENT:\n Finish Time  : now\n", encoding="ascii"
            )
            (output / "INPUT.info").write_text("calculation nscf\n", encoding="ascii")
            (output / "KPT.info").write_text(
                "\n".join(
                    [
                        "nkstot now = 2",
                        "K-POINTS DIRECT COORDINATES",
                        "KPOINTS DIRECT_X DIRECT_Y DIRECT_Z WEIGHT",
                        "1 0.0 0.0 0.0 1.0",
                        "2 0.5 0.0 0.5 1.0",
                        "",
                    ]
                ),
                encoding="ascii",
            )
            charge = output / "ABACUS-CHARGE-DENSITY.restart"
            charge.write_bytes(b"frozen-charge")
            for index in (1, 2):
                (wfc_dir / f"wfk{index}_nao.txt").write_text(
                    wfc_text(index), encoding="ascii"
                )
                (output / f"vxck{index}_nao.txt").write_text(
                    native_vxc_text(index), encoding="ascii"
                )
            (output / "vxc_out.dat").write_text(
                "2\n1\n2\n-0.25 -6.8\n0.1 2.7\n-0.2 -5.4\n0.15 4.1\n",
                encoding="ascii",
            )

            report = validator.validate_output(
                run,
                validator.sha256_file(charge),
                expected_n_bands=2,
                expected_n_basis=3,
                expected_n_spins=1,
            )
            report_path = root / "NSCF_VALIDATION.json"
            report_path.write_text(
                json.dumps(report, sort_keys=True) + "\n", encoding="ascii"
            )

            dataset = root / "dataset"
            dataset.mkdir()
            frozen_files = {
                "band_out": b"scf eigenvalues",
                "KS_eigenvector_0.dat": b"scf wavefunctions",
                "bz_sampling_out": b"scf kpoints",
                "qsgw_vxc_scf.manifest": b"scf vxc manifest",
                "stru_out": b"reader static",
            }
            for name, content in frozen_files.items():
                (dataset / name).write_bytes(content)
            scf_records = [
                ("mf0_eigenvalues", "band_out"),
                ("mf0_wavefunctions", "KS_eigenvector_0.dat"),
                ("scf_kpoints", "bz_sampling_out"),
                ("vxc_scf_manifest", "qsgw_vxc_scf.manifest"),
                ("reader_static", "stru_out"),
            ]
            scf_contract = "\n".join(
                [
                    "# librpa-qsgw-input-contract-v1",
                    "producer abacus",
                    "internal_energy_units hartree",
                    "mf0_basis state_coefficients_in_nao",
                    "mf0_gauge producer_state",
                    "n_spins 1",
                    "n_bands 2",
                    "n_aos 3",
                    "n_scf_kpoints 1",
                    "n_headwing_kpoints 0",
                    "n_band_kpoints 0",
                    "headwing_grid disabled",
                    "headwing_update none",
                    "hartree_update off",
                    "band_update off",
                    "role sha256 file",
                    *[
                        f"{role} {validator.sha256_file(dataset / name)} {name}"
                        for role, name in scf_records
                    ],
                    "",
                ]
            )
            (dataset / "qsgw_input.contract").write_text(
                scf_contract, encoding="ascii"
            )

            preprocess.preprocess(
                output,
                dataset,
                report_path,
                "qsgw_vxc_band.manifest",
                "band_preprocess.summary.json",
            )
            summary = band_contract.prepare(
                dataset,
                "qsgw_input.contract",
                "band_kpath_info",
                "qsgw_vxc_band.manifest",
                "qsgw_band_input.contract",
                "qsgw_band_input_contract.summary.json",
            )

            self.assertEqual(summary["n_basis"], 3)
            self.assertEqual(summary["n_states"], 2)
            self.assertEqual(summary["n_band_kpoints"], 2)
            self.assertEqual(
                (dataset / "band_KS_eigenvector_k_00001.txt").stat().st_size,
                2 * 3 * 16,
            )
            self.assertEqual(
                (dataset / "band_vxck1_nao.txt").read_bytes(),
                (output / "vxck1_nao.txt").read_bytes(),
            )
            rendered = (dataset / "qsgw_band_input.contract").read_text(
                encoding="ascii"
            )
            self.assertIn("band_update operator_fourier", rendered)
            self.assertIn("n_band_kpoints 2", rendered)


if __name__ == "__main__":
    unittest.main()
