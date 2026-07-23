#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_legacy_band_vxc_view_v1 import (  # noqa: E402
    LegacyBandVxcError,
    build_legacy_band_vxc_view,
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def native_matrix(values: list[list[complex]]) -> str:
    lines = [
        "# filename OUT.ABACUS/vxck1_nao.txt",
        f"# rows {len(values)}",
        f"# columns {len(values)}",
    ]
    for row, row_values in enumerate(values, start=1):
        lines.append(f"Row {row}")
        lines.append(
            " ".join(
                f"({value.real:.16e},{value.imag:.16e})"
                for value in row_values[row - 1 :]
            )
        )
    return "\n".join(lines) + "\n"


class LegacyBandVxcViewTest(unittest.TestCase):
    def build_fixture(
        self, root: Path, *, basis: str = "state", digest: str | None = None
    ) -> tuple[Path, Path]:
        source = root / "dataset"
        output = root / "overlay"
        (source / "vxc_band").mkdir(parents=True)
        output.mkdir()
        matrix = source / "vxc_band" / "vxck1s1_nao.txt"
        matrix.write_text(
            native_matrix(
                [
                    [complex(-0.8, 0.0), complex(0.1, -0.2)],
                    [complex(0.1, 0.2), complex(-0.4, 0.0)],
                ]
            ),
            encoding="ascii",
        )
        manifest = source / "qsgw_vxc_band.v2.manifest"
        manifest.write_text(
            "\n".join(
                [
                    "# librpa-qsgw-vxc-manifest-v2",
                    "kind band",
                    "producer abacus",
                    "units Ry",
                    f"basis {basis}",
                    "gauge mf0_state",
                    "spin k_index kx ky kz rows columns sha256 file",
                    "1 1 0.0 0.0 0.0 2 2 "
                    f"{digest or sha256(matrix)} vxc_band/vxck1s1_nao.txt",
                ]
            )
            + "\n",
            encoding="ascii",
        )
        return manifest, output

    def test_builds_exact_legacy_upper_triangle_without_unit_conversion(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, output = self.build_fixture(root)
            report = build_legacy_band_vxc_view(
                manifest.parent, manifest, output
            )

            legacy = output / "band_vxcs1k1_nao.txt"
            self.assertEqual(
                legacy.read_text(encoding="ascii"),
                "2\n"
                "(-8.0000000000000004e-01,0.0000000000000000e+00) "
                "(1.0000000000000001e-01,-2.0000000000000001e-01)\n"
                "(-4.0000000000000002e-01,0.0000000000000000e+00)\n",
            )
            self.assertEqual(report["status"], "PASS")
            self.assertEqual(report["files_generated"], 1)
            self.assertEqual(report["source_basis"], "state")
            self.assertEqual(report["source_gauge"], "mf0_state")
            self.assertEqual(report["energy_conversion"], "legacy_reader_Ry_to_Ha")
            self.assertTrue(report["numeric_roundtrip_equal"])

    def test_rejects_nao_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, output = self.build_fixture(root, basis="nao")
            with self.assertRaisesRegex(
                LegacyBandVxcError, "state.*mf0_state"
            ):
                build_legacy_band_vxc_view(manifest.parent, manifest, output)

    def test_rejects_source_hash_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, output = self.build_fixture(root, digest="0" * 64)
            with self.assertRaisesRegex(LegacyBandVxcError, "hash mismatch"):
                build_legacy_band_vxc_view(manifest.parent, manifest, output)

    def test_refuses_to_overwrite_legacy_view(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, output = self.build_fixture(root)
            (output / "band_vxcs1k1_nao.txt").write_text(
                "occupied\n", encoding="ascii"
            )
            with self.assertRaisesRegex(LegacyBandVxcError, "overwrite"):
                build_legacy_band_vxc_view(manifest.parent, manifest, output)


if __name__ == "__main__":
    unittest.main()
