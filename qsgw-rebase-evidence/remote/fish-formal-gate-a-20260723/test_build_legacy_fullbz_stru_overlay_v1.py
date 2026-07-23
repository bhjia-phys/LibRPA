#!/usr/bin/env python3
"""Tests for the legacy full-BZ ``stru_out`` compatibility overlay."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from build_legacy_fullbz_stru_overlay_v1 import OverlayError, build_overlay


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(65536), b""):
            digest.update(block)
    return digest.hexdigest()


class LegacyFullBzStruOverlayTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory(dir=Path(__file__).parent)
        self.root = Path(self.tmp.name)
        self.source = self.root / "source" / "dataset"
        self.source.mkdir(parents=True)
        self.output = self.root / "overlay" / "dataset"

        self.stru_text = """\
1 0 0
0 1 0
0 0 1
6.283185307179586 0 0
0 6.283185307179586 0
0 0 6.283185307179586
1
0 0 0 1
"""
        (self.source / "stru_out").write_text(self.stru_text, encoding="ascii")
        (self.source / "band_out").write_text(
            "2\n1\n1\n1\n0.0\n", encoding="ascii"
        )
        (self.source / "bz_sampling_out").write_text(
            """\
2 1 1
2 2
1 0.5 0.0 0.0 0.0 0.0 0.0 0.0 1 1
2 0.5 0.5 0.0 0.0 3.141592653589793 0.0 0.0 2 2
""",
            encoding="ascii",
        )
        (self.source / "payload.bin").write_bytes(b"unchanged payload\n")
        self.manifest = self.root / "source" / "DATASET_SHA256SUMS.txt"
        self.write_manifest()

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def write_manifest(self) -> None:
        lines = []
        for path in sorted(self.source.rglob("*")):
            if path.is_file():
                rel = path.relative_to(self.source.parent).as_posix()
                lines.append(f"{sha256(path)}  {rel}\n")
        self.manifest.write_text("".join(lines), encoding="ascii")

    def test_builds_same_input_overlay_and_hardlinks_unchanged_files(self) -> None:
        report = build_overlay(self.source, self.manifest, self.output)

        expected = self.stru_text + """\
2 1 1
0.0 0.0 0.0
3.141592653589793 0.0 0.0
1
2
"""
        self.assertEqual((self.output / "stru_out").read_text(encoding="ascii"), expected)
        self.assertFalse((self.output / "stru_out").samefile(self.source / "stru_out"))
        self.assertTrue((self.output / "payload.bin").samefile(self.source / "payload.bin"))
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(report["n_kpoints"], 2)
        self.assertEqual(report["grid"], [2, 1, 1])
        self.assertEqual(report["unchanged_files_hardlinked"], 3)

        report_path = self.output.parent / "LEGACY_FULLBZ_STRU_OVERLAY.json"
        self.assertEqual(json.loads(report_path.read_text(encoding="ascii")), report)
        output_manifest = self.output.parent / "DATASET_SHA256SUMS.txt"
        self.assertIn(f"{sha256(self.output / 'stru_out')}  dataset/stru_out",
                      output_manifest.read_text(encoding="ascii"))
        self.assertEqual((self.source / "stru_out").read_text(encoding="ascii"),
                         self.stru_text)

    def test_rejects_structure_with_existing_tail(self) -> None:
        with (self.source / "stru_out").open("a", encoding="ascii") as stream:
            stream.write("2 1 1\n")
        self.write_manifest()

        with self.assertRaisesRegex(OverlayError, "already has trailing data"):
            build_overlay(self.source, self.manifest, self.output)

    def test_rejects_nonidentity_full_bz_mapping(self) -> None:
        text = (self.source / "bz_sampling_out").read_text(encoding="ascii")
        (self.source / "bz_sampling_out").write_text(
            text.replace("2 2\n", "1 2\n", 1), encoding="ascii"
        )
        self.write_manifest()

        with self.assertRaisesRegex(OverlayError, "identity mapping"):
            build_overlay(self.source, self.manifest, self.output)

    def test_rejects_band_and_grid_count_mismatch(self) -> None:
        (self.source / "band_out").write_text(
            "3\n1\n1\n1\n0.0\n", encoding="ascii"
        )
        self.write_manifest()

        with self.assertRaisesRegex(OverlayError, "band_out k-point count"):
            build_overlay(self.source, self.manifest, self.output)

    def test_rejects_manifest_hash_mismatch(self) -> None:
        (self.source / "payload.bin").write_bytes(b"mutated\n")

        with self.assertRaisesRegex(OverlayError, "manifest hash mismatch"):
            build_overlay(self.source, self.manifest, self.output)

    def test_rejects_existing_output_and_source_symlinks(self) -> None:
        self.output.mkdir(parents=True)
        with self.assertRaisesRegex(OverlayError, "already exists"):
            build_overlay(self.source, self.manifest, self.output)

        self.output.rmdir()
        link = self.source / "payload-link"
        try:
            link.symlink_to(self.source / "payload.bin")
        except OSError as error:
            self.skipTest(f"symlinks unavailable: {error}")
        self.write_manifest()
        with self.assertRaisesRegex(OverlayError, "symlink"):
            build_overlay(self.source, self.manifest, self.output)


if __name__ == "__main__":
    unittest.main()
