#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import importlib.util
import unittest
from pathlib import Path

import gate2_test_fixture_v1 as fixture


HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location(
    "build_qsgw_contract_overlay_v1",
    HERE / "build_qsgw_contract_overlay_v1.py",
)
assert spec is not None and spec.loader is not None
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)


def sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def source_contract(stru_sha: str) -> bytes:
    return (
        "# librpa-qsgw-input-contract-v1\n"
        "producer abacus\n"
        "role sha256 file\n"
        f"reader_static {stru_sha} stru_out\n"
        f"reader_static {'1' * 64} basis_wfc_out\n"
    ).encode("ascii")


class ContractOverlayTests(unittest.TestCase):
    def paths(self, root: Path, *, duplicate: bool = False):
        source_stru = root / "source-stru"
        overlay_stru = root / "overlay-stru"
        source_stru.write_bytes(b"source structure\n")
        overlay_stru.write_bytes(b"source structure\nsymmetry tail\n")
        contract = source_contract(sha(source_stru.read_bytes()))
        if duplicate:
            contract += (
                f"reader_static {sha(source_stru.read_bytes())} stru_out\n"
            ).encode("ascii")
        source = root / "qsgw_input.contract"
        output = root / "derived.contract"
        report = root / "report.json"
        source.write_bytes(contract)
        return source, source_stru, overlay_stru, output, report

    def test_single_hash_replacement_passes(self):
        with fixture.scratch_directory(HERE) as root:
            paths = self.paths(root)
            report = builder.build(*paths)
            output = paths[3].read_bytes()
            restored = output.replace(
                report["overlay_stru_out_sha256"].encode("ascii"),
                report["source_stru_out_sha256"].encode("ascii"),
                1,
            )
            self.assertEqual(restored, paths[0].read_bytes())
        self.assertTrue(report["passed"])
        self.assertTrue(report["unchanged_except_stru_sha256"])
        self.assertEqual(report["replacement_count"], 1)

    def test_source_stru_hash_mismatch_is_rejected(self):
        with fixture.scratch_directory(HERE) as root:
            paths = self.paths(root)
            paths[1].write_bytes(b"changed source\n")
            with self.assertRaises(builder.ContractOverlayError):
                builder.build(*paths)

    def test_duplicate_stru_rows_are_rejected(self):
        with fixture.scratch_directory(HERE) as root:
            paths = self.paths(root, duplicate=True)
            with self.assertRaises(builder.ContractOverlayError):
                builder.build(*paths)

    def test_missing_stru_row_is_rejected(self):
        with fixture.scratch_directory(HERE) as root:
            paths = self.paths(root)
            paths[0].write_bytes(
                paths[0].read_bytes().replace(b" stru_out", b" other_file")
            )
            with self.assertRaises(builder.ContractOverlayError):
                builder.build(*paths)

    def test_identical_source_and_overlay_are_rejected(self):
        with fixture.scratch_directory(HERE) as root:
            paths = self.paths(root)
            paths[2].write_bytes(paths[1].read_bytes())
            with self.assertRaises(builder.ContractOverlayError):
                builder.build(*paths)


if __name__ == "__main__":
    unittest.main(verbosity=2)
