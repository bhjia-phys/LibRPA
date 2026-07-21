#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import shutil
import unittest
import uuid
from contextlib import contextmanager
from pathlib import Path

import validate_gate_a_input_views_v1 as validator


HERE = Path(__file__).resolve().parent


@contextmanager
def scratch_directory():
    path = HERE / f"input-view-test-{uuid.uuid4().hex}"
    path.mkdir()
    try:
        yield path
    finally:
        shutil.rmtree(path)


def sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def contract(stru: bytes, *, extra: str = "") -> bytes:
    return (
        "# librpa-qsgw-input-contract-v1\n"
        "producer abacus\n"
        "role sha256 file\n"
        f"reader_static {sha(stru)} stru_out\n"
        f"reader_static {'1' * 64} band_out\n"
        f"{extra}"
    ).encode("ascii")


class InputViewTests(unittest.TestCase):
    def views(self, root: Path) -> tuple[Path, Path]:
        legacy = root / "legacy"
        candidate = root / "candidate"
        legacy.mkdir()
        candidate.mkdir()
        source = b"legacy numeric stru\n"
        tail = (
            b"2 row\n"
            b"1 0 0 0 1 0 0 0 1 0 0 0\n"
            b"-1 0 0 0 -1 0 0 0 -1 0.25 0.25 0.25\n"
        )
        composite = source + tail
        for view in (legacy, candidate):
            (view / "band_out").write_bytes(b"same bands\n")
            (view / "vxc_out").write_bytes(b"same vxc\n")
        (legacy / "stru_out").write_bytes(source)
        (candidate / "stru_out").write_bytes(composite)
        (legacy / "qsgw_input.contract").write_bytes(contract(source))
        (candidate / "qsgw_input.contract").write_bytes(contract(composite))
        return legacy, candidate

    def test_accepts_only_reader_metadata_differences(self) -> None:
        with scratch_directory() as root:
            legacy, candidate = self.views(root)
            report = validator.validate(legacy, candidate)
        self.assertTrue(report["passed"])
        self.assertTrue(report["common_file_sha256_identical"])
        self.assertTrue(report["source_prefix_byte_identical"])
        self.assertTrue(report["contracts_unchanged_except_stru_sha256"])
        self.assertEqual(report["symmetry_operation_count"], 2)
        self.assertEqual(report["common_physical_file_count"], 2)

    def test_rejects_changed_common_file(self) -> None:
        with scratch_directory() as root:
            legacy, candidate = self.views(root)
            (candidate / "band_out").write_bytes(b"changed bands\n")
            with self.assertRaisesRegex(validator.InputViewError, "allowlist"):
                validator.validate(legacy, candidate)

    def test_rejects_different_file_sets(self) -> None:
        with scratch_directory() as root:
            legacy, candidate = self.views(root)
            (candidate / "extra").write_bytes(b"unexpected\n")
            with self.assertRaisesRegex(validator.InputViewError, "file sets differ"):
                validator.validate(legacy, candidate)

    def test_rejects_non_prefix_composite_stru(self) -> None:
        with scratch_directory() as root:
            legacy, candidate = self.views(root)
            value = (candidate / "stru_out").read_bytes().replace(
                b"legacy", b"changed", 1
            )
            (candidate / "stru_out").write_bytes(value)
            (candidate / "qsgw_input.contract").write_bytes(contract(value))
            with self.assertRaisesRegex(validator.InputViewError, "byte-prefixed"):
                validator.validate(legacy, candidate)

    def test_rejects_unrelated_contract_change(self) -> None:
        with scratch_directory() as root:
            legacy, candidate = self.views(root)
            composite = (candidate / "stru_out").read_bytes()
            (candidate / "qsgw_input.contract").write_bytes(
                contract(composite, extra="reader_static " + "2" * 64 + " extra\n")
            )
            with self.assertRaisesRegex(validator.InputViewError, "outside"):
                validator.validate(legacy, candidate)


if __name__ == "__main__":
    unittest.main(verbosity=2)
