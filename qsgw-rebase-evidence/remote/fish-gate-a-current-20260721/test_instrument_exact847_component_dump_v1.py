#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent
MODULE_PATH = HERE / "instrument_exact847_component_dump_v1.py"
SPEC = importlib.util.spec_from_file_location("instrument_under_test", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)
SOURCE = ("prefix\n" + MODULE.ANCHOR + "suffix\n").encode("utf-8")


class Exact847InstrumentationTests(unittest.TestCase):
    def test_instruments_frozen_scheme_a_source(self):
        expected = MODULE.EXPECTED_SOURCE_SHA256
        try:
            MODULE.EXPECTED_SOURCE_SHA256 = MODULE.sha256_bytes(SOURCE)
            output = MODULE.instrument(SOURCE)
            text = output.decode("utf-8")
            self.assertEqual(text.count("LIBRPA_QSGW_LEGACY_COMPONENT_DUMP"), 1)
            self.assertEqual(text.count("dump_static_component(\"vc\", Vc_all)"), 1)
            self.assertEqual(text.count("sigma_c_iw_"), 1)
            self.assertIn("librpa-exact847-component-dump-v1", text)
        finally:
            MODULE.EXPECTED_SOURCE_SHA256 = expected

    def test_rejects_modified_source(self):
        with self.assertRaisesRegex(MODULE.InstrumentationError, "sha256 mismatch"):
            MODULE.instrument(SOURCE)


if __name__ == "__main__":
    unittest.main()
