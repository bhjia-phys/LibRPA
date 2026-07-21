import unittest
from pathlib import Path

from verify_historical_band0_source_against_bundle_v1 import compare

FIXTURE = Path(__file__).with_name("manifest_source_check_fixture")


class SourceBundleCheckTest(unittest.TestCase):
    def test_matching_existing_and_missing_generated_file(self):
        result = compare(FIXTURE / "manifest_match.txt", FIXTURE / "source")
        self.assertEqual(result["matching_entries"], 1)
        self.assertEqual(result["missing_count"], 1)
        self.assertEqual(result["mismatch_count"], 0)

    def test_detects_drift(self):
        result = compare(FIXTURE / "manifest_mismatch.txt", FIXTURE / "source")
        self.assertEqual(result["mismatch_count"], 1)


if __name__ == "__main__":
    unittest.main()
