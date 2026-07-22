import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "qsgw-rebase-evidence/validation/refresh-revised-goal-manifest.ps1"
MANIFEST = ROOT / "qsgw-rebase-manifest.json"

UPSTREAM = "67b9888dac0d09870361398165d0b3c1acc931ff"
PRODUCT = "4f9ab0cfc90f54910158ab01a877581b080f136e"
NEW_CHANGES = {
    "UP-ELPA-DEVICE-ALLOC-67B-001": "U1",
    "UP-DDLA-BUNDLE-67B-001": "U1",
    "UP-HEADWING-BODY-SOLVE-67B-001": "U1",
    "UP-HEAD-RANK1-67B-001": "U1",
    "UP-DDLA-REVISION-67B-001": "U0",
    "UP-HEADWING-BODY-CLEANUP-67B-001": "U1",
    "UP-DIELECTRIC-SOLVE-ERROR-67B-001": "U1",
}


class RefreshManifest67bTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.script = SCRIPT.read_text(encoding="utf-8")
        cls.manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))

    def test_script_binds_current_upstream_branch_and_gates(self):
        expected = (
            f"$upstream = '{UPSTREAM}'",
            "$frozenParent = $upstream",
            "$branch = 'codex/qsgw-symmetry-no-headwing-67b-20260723'",
            "fish-gate0-current-20260723\\4f9ab0cf-v1",
            "gate = 'fish_gate0_current_v2'",
            "run_fish_gate0_current_v2.sh",
            "[string]$Gate1Evidence = ''",
            "fish-gate1-current-20260723\\36d74369-recovery-v1",
            "gate = 'fish_gate1_current_g0w0_ab_recovery_v2'",
            "recover_fish_gate1_current_v2.sh",
        )
        for value in expected:
            self.assertIn(value, self.script)

    def test_script_hashes_lf_runner_bytes(self):
        self.assertIn("function Get-LfTextSha256", self.script)
        self.assertIn('$text.Replace("`r`n", "`n")', self.script)
        self.assertIn("$runnerHash = Get-LfTextSha256", self.script)

    def test_manifest_has_complete_current_inventory(self):
        commits = self.manifest["repository"]["commits"]
        self.assertEqual(commits["upstream_new"]["hash"], UPSTREAM)
        self.assertEqual(commits["rebase_head"]["hash"], PRODUCT)
        inventory = self.manifest["upstream_inventory"]
        self.assertEqual(len(inventory["commit_hashes"]), 37)
        self.assertEqual(len(inventory["commit_change_map"]), 37)
        self.assertEqual(inventory["coverage_assertion"], "complete")

    def test_manifest_classifies_all_new_commits(self):
        changes = {item["id"]: item for item in self.manifest["upstream_changes"]}
        for change_id, classification in NEW_CHANGES.items():
            self.assertEqual(changes[change_id]["classification"], classification)
            self.assertEqual(
                changes[change_id]["commit_reference"],
                "qsgw-rebase-evidence/git/upstream-refresh-42d3863c-to-67b9888d.md",
            )

    def test_manifest_has_current_formula_impacts_and_gates(self):
        formula_ids = {
            item["formula_id"] for item in self.manifest["formula_code_impacts"]
        }
        self.assertTrue(
            {
                "F-DDLA-DEVICE-CONTRACT-67B",
                "F-HEADWING-BODY-INVERSE-67B",
                "F-GAMMA-HEAD-RANK1-67B",
            }.issubset(formula_ids)
        )
        self.assertEqual(
            self.manifest["planning_state"]["fish_gate0"],
            "accepted_39_upstream_63_candidate_10_focused_protected_diff_empty",
        )
        self.assertEqual(
            self.manifest["planning_state"]["fish_gate1"],
            "accepted_sigc_48_qp_2816_source_outputs_immutable_postcheck",
        )
        gate1 = self.manifest["benchmarks"][1]
        self.assertEqual(gate1["id"], "g0w0-upstream-vs-rebased")
        self.assertEqual(gate1["status"], "accepted")
        self.assertTrue(
            {
                "upstream-g0w0-log",
                "rebased-g0w0-log",
                "g0w0-comparison",
                "gate1-provenance",
                "gate1-source-manifest",
            }.issubset({artifact["id"] for artifact in gate1["artifacts"]})
        )
        self.assertEqual(
            self.manifest["current_gate"], "qsgw-iter0-vs-upstream-g0w0"
        )


if __name__ == "__main__":
    unittest.main()
