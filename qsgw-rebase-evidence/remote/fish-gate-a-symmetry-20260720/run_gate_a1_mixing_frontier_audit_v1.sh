#!/usr/bin/env bash
set -euo pipefail

: "${RUNNER_SHA256:?RUNNER_SHA256 must identify this exact runner}"

base=/home/bhj/ai-runs
run_root=$base/librpa-qsgw-gate-a1-mixing-frontier-audit-20260720-v1
bundle=$base/librpa-qsgw-gate-a-symmetry-band0-bundle-20260720-v2
dataset=$bundle/dataset
legacy_run=$base/librpa-qsgw-gate-a1-exact847-scheme-a-legacy-symmetry-nomix-miniter2-20260720-v4
checkpoint_root=$legacy_run/legacy/librpa.d/qsgw_checkpoints
staging=/tmp/librpa-qsgw-gate-a1-mixing-frontier-audit-20260720-v1
audit_name=audit_gate_a1_mixing_frontier_v1.py
test_name=test_audit_gate_a1_mixing_frontier_v1.py

expected_dataset_manifest_sha=7fe17e43e20978833cd3c4bead17943c1ed9f231e5b03b3e46d9f342e9904956
expected_bundle_output_sha=25590340b95edd0aaa91e3c8b34e332545afd9143ba96807a391290a7e0e9a43
expected_audit_sha=a9167f3bd821aaaec429a059b2493dd0f1b09e0a182a1841f1114c1a93355eb6
expected_test_sha=2efdb2caa4f18760b349c960b1858d60104429b10dc9b307e5372069c483fd69

record_failure() {
  local rc=$?
  trap - ERR
  if [[ -d ${run_root:-/nonexistent} ]]; then
    printf 'failed_utc=%s\nexit_code=%s\n' \
      "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$rc" >"$run_root/FAILED"
  fi
  exit "$rc"
}
trap record_failure ERR

test ! -e "$run_root"
test -e "$bundle/COMPLETE"
test ! -e "$bundle/FAILED"
test -e "$legacy_run/FAILED"
test ! -e "$legacy_run/COMPLETE"
test "$(sha256sum "$bundle/DATASET_SHA256SUMS.txt" | awk '{print $1}')" = \
  "$expected_dataset_manifest_sha"
test "$(sha256sum "$bundle/OUTPUT_SHA256SUMS.txt" | awk '{print $1}')" = \
  "$expected_bundle_output_sha"
test "$(sha256sum "$staging/$audit_name" | awk '{print $1}')" = \
  "$expected_audit_sha"
test "$(sha256sum "$staging/$test_name" | awk '{print $1}')" = \
  "$expected_test_sha"
test "$(find "$checkpoint_root/iter_00001" -maxdepth 1 \
  -name 'H0_GW_spin_01_k_*.bin' -type f | wc -l)" -eq 8

mkdir -p "$run_root/tools"
install -m 0444 "$staging/$audit_name" "$run_root/tools/$audit_name"
install -m 0444 "$staging/$test_name" "$run_root/tools/$test_name"
printf '%s\n' "$RUNNER_SHA256" >"$run_root/runner-sha256.txt"

cat >"$run_root/PROVENANCE.txt" <<EOF
gate=gate_a1_mixing_frontier_audit_v1
purpose=classify_symmetry_nomix_iter2_failure_before_candidate_edits
candidate_source_modified=false
input_bundle=$bundle
input_dataset_manifest_sha256=$expected_dataset_manifest_sha
legacy_checkpoint_source=$legacy_run
legacy_checkpoint_source_status=failed_downstream_iter2
legacy_checkpoint_iteration=1
betas=0,0.2,1
occupation=global_symmetry_weighted_charge_conserving
started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
EOF

(
  cd "$run_root/tools"
  python3 -m unittest "$test_name"
) >"$run_root/unit-tests.stdout" 2>"$run_root/unit-tests.stderr"

python3 "$run_root/tools/$audit_name" \
  "$dataset/band_out" \
  "$dataset/bz_sampling_out" \
  "$checkpoint_root" \
  "$run_root/mixing-frontier-audit.json" \
  --iteration 1 --betas 0,0.2,1 \
  >"$run_root/audit.stdout" 2>"$run_root/audit.stderr"

python3 - "$run_root/mixing-frontier-audit.json" \
  "$run_root/acceptance-summary.txt" <<'PY'
import json
import pathlib
import sys

report = json.loads(pathlib.Path(sys.argv[1]).read_text())
assert report["schema"] == "librpa-qsgw-mixing-frontier-audit-v1"
assert abs(report["target_electrons"] - 8.0) <= 1.0e-10
ks = report["reports"]["0"]
linear = report["reports"]["0.20000000000000001"]
direct = report["reports"]["1"]
for item in (ks, linear, direct):
    assert abs(item["electron_count_error"]) <= 1.0e-10
    assert item["hermiticity_max_abs_ha"] <= 1.0e-12
assert ks["metallic"] is False
assert ks["gap_ev"] > 0.0
assert linear["metallic"] is False
assert linear["gap_ev"] > 0.0
assert linear["fixed_four_band_gap_ev"] > 0.0
assert direct["metallic"] is True
assert direct["gap_ev"] == 0.0
assert direct["fixed_four_band_gap_ev"] < 0.0
assert len(direct["partial_groups"]) >= 1

summary = {
    "passed": True,
    "classification": (
        "direct_update_is_charge_conserving_metal;_linear_beta_0.2_"
        "preserves_an_insulating_frontier"
    ),
    "ks_gap_ev": ks["gap_ev"],
    "linear_beta_0.2_gap_ev": linear["gap_ev"],
    "direct_update_gap_ev": direct["gap_ev"],
    "direct_update_fixed_four_band_gap_ev": direct["fixed_four_band_gap_ev"],
    "direct_update_partial_groups": direct["partial_groups"],
}
pathlib.Path(sys.argv[2]).write_text(
    json.dumps(summary, indent=2, sort_keys=True) + "\n"
)
PY

printf 'acceptance=true_diagnostic_classification\ncompleted_utc=%s\n' \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >>"$run_root/PROVENANCE.txt"
touch "$run_root/DIAGNOSTIC_GREEN"
(
  cd "$run_root"
  find . -type f ! -name OUTPUT_SHA256SUMS.txt ! -name COMPLETE \
    ! -name FAILED -print0 | LC_ALL=C sort -z | xargs -0 sha256sum \
    >OUTPUT_SHA256SUMS.txt
  sha256sum --check --quiet OUTPUT_SHA256SUMS.txt
)
touch "$run_root/COMPLETE"
find "$run_root" -type d -exec chmod 0555 {} +
find "$run_root" -type f -exec chmod 0444 {} +

echo GATE_A1_MIXING_FRONTIER_AUDIT_V1=PASS
cat "$run_root/acceptance-summary.txt"
