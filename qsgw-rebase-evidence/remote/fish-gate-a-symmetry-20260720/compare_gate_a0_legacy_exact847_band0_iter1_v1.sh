#!/usr/bin/env bash
set -euo pipefail

: "${RUNNER_SHA256:?RUNNER_SHA256 must identify this exact runner}"

run_root=/home/bhj/ai-runs/librpa-qsgw-gate-a0-legacy-exact847-band0-iter1-20260720-v1
reproduced=$run_root/work
oracle_bundle=/home/bhj/ai-runs/librpa-qsgw-gate-a-historical-band0-oracle-20260720-v1-2380156
oracle=$oracle_bundle/oracle
staging=/tmp/librpa-qsgw-gate-a-band0-import-20260720-v1
evidence=/home/bhj/ai-runs/librpa-qsgw-gate-a0-legacy-exact847-band0-comparison-20260720-v1
v1_name=compare_legacy_band0_native_outputs_v1.py
v2_name=compare_legacy_band0_native_outputs_v2.py
v2_test_name=test_compare_legacy_band0_native_outputs_v2.py
expected_v1_sha=350e6589b74a27a588d44ba7f1b53f42bcd46cfebd4c9a5c7658debdb520e88d
expected_v2_sha=69bb86ecf7675e644fbd500a49e0f44c92950aba478e7f764bcea30c28f7afab
expected_v2_test_sha=4cd649cdb65dae707f698780e860acea5454be86c2c4f7419ee8c6c10804e551
expected_oracle_manifest_sha=1b9646b744d82de506fdfb0487644427e47ec0430929944430a9b9c586aba800

record_failure() {
  local rc=$?
  trap - ERR
  if [[ -d ${evidence:-/nonexistent} ]]; then
    printf 'failed_utc=%s\nexit_code=%s\n' \
      "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$rc" >"$evidence/FAILED"
  fi
  exit "$rc"
}
trap record_failure ERR

test ! -e "$evidence"
test -e "$run_root/RUN_GREEN"
test ! -e "$run_root/FAILED"
test -e "$oracle_bundle/COMPLETE"
test ! -e "$oracle_bundle/FAILED"
test "$(sha256sum "$oracle_bundle/ORACLE_SHA256SUMS.txt" | awk '{print $1}')" = \
  "$expected_oracle_manifest_sha"
test "$(sha256sum "$staging/$v1_name" | awk '{print $1}')" = "$expected_v1_sha"
test "$(sha256sum "$staging/$v2_name" | awk '{print $1}')" = "$expected_v2_sha"
test "$(sha256sum "$staging/$v2_test_name" | awk '{print $1}')" = \
  "$expected_v2_test_sha"
(
  cd "$oracle"
  sha256sum --check --quiet "$oracle_bundle/ORACLE_SHA256SUMS.txt"
)
(
  cd "$run_root"
  sha256sum --check --quiet OUTPUT_SHA256SUMS.txt
)

mkdir -p "$evidence/tools"
install -m 0444 "$staging/$v1_name" "$evidence/tools/$v1_name"
install -m 0444 "$staging/$v2_name" "$evidence/tools/$v2_name"
install -m 0444 "$staging/$v2_test_name" "$evidence/tools/$v2_test_name"
printf '%s\n' "$RUNNER_SHA256" >"$evidence/runner-sha256.txt"

(
  cd "$evidence/tools"
  python3 -m unittest "$v2_test_name"
) >"$evidence/comparator-tests.stdout" 2>"$evidence/comparator-tests.stderr"

python3 "$evidence/tools/$v2_name" "$oracle" "$reproduced" \
  "$evidence/native-comparison-v2.json" \
  >"$evidence/comparator.stdout" 2>"$evidence/comparator.stderr"

grep -Fq '"historical_reproduction_passed": true' \
  "$evidence/native-comparison-v2.json"
grep -Fq '"legacy_oracle_invariant_gap": true' \
  "$evidence/native-comparison-v2.json"
grep -Fq '"absolute_invariants_passed": false' \
  "$evidence/native-comparison-v2.json"
grep -Fq '"goal_thresholds_passed": false' \
  "$evidence/native-comparison-v2.json"

python3 -c \
  'import json,sys; r=json.load(open(sys.argv[1])); h=r["h0"]; print("historical_reproduction_passed="+str(r["historical_reproduction_passed"]).lower()); print("goal_thresholds_passed="+str(r["goal_thresholds_passed"]).lower()); print("legacy_oracle_invariant_gap="+str(r["legacy_oracle_invariant_gap"]).lower()); print("h0_max_abs_ha="+repr(h["max_abs_ha"])); print("h0_max_relative_frobenius="+repr(h["max_relative_frobenius"])); print("oracle_hermiticity_max_abs_ha="+repr(h["oracle_hermiticity_max_abs_ha"])); print("reproduced_hermiticity_max_abs_ha="+repr(h["reproduced_hermiticity_max_abs_ha"])); print("sigcrf_max_abs_ha="+repr(r["sigcrf"]["max_abs_ha"])); print("band_max_abs_ev="+repr(r["text_outputs"]["max_abs_ev"]))' \
  "$evidence/native-comparison-v2.json" >"$evidence/acceptance-summary.txt"

cat >"$evidence/PROVENANCE.txt" <<EOF
gate=gate_a0_legacy_exact847_qsgw_band0_iter1_native_comparison_v1
acceptance=historical_reproduction_parity
historical_reproduction_passed=true
goal_thresholds_passed=false
goal_threshold_gap=pre_existing_legacy_H0_hermiticity
thresholds_relaxed=false
oracle=$oracle
oracle_manifest_sha256=$expected_oracle_manifest_sha
reproduced=$reproduced
reproduced_manifest=$run_root/OUTPUT_SHA256SUMS.txt
reproduced_manifest_sha256=$(sha256sum "$run_root/OUTPUT_SHA256SUMS.txt" | awk '{print $1}')
comparator_v1_sha256=$expected_v1_sha
comparator_v2_sha256=$expected_v2_sha
comparator_v2_test_sha256=$expected_v2_test_sha
runner_sha256=$RUNNER_SHA256
completed_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
EOF

touch "$evidence/LEGACY_PARITY_GREEN"
touch "$evidence/LEGACY_ABSOLUTE_INVARIANT_GAP"
(
  cd "$evidence"
  find . -type f ! -name OUTPUT_SHA256SUMS.txt ! -name COMPLETE \
    ! -name FAILED -print0 | LC_ALL=C sort -z | xargs -0 sha256sum \
    >OUTPUT_SHA256SUMS.txt
  sha256sum --check --quiet OUTPUT_SHA256SUMS.txt
)
touch "$evidence/COMPLETE"
find "$evidence" -type d -exec chmod 0555 {} +
find "$evidence" -type f -exec chmod 0444 {} +

echo GATE_A0_LEGACY_EXACT847_BAND0_NATIVE_COMPARISON_V1=PASS
cat "$evidence/acceptance-summary.txt"
