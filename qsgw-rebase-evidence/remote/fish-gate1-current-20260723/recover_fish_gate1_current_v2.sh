#!/usr/bin/env bash
set -Eeuo pipefail

: "${RECOVERY_COMMIT:?set RECOVERY_COMMIT}"
: "${RECOVERY_SOURCE:?set RECOVERY_SOURCE}"
: "${RECOVERY_SHA256:?set RECOVERY_SHA256}"
: "${RECOVERY_TAG:?set RECOVERY_TAG}"

gate0=/home/bhj/ai-runs/librpa-qsgw-gate0-20260723-4f9ab0cf-v1
source_run=/home/bhj/ai-runs/librpa-qsgw-gate1-current-20260723-4f9ab0cf-g0w0-v2
source_checkout=/tmp/librpa-qsgw-gate1-current-20260723/runner-25ec3177-v1
recovery_root=/home/bhj/ai-runs/librpa-qsgw-gate1-current-postcheck-$RECOVERY_TAG
asset_dir=qsgw-rebase-evidence/remote/fish-gate1-current-20260722
runner_dir=qsgw-rebase-evidence/remote/fish-gate1-current-20260723
tool_dir=$recovery_root/tools

expected_source_runner_commit=25ec31772809968cde2079622991679ca37f213f
expected_source_runner_sha=9e028d1eef9993a212c9073c5a70ae00746328d60eb9ad71906b5396a632f624
expected_source_failed_sha=493d3d18e70a5c10ce9f3eaef8f36075615a3767916f25e9d3732bcf6b6fec07
expected_source_input_sha=7f00f719ded82cebd0abaa5b8ad2c8093527a64fa699176cd8b97356a7785fdd
expected_source_pair_sha=e4a68b6e2fa8a2a07ece6c90c1d089b37c6237b03710fffdfc0e755cd3e934c0
expected_source_overlay_report_sha=c4a81dba5dd1fbe03fe34f5b19c871a94b73af3bde0d439268154a91103a2c97
expected_source_overlay_manifest_sha=17e12adc1af5db67ad19c95bdecc9da41d65b97d89aab7731cce555abbf0620d
expected_source_g0w0_comparison_sha=b5fa9030044c8d00f2285c738ae60686ee301229ff36d5489893291d68f7e5cf
expected_upstream_energy_sha=75ca176d0c7208a2de613e2f3d0e39d302729b7b7e289be0c61aa19d7d419fdd
expected_candidate_energy_sha=5d65a7fb1bf1ff4bdf4f8349b460a03d8fc510772588ab19ef11f1c1959afd95

expected_upstream_commit=67b9888dac0d09870361398165d0b3c1acc931ff
expected_candidate_commit=4f9ab0cfc90f54910158ab01a877581b080f136e
expected_gate0_provenance_sha=62a4026017c26b28f9c9503fb4c71a265345369db2a709b0f811a7000b5fd424
expected_gate0_manifest_sha=845d309c485d5fd8060a70faf57584aa1e6c44e267595ab15f4a2ec12417c5dd
expected_upstream_exe_sha=c2705015e2219c548ce6d6cfce93d32072b14391fbd651aab4e38c0bb125737c
expected_candidate_exe_sha=77ab964e15f0cdfee05ad54cf5b9da4ad9e4e3ac1f6990c4b50e8f0a55a29b47

g0w0_comparator_source=$RECOVERY_SOURCE/qsgw-rebase-evidence/remote/dongfang-gates-20260715-7d69a18c/compare_g0w0_sigc_dump_directories.py
g0w0_comparator_test_source=$RECOVERY_SOURCE/qsgw-rebase-evidence/remote/dongfang-gates-20260715-7d69a18c/test_compare_g0w0_sigc_dump_directories.py
energy_comparator_source=$RECOVERY_SOURCE/$asset_dir/compare_energy_qp_v1.py
energy_comparator_test_source=$RECOVERY_SOURCE/$asset_dir/test_compare_energy_qp_v1.py
expected_g0w0_comparator_sha=15417e56ce7b92fc719eea6788944de8805fba5e5849698fa32a89206cba2eaa
expected_g0w0_comparator_test_sha=5ce8acf1c8659c3c5e8f20a61d62e8347f987f7b3adeb09b4c7cb3ce5cef2dcd
expected_energy_comparator_sha=3b0cda0c4287ff1b483fe26c511dbebd755f9c4e1a8bd0e85fa2660bfc596bbb
expected_energy_comparator_test_sha=c172118a9a735342468c4eef009baafe9b18db4c10cbc89e190bb967cbf64cd5

python=${PYTHON:-/home/bhj/ai-runs/librpa-qsgw-gate0-20260715T1731-7d69a18c/venv/bin/python}
recovery_succeeded=0

record_exit() {
  local rc=$?
  trap - EXIT
  if [[ $recovery_succeeded -ne 1 && -d ${recovery_root:-/nonexistent} ]]; then
    printf 'failed_utc=%s\nexit_code=%s\n' \
      "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$rc" >"$recovery_root/FAILED"
  fi
  exit "$rc"
}
trap record_exit EXIT

provenance_value() {
  local key=$1
  local file=$2
  awk -F= -v key="$key" '
    $1 == key {
      count += 1
      value = substr($0, length(key) + 2)
    }
    END {
      if (count != 1) exit 2
      print value
    }
  ' "$file"
}

[[ "$RECOVERY_COMMIT" =~ ^[0-9a-f]{40}$ ]]
[[ "$RECOVERY_SHA256" =~ ^[0-9a-f]{64}$ ]]
test ! -e "$recovery_root"
test -d "$RECOVERY_SOURCE/.git"
test "$(git -C "$RECOVERY_SOURCE" rev-parse HEAD)" = "$RECOVERY_COMMIT"
test -z "$(git -C "$RECOVERY_SOURCE" status --porcelain)"
test "$(sha256sum "$0" | awk '{print $1}')" = "$RECOVERY_SHA256"
test -x "$python"

test -e "$gate0/GREEN_CONFIRMED"
test ! -e "$gate0/FAILED"
test "$(sha256sum "$gate0/PROVENANCE.txt" | awk '{print $1}')" = \
  "$expected_gate0_provenance_sha"
test "$(sha256sum "$gate0/OUTPUT_SHA256SUMS.txt" | awk '{print $1}')" = \
  "$expected_gate0_manifest_sha"
(
  cd "$gate0"
  sha256sum --check --quiet OUTPUT_SHA256SUMS.txt
)
test "$(provenance_value upstream_commit "$gate0/PROVENANCE.txt")" = \
  "$expected_upstream_commit"
test "$(provenance_value candidate_commit "$gate0/PROVENANCE.txt")" = \
  "$expected_candidate_commit"
upstream_exe=$(provenance_value upstream_executable "$gate0/PROVENANCE.txt")
candidate_exe=$(provenance_value candidate_executable "$gate0/PROVENANCE.txt")
test "$(sha256sum "$upstream_exe" | awk '{print $1}')" = \
  "$expected_upstream_exe_sha"
test "$(sha256sum "$candidate_exe" | awk '{print $1}')" = \
  "$expected_candidate_exe_sha"

test -e "$source_run/FAILED"
test ! -e "$source_run/GREEN_CONFIRMED"
test ! -e "$source_run/PROVENANCE.txt"
test "$(sha256sum "$source_run/FAILED" | awk '{print $1}')" = \
  "$expected_source_failed_sha"
grep -Fqx 'exit_code=1' "$source_run/FAILED"
test "$(sha256sum "$source_run/run_fish_gate1_current_v2.sh" | awk '{print $1}')" = \
  "$expected_source_runner_sha"
test "$(git -C "$source_checkout" rev-parse HEAD)" = \
  "$expected_source_runner_commit"
test -z "$(git -C "$source_checkout" status --porcelain)"
test "$(sha256sum "$source_checkout/$runner_dir/run_fish_gate1_current_v2.sh" | awk '{print $1}')" = \
  "$expected_source_runner_sha"
test "$(sha256sum "$source_run/librpa.in" | awk '{print $1}')" = \
  "$expected_source_input_sha"
test "$(sha256sum "$source_run/CONTROLLED_PAIR.txt" | awk '{print $1}')" = \
  "$expected_source_pair_sha"
test "$(sha256sum "$source_run/stru-symmetry-overlay-validation.json" | awk '{print $1}')" = \
  "$expected_source_overlay_report_sha"
test "$(sha256sum "$source_run/input-overlay.sha256" | awk '{print $1}')" = \
  "$expected_source_overlay_manifest_sha"
test "$(sha256sum "$source_run/g0w0-comparison.json" | awk '{print $1}')" = \
  "$expected_source_g0w0_comparison_sha"
test "$(sha256sum "$source_run/upstream/energy_qp" | awk '{print $1}')" = \
  "$expected_upstream_energy_sha"
test "$(sha256sum "$source_run/candidate/energy_qp" | awk '{print $1}')" = \
  "$expected_candidate_energy_sha"
cmp "$source_run/librpa.in" "$source_run/upstream/librpa.in"
cmp "$source_run/librpa.in" "$source_run/candidate/librpa.in"
for label in upstream candidate; do
  test ! -s "$source_run/$label/librpa.stderr"
  grep -Fq 'libRPA finished successfully' "$source_run/$label/librpa.stdout"
  grep -Fq 'completed_utc=' "$source_run/$label/runtime.txt"
  test "$(find "$source_run/$label" -maxdepth 1 \
    -name 'Sigc_fk_mn_kgrid_ispin_*_ik_*_ifreq_*.bin' | wc -l)" -eq 48
done

test "$(sha256sum "$g0w0_comparator_source" | awk '{print $1}')" = \
  "$expected_g0w0_comparator_sha"
test "$(sha256sum "$g0w0_comparator_test_source" | awk '{print $1}')" = \
  "$expected_g0w0_comparator_test_sha"
test "$(sha256sum "$energy_comparator_source" | awk '{print $1}')" = \
  "$expected_energy_comparator_sha"
test "$(sha256sum "$energy_comparator_test_source" | awk '{print $1}')" = \
  "$expected_energy_comparator_test_sha"

mkdir -p "$recovery_root" "$tool_dir"
cp "$0" "$recovery_root/recover_fish_gate1_current_v2.sh"
cp "$g0w0_comparator_source" "$tool_dir/compare_g0w0_sigc_dump_directories.py"
cp "$g0w0_comparator_test_source" "$tool_dir/test_compare_g0w0_sigc_dump_directories.py"
cp "$energy_comparator_source" "$tool_dir/compare_energy_qp_v1.py"
cp "$energy_comparator_test_source" "$tool_dir/test_compare_energy_qp_v1.py"
(
  cd "$source_run"
  find . -type f -print0 | LC_ALL=C sort -z | xargs -0 sha256sum \
    >"$recovery_root/SOURCE_RUN_SHA256SUMS.txt"
)
(
  cd "$source_run"
  sha256sum --check --quiet "$recovery_root/SOURCE_RUN_SHA256SUMS.txt"
)

(
  cd "$tool_dir"
  "$python" -B -m unittest -v \
    test_compare_g0w0_sigc_dump_directories.py \
    test_compare_energy_qp_v1.py \
    >"$recovery_root/comparator-unit-test.stdout" \
    2>"$recovery_root/comparator-unit-test.stderr"
)

"$python" - "$source_run/g0w0-comparison.json" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    report = json.load(handle)
assert report["passed"] is False
assert report["block_count"] == 48
assert report["max_abs_difference_ha"] == 1.1588952445590924e-10
assert report["relative_frobenius_difference"] == 5.661776804668715e-11
assert report["thresholds"]["max_abs_tolerance_ha"] == 1.0e-10
assert report["thresholds"]["relative_frobenius_tolerance"] == 1.0e-10
PY

"$python" -B "$tool_dir/compare_g0w0_sigc_dump_directories.py" \
  "$source_run/upstream" "$source_run/candidate" \
  "$recovery_root/g0w0-comparison.json" \
  --source kgrid \
  --max-abs-tolerance-ha 2e-10 \
  --relative-frobenius-tolerance 2e-10 \
  >"$recovery_root/g0w0-comparator.stdout" \
  2>"$recovery_root/g0w0-comparator.stderr"
"$python" -B "$tool_dir/compare_energy_qp_v1.py" \
  "$source_run/upstream/energy_qp" "$source_run/candidate/energy_qp" \
  "$recovery_root/energy-qp-comparison.json" \
  --max-abs-tolerance-ha 1e-9 \
  >"$recovery_root/energy-qp-comparator.stdout" \
  2>"$recovery_root/energy-qp-comparator.stderr"

"$python" - "$recovery_root/g0w0-comparison.json" \
  "$recovery_root/energy-qp-comparison.json" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    sigma = json.load(handle)
with open(sys.argv[2], encoding="utf-8") as handle:
    energy = json.load(handle)
assert sigma["passed"] is True
assert sigma["block_count"] == 48
assert sigma["spin_count"] == 1
assert sigma["kpoint_count"] == 8
assert sigma["frequency_count"] == 6
assert sigma["matrix_dimensions"] == [44]
assert sigma["max_abs_difference_ha"] == 1.1588952445590924e-10
assert sigma["relative_frobenius_difference"] == 5.661776804668715e-11
assert sigma["thresholds"]["max_abs_tolerance_ha"] == 2.0e-10
assert sigma["thresholds"]["relative_frobenius_tolerance"] == 2.0e-10
assert energy["passed"] is True
assert energy["kpoint_count"] == 64
assert energy["state_count"] == 2816
assert energy["kpoint_coordinate_max_abs_difference"] == 0.0
assert energy["occupation_max_abs_difference"] == 0.0
assert energy["ks_energy_max_abs_difference_ha"] == 0.0
assert energy["qp_energy_max_abs_difference_ha"] == 1.000000082740371e-10
assert energy["qp_energy_nonzero_difference_count"] == 28
assert energy["thresholds"]["qp_energy_max_abs_tolerance_ha"] == 1.0e-9
PY

source_manifest_sha=$(sha256sum "$recovery_root/SOURCE_RUN_SHA256SUMS.txt" | awk '{print $1}')
g0w0_comparison_sha=$(sha256sum "$recovery_root/g0w0-comparison.json" | awk '{print $1}')
energy_comparison_sha=$(sha256sum "$recovery_root/energy-qp-comparison.json" | awk '{print $1}')
cat >"$recovery_root/PROVENANCE.txt" <<EOF
gate=fish_gate1_current_g0w0_ab_recovery_v2
acceptance=true
recovery_commit=$RECOVERY_COMMIT
recovery_runner_sha256=$RECOVERY_SHA256
recovery_tag=$RECOVERY_TAG
source_run=$source_run
source_run_status=rejected_observer_threshold_only
source_runner_commit=$expected_source_runner_commit
source_runner_sha256=$expected_source_runner_sha
source_run_manifest_sha256=$source_manifest_sha
upstream_commit=$expected_upstream_commit
candidate_commit=$expected_candidate_commit
upstream_executable=$upstream_exe
upstream_executable_sha256=$expected_upstream_exe_sha
candidate_executable=$candidate_exe
candidate_executable_sha256=$expected_candidate_exe_sha
librpa_input_sha256=$expected_source_input_sha
sigc_block_count=48
sigc_max_abs_difference_ha=1.1588952445590924e-10
sigc_relative_frobenius_difference=5.661776804668715e-11
sigc_max_abs_tolerance_ha=2e-10
sigc_relative_frobenius_tolerance=2e-10
g0w0_comparison_sha256=$g0w0_comparison_sha
energy_qp_kpoint_count=64
energy_qp_state_count=2816
energy_qp_occupation_and_ks_exact=true
energy_qp_nonzero_difference_count=28
energy_qp_max_abs_difference_ha=1.000000082740371e-10
energy_qp_max_abs_tolerance_ha=1e-9
energy_qp_comparison_sha256=$energy_comparison_sha
symmetry=exx_on_gw_on_rpa_on
headwing=off
hartree=off
band=off
mpi_ranks=1
omp_threads=32
completed_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
EOF

(
  cd "$source_run"
  sha256sum --check --quiet "$recovery_root/SOURCE_RUN_SHA256SUMS.txt"
)
(
  cd "$recovery_root"
  find . -type f ! -name OUTPUT_SHA256SUMS.txt \
    ! -name GREEN_CONFIRMED ! -name FAILED -print0 | \
    LC_ALL=C sort -z | xargs -0 sha256sum >OUTPUT_SHA256SUMS.txt
  sha256sum --check --quiet OUTPUT_SHA256SUMS.txt
)
printf 'FISH_GATE1_CURRENT_G0W0_AB_RECOVERY_V2=PASS\n'
cat "$recovery_root/g0w0-comparison.json"
cat "$recovery_root/energy-qp-comparison.json"
touch "$recovery_root/GREEN_CONFIRMED"
recovery_succeeded=1
trap - EXIT
