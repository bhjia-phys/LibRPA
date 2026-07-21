#!/usr/bin/env bash
set -euo pipefail

: "${RUNNER_SHA256:?RUNNER_SHA256 must identify this exact runner}"

base=/home/bhj/ai-runs
staging=/tmp/librpa-qsgw-gate-a-band0-import-20260720-v1/dongfang-historical-toolchain-baseline-2380797
run_root=$base/librpa-qsgw-gate-a0-historical-toolchain-attestation-20260720-v1

expected_provenance_sha=38900d60720e8da7ecd812489214df73c933bf5b17e0ae59dda2dc8057ff86f3
expected_summary_sha=e68fe3f43b8d3909e9023bafc91022d00c9ef0614bfcfb0eb69d512794fdf617
expected_comparison_sha=0c1fe4253bb38f6444ac6936730b92a93c6cc849b101334a912953b5fa1e1a48
expected_comparison_rc_sha=9a271f2a916b0b6ee6cecb2426f0b3206ef074578be55d9bc94f6f3fe3ab86aa
expected_iteration_sha=aac335a593aef775debb19c5753566637fa6981ff7e2a65903451814b7c84a28
expected_source_map_sha=6318362f17c563cb7a28368414b1af7622b29501de7e5e870575c3a9025cdbb7
expected_remote_manifest_sha=9790e084ca906dea3b088d9c83dad698c1c5967c06d66d78fcc46a162764bd6a
expected_remote_runner_sha=0b6c6509dae60804e76b9e46457989fddcdd363887a22f197efd6ab0af2469df
expected_remote_runner_record_sha=9894a20d2acf6b65db68b89c3f7e9a3f9f9e70df4e941a3417129b7bc9129e81
expected_runtime_provenance_sha=3560531c121402423503e12322fd89580316d87b3abbdfec751182266813976e
expected_runtime_summary_sha=6a4ef2835c29d97aa1b0b418705556cdb8247891f72468918c8dfef74c5ab77d
expected_runtime_comparison_sha=6038c5fd7e3784f2df6b7f12404d0cdf5c231892acdc5a61b91d5df8212fc2ab
expected_runtime_manifest_sha=31f04e04fe44bd22827dbfcee0d3c41c023d8de5bbbb8ea372af77c42d674d24
expected_runtime_sha_record=443f71313f793ebadbb80a956ba0148cac3cbfd6e61b0d67650f6aa68199c50a
expected_runtime_runner_record_sha=397122499504405ad7f24bb1c83d91bfe18cd02a986505c9698004bfd36422d9

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
test -d "$staging"
test -e "$staging/COMPLETE"
test -e "$staging/HISTORICAL_REPRODUCTION_GREEN"
test "$(sha256sum "$staging/PROVENANCE.txt" | awk '{print $1}')" = "$expected_provenance_sha"
test "$(sha256sum "$staging/diagnostic-summary.txt" | awk '{print $1}')" = "$expected_summary_sha"
test "$(sha256sum "$staging/native-comparison-v2.json" | awk '{print $1}')" = "$expected_comparison_sha"
test "$(sha256sum "$staging/native-comparison-v2.exit-code" | awk '{print $1}')" = "$expected_comparison_rc_sha"
test "$(sha256sum "$staging/iteration1-summary.txt" | awk '{print $1}')" = "$expected_iteration_sha"
test "$(sha256sum "$staging/historical-source-vs-bundle.json" | awk '{print $1}')" = "$expected_source_map_sha"
test "$(sha256sum "$staging/OUTPUT_SHA256SUMS.txt" | awk '{print $1}')" = "$expected_remote_manifest_sha"
test "$(sha256sum "$staging/runner.slurm" | awk '{print $1}')" = "$expected_remote_runner_sha"
test "$(sha256sum "$staging/runner-sha256.txt" | awk '{print $1}')" = "$expected_remote_runner_record_sha"
test "$(sha256sum "$staging/runtime-parent/PROVENANCE.txt" | awk '{print $1}')" = "$expected_runtime_provenance_sha"
test "$(sha256sum "$staging/runtime-parent/diagnostic-summary.txt" | awk '{print $1}')" = "$expected_runtime_summary_sha"
test "$(sha256sum "$staging/runtime-parent/band-comparison.json" | awk '{print $1}')" = "$expected_runtime_comparison_sha"
test "$(sha256sum "$staging/runtime-parent/OUTPUT_SHA256SUMS.txt" | awk '{print $1}')" = "$expected_runtime_manifest_sha"
test "$(sha256sum "$staging/runtime-parent/runtime-SHA256SUMS.txt" | awk '{print $1}')" = "$expected_runtime_sha_record"
test "$(sha256sum "$staging/runtime-parent/runner-sha256.txt" | awk '{print $1}')" = "$expected_runtime_runner_record_sha"
test "$(cat "$staging/native-comparison-v2.exit-code")" = 0
test "$(cat "$staging/runner-sha256.txt")" = "$expected_remote_runner_sha"

mkdir -p "$run_root/evidence/runtime-parent"
cp "$staging/PROVENANCE.txt" "$staging/diagnostic-summary.txt" \
  "$staging/native-comparison-v2.json" \
  "$staging/native-comparison-v2.exit-code" \
  "$staging/iteration1-summary.txt" \
  "$staging/historical-source-vs-bundle.json" \
  "$staging/OUTPUT_SHA256SUMS.txt" "$staging/runner-sha256.txt" \
  "$staging/runner.slurm" "$run_root/evidence/"
cp "$staging/runtime-parent/"* "$run_root/evidence/runtime-parent/"
printf '%s\n' "$RUNNER_SHA256" >"$run_root/import-runner-sha256.txt"

python3 -c \
  'import json,sys; r=json.load(open(sys.argv[1])); h=r["h0"]; s=r["sigcrf"]; t=r["text_outputs"]; assert r["schema"]=="librpa-legacy-qsgw-band0-native-comparison-v2"; assert r["historical_reproduction_passed"] is True; assert r["goal_thresholds_passed"] is False; assert r["legacy_oracle_invariant_gap"] is True; assert h["max_abs_ha"] <= 1e-6; assert h["max_relative_frobenius"] <= 1e-8; assert h["oracle_hermiticity_max_abs_ha"] > 1e-10; assert h["reproduced_hermiticity_max_abs_ha"] > 1e-10; assert s["max_abs_ha"] <= 1e-6; assert s["max_relative_frobenius"] <= 1e-8; assert t["max_abs_ev"] <= 1e-5; assert t["all_byte_equal"] is True; print("historical_reproduction_passed=true"); print("text_outputs_byte_equal=true"); print("legacy_raw_h0_invariant_gap=true")' \
  "$run_root/evidence/native-comparison-v2.json" \
  >"$run_root/validation-summary.txt"
python3 -c \
  'import json,sys; r=json.load(open(sys.argv[1])); assert r["manifest_entries"]==1334; assert r["existing_entries"]==1123; assert r["matching_entries"]==1123; assert r["mismatch_count"]==0; print("historical_input_matches=1123/1123"); print("historical_input_mismatches=0")' \
  "$run_root/evidence/historical-source-vs-bundle.json" \
  >>"$run_root/validation-summary.txt"

cat >"$run_root/PROVENANCE.txt" <<EOF
gate=gate_a0_historical_toolchain_attestation_v1
acceptance=true
source_host=dongfang
source_slurm_job_id=2380797
source_slurm_state=COMPLETED
source_slurm_exit_code=0:0
source_slurm_elapsed=00:11:07
source_slurm_node=c21430
legacy_commit=8476213f66c68efb43404713eacbd04966820f26
runtime_executable_sha256=33fbed74ff744e3e6e85fcb825603d1fb478440b3500f3fb6d058a68726f14a1
runtime_toolchain=oneAPI_2024.2_mpiicpx_2021.13_ifx_2024.2
historical_reproduction_passed=true
historical_text_outputs_byte_equal=true
legacy_raw_h0_invariant_gap=true
remote_output_manifest_sha256=$expected_remote_manifest_sha
remote_output_manifest_verified_on_source=true
remote_output_manifest_verification=sha256sum_check_quiet_exit_0
runtime_parent_output_manifest_sha256=$expected_runtime_manifest_sha
import_runner_sha256=$RUNNER_SHA256
imported_host=$(hostname -f 2>/dev/null || hostname)
imported_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
EOF
touch "$run_root/HISTORICAL_GRID_AND_BAND_PARITY_GREEN"
touch "$run_root/LEGACY_RAW_H0_INVARIANT_GAP"
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

echo GATE_A0_HISTORICAL_TOOLCHAIN_ATTESTATION_V1=PASS
cat "$run_root/validation-summary.txt"
