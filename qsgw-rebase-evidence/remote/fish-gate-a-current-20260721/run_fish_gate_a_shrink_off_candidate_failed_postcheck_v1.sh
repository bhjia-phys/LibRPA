#!/usr/bin/env bash
set -euo pipefail

: "${RUNNER_COMMIT:?RUNNER_COMMIT must identify the clean checkout}"
: "${RUNNER_SOURCE:?RUNNER_SOURCE must identify the clean checkout}"
: "${RUNNER_SHA256:?RUNNER_SHA256 must identify this exact runner}"
: "${RUN_TAG:?RUN_TAG must make the diagnostic archive immutable}"

run_root=/home/bhj/ai-runs/librpa-qsgw-gate-a-shrink-off-failed-postcheck-$RUN_TAG
source_run=/home/bhj/ai-runs/librpa-qsgw-gate-a-shrink-off-candidate-recovery-20260722-07785fa4-v1
source_mode=$source_run/shrink-off-candidate-recovery-iter1
candidate=$source_mode/candidate
legacy_source=/home/bhj/ai-runs/librpa-qsgw-gate-a-shrink-off-iter1-20260722-398e354d-v1
legacy=$legacy_source/shrink-off-no-mix-iter1/legacy

expected_source_failed_sha=45906f6be09f8ab15f118272c1eeaca4f3d9f30124486e17b2715bb26a3ae395
expected_derivation_sha=f9dc5ffcc6e8af54f7b751bfb34c8e3dc907c134b0456a3e174028762665b12c
expected_full_contract_sha=5f7ebe1dfc4d4b18d374f80787b5450745478d469681ac6cf93c99b06c828bc1
expected_full_summary_sha=347e8f26473a42a3b05d3e705c0f8690af04800d5730acb2c15b288daf96cb23
expected_candidate_matrix_sha=f56151605e66901aef950d9a5d4338ed6c0a78f19af9fecbeb16e137fc5188ca
expected_candidate_eigenvalue_sha=b179042892a3407377ba42ba31652c914e637eb5826bd95a047463d5462ef4cc
expected_candidate_iteration_sha=7d44a5c90d12d34499e2b157eebda74528811c20f3a08ee08748939cc1fff191
expected_candidate_stdout_sha=96189ae19e3815b67b258c5922b203da2d1adf391e681f23419d0898db0b901c
expected_candidate_stderr_sha=e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
expected_candidate_input_sha=83acf0d943dcb78ed51b100129ed2a9748c92dbf8520d9c760e1674514cd657c
expected_candidate_runtime_sha=c9ae9a24a7c174124c51e31a85d27bd4df88e7ae27e3cbfea7397b46ab4ffb1a
expected_parameters_sha=20949df3a3a841cb4e18079ff1a5a1bf2bd0843f573c902005798031916249f9
expected_legacy_matrix_sha=ac3abf94bd72b59ed9e9b1e14a3d3556f12ed3ccb6721420141c6c7fc85ea54c
expected_legacy_history_sha=73a6d275604d515665e7dc1f124d45b99980830dd8c5e95837be148e8fade80a

symmetry_dir=$RUNNER_SOURCE/qsgw-rebase-evidence/remote/fish-gate-a-symmetry-20260720
current_dir=$RUNNER_SOURCE/qsgw-rebase-evidence/remote/fish-gate-a-current-20260721
base_comparator=$symmetry_dir/compare_qsgw_component_traces-v4-c3daf072.py
closure=$symmetry_dir/observer-tools-v1/validate_qsgw_trace_closure-v3-4a5de94e.py
fixed=$symmetry_dir/observer-tools-v1/validate_qsgw_fixed_basis.py
initial=$symmetry_dir/observer-tools-v1/validate_qsgw_initial_state-v1.py
adapter=$current_dir/compare_qsgw_legacy_v4_current_v6.py
summary=$current_dir/summarize_qsgw_trace_components_v1.py
parser=$RUNNER_SOURCE/regression_tests/backend/comparisons/cmp_qsgw.py
python=${PYTHON:-/home/bhj/ai-runs/librpa-qsgw-gate0-20260715T1731-7d69a18c/venv/bin/python}

run_succeeded=0
record_exit() {
  local rc=$?
  trap - EXIT
  if [[ $run_succeeded -ne 1 && -d ${run_root:-/nonexistent} ]]; then
    printf 'failed_utc=%s\nexit_code=%s\n' \
      "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$rc" >"$run_root/FAILED"
  fi
  exit "$rc"
}
trap record_exit EXIT

test ! -e "$run_root"
test -e "$RUNNER_SOURCE/.git"
test "$(git -C "$RUNNER_SOURCE" rev-parse HEAD)" = "$RUNNER_COMMIT"
test -z "$(git -C "$RUNNER_SOURCE" status --porcelain)"
test "$(sha256sum "$0" | awk '{print $1}')" = "$RUNNER_SHA256"
test -x "$python"

while read -r path expected; do
  test -f "$path"
  test "$(sha256sum "$path" | awk '{print $1}')" = "$expected"
done <<EOF
$source_run/FAILED $expected_source_failed_sha
$source_run/FULL_ABF_CONTRACT_DERIVATION.txt $expected_derivation_sha
$source_run/input-views/candidate/qsgw_input.full.contract $expected_full_contract_sha
$source_run/input-views/candidate/qsgw_input_full.summary.json $expected_full_summary_sha
$candidate/qsgw_matrices.dat $expected_candidate_matrix_sha
$candidate/qsgw_eigenvalues.dat $expected_candidate_eigenvalue_sha
$candidate/qsgw_iterations.dat $expected_candidate_iteration_sha
$candidate/librpa.stdout $expected_candidate_stdout_sha
$candidate/librpa.stderr $expected_candidate_stderr_sha
$candidate/librpa.in $expected_candidate_input_sha
$candidate/runtime.txt $expected_candidate_runtime_sha
$source_mode/PARAMETERS.txt $expected_parameters_sha
$legacy/qsgw_oracle_matrices.dat $expected_legacy_matrix_sha
$legacy/homo_lumo_vs_iterations.dat $expected_legacy_history_sha
EOF

while read -r path expected; do
  test -f "$path"
  test "$(sha256sum "$path" | awk '{print $1}')" = "$expected"
done <<EOF
$base_comparator c3daf072f222083a7ebdb9cf45f154d4bef64474f76db05992479a66fe30ebbc
$closure 4a5de94e6dbf590dded4a6ecd140aa4227fa61ffa0e73af17ec0f388610cffaf
$fixed 569ecb1366bc6dd4584216bd42ac55905a4848b1a0b1bdf96c6236c782de7cb2
$initial 6bbade9eaeb207b6cea9fa2f80d8cbcd0baeb8cbd760a2ffab5a0fe6f6d4868a
$adapter 8ffa4a125f3b95b25b2f64abd50f9ad119d52c4c8f89a8233e2208670d15541a
$summary a156d2a46d67bab73d22c91609cbe66cb3dc6e991e42bdd7407bec5a17554698
$parser f1e2b6f19250b0ff8b18785d3d29072f5f423fb4fdc2ae2b35381900f1282dbb
EOF

grep -Fq 'libRPA finished successfully' "$candidate/librpa.stdout"
test ! -s "$candidate/librpa.stderr"
grep -Fqx 'use_shrink_abfs = false' "$candidate/librpa.in"
grep -Fqx 'qsgw_input_contract = qsgw_input.full.contract' "$candidate/librpa.in"
grep -Fqx "# qsgw_input_contract_sha256 $expected_full_contract_sha" \
  "$candidate/qsgw_iterations.dat"
test "$(awk 'NF && $1 !~ /^#/ {last=$1} END {print last}' \
  "$candidate/qsgw_iterations.dat")" = 1
test "$(awk 'NF && $1 !~ /^#/ {last=$1} END {print last}' \
  "$legacy/homo_lumo_vs_iterations.dat")" = 1

mkdir -p "$run_root/tools" "$run_root/source"
cp "$0" "$run_root/"
cp "$base_comparator" "$run_root/tools/base_comparator.py"
cp "$base_comparator" "$run_root/tools/compare_qsgw_component_traces.py"
cp "$closure" "$run_root/tools/validate_qsgw_trace_closure.py"
cp "$fixed" "$run_root/tools/validate_qsgw_fixed_basis.py"
cp "$initial" "$run_root/tools/validate_qsgw_initial_state.py"
cp "$adapter" "$run_root/tools/compare_qsgw_legacy_v4_current_v6.py"
cp "$summary" "$run_root/tools/summarize_qsgw_trace_components_v1.py"
cp "$parser" "$run_root/tools/cmp_qsgw_v6.py"
cp "$source_run/FAILED" "$run_root/source/"
cp "$source_run/FULL_ABF_CONTRACT_DERIVATION.txt" "$run_root/source/"
cp "$source_run/input-views/candidate/qsgw_input.full.contract" "$run_root/source/"
cp "$source_run/input-views/candidate/qsgw_input_full.summary.json" "$run_root/source/"
cp "$candidate/qsgw_iterations.dat" "$run_root/source/"
cp "$candidate/librpa.in" "$run_root/source/"
cp "$candidate/runtime.txt" "$run_root/source/"
cp "$candidate/librpa.stdout" "$run_root/source/"
cp "$candidate/librpa.stderr" "$run_root/source/"
cp "$legacy/homo_lumo_vs_iterations.dat" "$run_root/source/legacy-homo_lumo_vs_iterations.dat"
printf '%s\n' "$RUNNER_SHA256" >"$run_root/runner-sha256.txt"

set +e
"$python" -B "$run_root/tools/compare_qsgw_legacy_v4_current_v6.py" \
  "$legacy/qsgw_oracle_matrices.dat" \
  "$candidate/qsgw_matrices.dat" \
  "$candidate/qsgw_eigenvalues.dat" \
  "$candidate/qsgw_iterations.dat" \
  "$run_root/legacy-current-comparison.json" \
  --base-comparator "$run_root/tools/base_comparator.py" \
  --current-contract-parser "$run_root/tools/cmp_qsgw_v6.py" \
  --iterations 0:1 --expected-mode none \
  --expected-legacy-beta 1 --expected-current-beta 0.2 \
  --expected-legacy-use-shrink-abfs 0 \
  --frequency-tolerance 1e-10 --matrix-max-abs-tolerance-ha 1e-8 \
  --matrix-relative-tolerance 1e-8 --eigenvalue-tolerance-ha 1e-6 \
  --gap-tolerance-ev 1e-5 --degeneracy-tolerance-ha 1e-8 \
  --state-tolerance 1e-10 \
  --normalized-current-matrix "$run_root/current-v5-self-matrices.dat" \
  --normalized-current-eigenvalues "$run_root/current-v5-self-eigenvalues.dat" \
  --normalized-current-iterations "$run_root/current-v5-self-iterations.dat" \
  >"$run_root/legacy-current-comparison.stdout" \
  2>"$run_root/legacy-current-comparison.stderr"
comparison_rc=$?
set -e
test "$comparison_rc" -eq 2
grep -Fq '"passed": false' "$run_root/legacy-current-comparison.json"
printf '%s\n' "$comparison_rc" >"$run_root/comparison-exit-code.txt"

"$python" -B "$run_root/tools/summarize_qsgw_trace_components_v1.py" \
  "$legacy/qsgw_oracle_matrices.dat" "$run_root/legacy-component-summary.json" \
  --iterations 0:1 >"$run_root/legacy-component-summary.stdout"
"$python" -B "$run_root/tools/summarize_qsgw_trace_components_v1.py" \
  "$candidate/qsgw_matrices.dat" "$run_root/current-component-summary.json" \
  --iterations 0:1 >"$run_root/current-component-summary.stdout"

PYTHONPATH="$run_root/tools" "$python" -B \
  "$run_root/tools/validate_qsgw_trace_closure.py" \
  "$legacy/qsgw_oracle_matrices.dat" "$run_root/legacy-closure.json" \
  --iterations 0:1 --channel 0 --legacy-contract \
  --closure-tolerance-ha 1e-10 --hermiticity-tolerance-ha 1e-10
grep -Fq '"passed": true' "$run_root/legacy-closure.json"
PYTHONPATH="$run_root/tools" "$python" -B \
  "$run_root/tools/validate_qsgw_trace_closure.py" \
  "$run_root/current-v5-self-matrices.dat" "$run_root/current-closure.json" \
  --iterations 0:1 --channel 0 \
  --closure-tolerance-ha 1e-10 --hermiticity-tolerance-ha 1e-10
grep -Fq '"passed": true' "$run_root/current-closure.json"

PYTHONPATH="$run_root/tools" "$python" -B \
  "$run_root/tools/validate_qsgw_fixed_basis.py" \
  "$run_root/current-v5-self-matrices.dat" \
  "$run_root/current-v5-self-eigenvalues.dat" \
  "$source_run/input-views/candidate/band_out" "$run_root/current-fixed-basis.json" \
  --iterations 0:1 --channel 0 \
  --eigenvalue-tolerance-ha 1e-10 --invariant-tolerance 1e-10
grep -Fq '"passed": true' "$run_root/current-fixed-basis.json"
PYTHONPATH="$run_root/tools" "$python" -B \
  "$run_root/tools/validate_qsgw_initial_state.py" \
  "$candidate/qsgw_matrices.dat" "$candidate/qsgw_iterations.dat" \
  "$source_run/input-views/candidate/band_out" "$run_root/current-initial-state.json" \
  --efermi-tolerance-ha 1e-12 --occupation-tolerance 1e-12
grep -Fq '"passed": true' "$run_root/current-initial-state.json"

awk '
  NF && $1 !~ /^#/ {
    printf "iter=%s max_delta_eV=%s residual_l2_Ha=%s residual_max_Ha=%s efermi_eV=%s gap_eV=%s electron_count=%s\n", $1, $2, $3, $4, $5, $6, $7
    if ($1 == 1) {
      if (($2 + 0.0) <= 1.0e5 || ($5 + 0.0) >= -100.0 || ($6 + 0.0) != 0.0) exit 2
      seen = 1
    }
  }
  END { if (!seen) exit 3 }
' "$candidate/qsgw_iterations.dat" >"$run_root/current-instability.txt"

awk '
  NF && $1 !~ /^#/ {
    printf "iter=%s homo_eV=%s lumo_eV=%s efermi_eV=%s\n", $1, $2, $3, $4
    if ($1 == 1) {
      if (($2 + 0.0) >= -1000.0 || ($3 + 0.0) >= -1000.0) exit 2
      seen = 1
    }
  }
  END { if (!seen) exit 3 }
' "$legacy/homo_lumo_vs_iterations.dat" >"$run_root/legacy-instability.txt"

awk -v tolerance=1e-10 '
  NF && $1 !~ /^#/ {
    value = $7 + 0.0
    if (!seen) { reference = value; seen = 1 }
    delta = value - reference
    if (delta < 0.0) delta = -delta
    if (delta > max_abs) max_abs = delta
    if (delta > tolerance) failed = 1
    count += 1
  }
  END {
    printf "reference_electron_count=%.17g\nmax_abs_delta=%.17g\niteration_count=%d\n", reference, max_abs, count
    if (!seen || failed) exit 1
  }
' "$candidate/qsgw_iterations.dat" >"$run_root/current-electron-count.txt"

cat >"$run_root/PROVENANCE.txt" <<EOF
gate=gate_a_shrink_off_candidate_failed_postcheck_v1
acceptance=false_diagnostic_only
runner_commit=$RUNNER_COMMIT
runner_sha256=$RUNNER_SHA256
source_failed_run=$source_run
source_candidate_matrix_sha256=$expected_candidate_matrix_sha
source_candidate_eigenvalue_sha256=$expected_candidate_eigenvalue_sha
source_candidate_iteration_sha256=$expected_candidate_iteration_sha
source_legacy_matrix_sha256=$expected_legacy_matrix_sha
full_abf_contract_sha256=$expected_full_contract_sha
comparison_exit_code=$comparison_rc
legacy_current_parity=false
legacy_full_abf_instability=true
candidate_full_abf_instability=true
candidate_numerical_execution=completed_successfully
runner_failure_reason=obsolete_homo_lumo_file_expectation
symmetry=on
use_shrink_abfs=false
headwing=off
hartree=off
band=off
iterations=0:1
EOF

(
  cd "$run_root"
  find . -type f ! -name OUTPUT_SHA256SUMS.txt \
    ! -name DIAGNOSTIC_COMPLETE ! -name FAILED -print0 | \
    LC_ALL=C sort -z | xargs -0 sha256sum >OUTPUT_SHA256SUMS.txt
  sha256sum --check --quiet OUTPUT_SHA256SUMS.txt
)
touch "$run_root/DIAGNOSTIC_COMPLETE"
cat "$run_root/PROVENANCE.txt"
cat "$run_root/legacy-instability.txt"
cat "$run_root/current-instability.txt"
run_succeeded=1
trap - EXIT
