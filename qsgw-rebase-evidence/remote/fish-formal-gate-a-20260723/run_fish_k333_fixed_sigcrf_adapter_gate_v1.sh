#!/usr/bin/env bash
set -euo pipefail

run=/home/bhj/ai-runs/qsgw-regression-aims-si-k333-headonly-fixed-sigcrf-gate-20260723-v1
legacy_run=/home/bhj/ai-runs/qsgw-regression-aims-si-k333-headonly-old-sigcrf-20260723-v2
current_run=/home/bhj/ai-runs/qsgw-regression-aims-si-k333-headonly-current-read-old-sigcrf-20260723-v1
tool_source=/tmp/qsgw-k333-fixed-sigcrf-compare-v1
legacy_trace=$legacy_run/work/sigcrf/qsgw_oracle_matrices.dat
current_matrix=$current_run/work/librpa.d/qsgw_matrices.dat
current_eigenvalues=$current_run/work/librpa.d/qsgw_eigenvalues.dat
current_iterations=$current_run/work/librpa.d/qsgw_iterations.dat

test ! -e "$run"
test -f "$legacy_run/COMPLETE"
test -f "$current_run/COMPLETE"
test "$(sha256sum "$legacy_trace" | awk '{print $1}')" = \
  5df7d8987f855933b96385faa9175f29e8fccb1da0f94da7e3f87c01affbc5ba
test "$(find "$legacy_run/work/sigcrf" -maxdepth 1 -type f -name 'SigcRF*' | wc -l)" -eq 6
test -s "$current_matrix"
test -s "$current_eigenvalues"
test -s "$current_iterations"

mkdir -p "$run/tools"
complete=0
record_exit() {
  local rc=$?
  trap - EXIT
  if [[ $complete -ne 1 ]]; then
    printf 'exit_code=%s\nfailed_utc=%s\n' \
      "$rc" "$(date --iso-8601=seconds)" >"$run/FAILED"
  fi
  exit "$rc"
}
trap record_exit EXIT

cp "$tool_source/compare_qsgw_legacy_v4_current_v6.py" "$run/tools/"
cp "$tool_source/compare_qsgw_component_traces-v4-c3daf072.py" \
  "$run/tools/base_comparator.py"
cp "$tool_source/cmp_qsgw.py" "$run/tools/"

cat >"$run/PROVENANCE.txt" <<EOF
run=$run
purpose=legacy_current_qsgw_adapter_gate_with_identical_sigcrf
legacy_run=$legacy_run
legacy_trace=$legacy_trace
legacy_trace_sha256=$(sha256sum "$legacy_trace" | awk '{print $1}')
legacy_sigcrf_checksums=$legacy_run/SIGCRF_SHA256SUMS.txt
current_run=$current_run
current_matrix=$current_matrix
current_matrix_sha256=$(sha256sum "$current_matrix" | awk '{print $1}')
current_eigenvalues_sha256=$(sha256sum "$current_eigenvalues" | awk '{print $1}')
current_iterations_sha256=$(sha256sum "$current_iterations" | awk '{print $1}')
matrix_max_abs_tolerance_ha=1e-8
matrix_relative_tolerance=1e-8
eigenvalue_tolerance_ha=1e-6
gap_tolerance_ev=1e-5
degeneracy_tolerance_ha=1e-5
state_gauge_tolerance=2e-6
current_ha2ev=27.2113845
hartree=off
head=on
wing=off
EOF

python3 "$run/tools/compare_qsgw_legacy_v4_current_v6.py" \
  "$legacy_trace" \
  "$current_matrix" \
  "$current_eigenvalues" \
  "$current_iterations" \
  "$run/legacy-current-fixed-sigcrf-comparison.json" \
  --base-comparator "$run/tools/base_comparator.py" \
  --current-contract-parser "$run/tools/cmp_qsgw.py" \
  --iterations 0:1 \
  --expected-mode none \
  --expected-legacy-beta 1 \
  --expected-current-beta 0.2 \
  --expected-legacy-symmetry on \
  --expected-current-symmetry off \
  --expected-legacy-head on \
  --expected-current-head on \
  --expected-legacy-use-shrink-abfs 0 \
  --frequency-tolerance 1e-10 \
  --matrix-max-abs-tolerance-ha 1e-8 \
  --matrix-relative-tolerance 1e-8 \
  --eigenvalue-tolerance-ha 1e-6 \
  --gap-tolerance-ev 1e-5 \
  --degeneracy-tolerance-ha 1e-5 \
  --state-tolerance 2e-6 \
  --current-ha2ev 27.2113845 \
  >"$run/comparison.stdout" 2>"$run/comparison.stderr"

test ! -s "$run/comparison.stderr"
grep -Fq '"passed": true' \
  "$run/legacy-current-fixed-sigcrf-comparison.json"
cat >>"$run/PROVENANCE.txt" <<EOF
comparison_sha256=$(sha256sum "$run/legacy-current-fixed-sigcrf-comparison.json" | awk '{print $1}')
completed_utc=$(date --iso-8601=seconds)
EOF

(
  cd "$run"
  find . -type f ! -name OUTPUT_SHA256SUMS.txt \
    ! -name GREEN ! -name FAILED -print0 \
    | LC_ALL=C sort -z | xargs -0 sha256sum >OUTPUT_SHA256SUMS.txt
  sha256sum --check --quiet OUTPUT_SHA256SUMS.txt
)
touch "$run/GREEN"
complete=1
trap - EXIT
