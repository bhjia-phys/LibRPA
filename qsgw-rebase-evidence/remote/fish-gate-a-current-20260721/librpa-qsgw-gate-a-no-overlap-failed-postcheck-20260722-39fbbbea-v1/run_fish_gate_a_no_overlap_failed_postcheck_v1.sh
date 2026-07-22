#!/usr/bin/env bash
set -euo pipefail

: "${RUNNER_COMMIT:?RUNNER_COMMIT must identify the clean checkout}"
: "${RUNNER_SHA256:?RUNNER_SHA256 must identify this exact runner}"
: "${RUN_TAG:?RUN_TAG must make the diagnostic archive immutable}"

source_root=$(cd "$(dirname "$0")/../../.." && pwd)
run_root=/home/bhj/ai-runs/librpa-qsgw-gate-a-no-overlap-failed-postcheck-$RUN_TAG
failed_run=/home/bhj/ai-runs/librpa-qsgw-gate-a-legacy-no-overlap-iter1-20260722-00517ab0-v1
alias_postcheck=/home/bhj/ai-runs/librpa-qsgw-gate-a-failed-prefix-postcheck-20260722-1dd5dbb5-v1
alias_trace=/home/bhj/ai-runs/librpa-qsgw-gate-a-current-20260722-1b5387e2-v1/no-mix-miniter2/legacy/qsgw_oracle_matrices.dat
no_overlap_trace=$failed_run/legacy/qsgw_oracle_matrices.dat
tool_source=$source_root/qsgw-rebase-evidence/remote/fish-gate-a-current-20260721
base_comparator=$source_root/qsgw-rebase-evidence/remote/fish-gate-a-symmetry-20260720/compare_qsgw_component_traces-v4-c3daf072.py
python=${PYTHON:-$(command -v python3)}

expected_failed_sha=b57edd98b17ac66a73da27a0d5b604e695c3455e88c2e7cb0851d0f50221cec9
expected_trace_sha=c3a8907e397c010be535d5d3a8526470c6353fa390f3f70ee2149ae7eb5fbeda
expected_history_sha=ee7b4b46925db832859e5a591f05b1559f979bb5540c8a22ec893cd898a5a9ee
expected_stdout_sha=cc144de247642c56361a77b65eecb4391cd5a3b3394cfa7521d48f302930d7df
expected_stderr_sha=f94f14e7ed7c74d4f50737a9d61f7ffd4b128137832d745b1813793a4708ed34
expected_summary_sha=578047145a68ef3db361004cc34507709f9453bbcc17a476eaa173dfd229164f
expected_current_comparison_sha=067f407c3e1ad62de1f7898e44b9bd4e59266a2c9d14d58af94a64a530eb0372
expected_alias_trace_sha=99eabc038a2d3f566267a5a80a2fb7101011bea256eb2dcc90e05d443d7b69be
expected_alias_postcheck_provenance_sha=5156999a36a5ca3282579238a2e25146691a5bf95ce247c6f9b4d6f6dde24519
expected_base_comparator_sha=c3daf072f222083a7ebdb9cf45f154d4bef64474f76db05992479a66fe30ebbc
expected_source_runner_commit=00517ab0c1ed11e4f592732386df73c77d597f24
expected_source_runner_sha=df5f6d30083d6ef35a563db289cd0f9cb428dfec2a6c508a1b708721d3a06f90

test ! -e "$run_root"
test -e "$source_root/.git"
test "$(git -C "$source_root" rev-parse HEAD)" = "$RUNNER_COMMIT"
git -C "$source_root" diff --exit-code
git -C "$source_root" diff --cached --exit-code
test "$(sha256sum "$0" | awk '{print $1}')" = "$RUNNER_SHA256"
test "$(sha256sum "$failed_run/FAILED" | awk '{print $1}')" = "$expected_failed_sha"
test "$(cat "$failed_run/FAILED")" = $'failed_utc=2026-07-22T01:12:17Z\nexit_code=2'
test "$(sha256sum "$no_overlap_trace" | awk '{print $1}')" = "$expected_trace_sha"
test "$(sha256sum "$failed_run/legacy/homo_lumo_vs_iterations.dat" | awk '{print $1}')" = "$expected_history_sha"
test "$(sha256sum "$failed_run/legacy/librpa.stdout" | awk '{print $1}')" = "$expected_stdout_sha"
test "$(sha256sum "$failed_run/legacy/librpa.stderr" | awk '{print $1}')" = "$expected_stderr_sha"
test "$(sha256sum "$failed_run/legacy-component-summary.json" | awk '{print $1}')" = "$expected_summary_sha"
test "$(sha256sum "$failed_run/legacy-current-comparison.json" | awk '{print $1}')" = "$expected_current_comparison_sha"
test "$(sha256sum "$alias_trace" | awk '{print $1}')" = "$expected_alias_trace_sha"
test "$(sha256sum "$alias_postcheck/PROVENANCE.txt" | awk '{print $1}')" = "$expected_alias_postcheck_provenance_sha"
test "$(sha256sum "$base_comparator" | awk '{print $1}')" = "$expected_base_comparator_sha"
test "$(cat "$failed_run/runner-sha256.txt")" = "$expected_source_runner_sha"
test "$(git -C "$source_root" show \
  "$expected_source_runner_commit:qsgw-rebase-evidence/remote/fish-gate-a-current-20260721/run_fish_gate_a_legacy_no_overlap_iter1_v1.sh" \
  | sha256sum | awk '{print $1}')" = "$expected_source_runner_sha"
grep -Fq 'libRPA finished successfully' "$failed_run/legacy/librpa.stdout"
test "$(grep -c '^S matrix file not found:' "$failed_run/legacy/librpa.stderr")" -eq 8
test "$(grep -c '^HF file not found:' "$failed_run/legacy/librpa.stderr")" -eq 8
test -z "$(grep -Ev '^(S matrix file not found:|HF file not found:)' \
  "$failed_run/legacy/librpa.stderr")"
test -f "$tool_source/validate_gate_a_overlap_diagnostic_v1.py"
test -f "$tool_source/test_validate_gate_a_overlap_diagnostic_v1.py"
test -x "$python"

mkdir -p "$run_root/tools" "$run_root/source-no-overlap" "$run_root/source-alias"
cp "$0" "$run_root/"
cp "$base_comparator" "$run_root/tools/base_comparator.py"
cp "$tool_source/validate_gate_a_overlap_diagnostic_v1.py" "$run_root/tools/"
cp "$tool_source/test_validate_gate_a_overlap_diagnostic_v1.py" "$run_root/tools/"
cp "$failed_run/FAILED" "$run_root/source-no-overlap/"
cp "$failed_run/legacy/homo_lumo_vs_iterations.dat" \
  "$run_root/source-no-overlap/"
cp "$failed_run/legacy/librpa.stdout" "$run_root/source-no-overlap/"
cp "$failed_run/legacy/librpa.stderr" "$run_root/source-no-overlap/"
cp "$failed_run/legacy-component-summary.json" "$run_root/source-no-overlap/"
cp "$failed_run/legacy-current-comparison.json" "$run_root/source-no-overlap/"
cp "$failed_run/runner-sha256.txt" "$run_root/source-no-overlap/"
cp "$alias_postcheck/PROVENANCE.txt" "$run_root/source-alias/"
cp "$alias_postcheck/legacy-component-summary.json" "$run_root/source-alias/"
cp "$alias_postcheck/legacy-current-prefix-comparison.json" "$run_root/source-alias/"
printf '%s\n' "$RUNNER_SHA256" >"$run_root/runner-sha256.txt"

(
  cd "$run_root/tools"
  "$python" -B test_validate_gate_a_overlap_diagnostic_v1.py
) >"$run_root/unit-tests.stdout" 2>"$run_root/unit-tests.stderr"

set +e
"$python" -B "$run_root/tools/base_comparator.py" \
  "$no_overlap_trace" "$alias_trace" \
  "$run_root/alias-vs-no-overlap-prefix.json" \
  --iterations 0:1 --allow-iteration-prefix --contract-mode oracle \
  --matrix-max-abs-tolerance-ha 1e-8 \
  --matrix-relative-tolerance 1e-8 \
  --eigenvalue-tolerance 1e-6 --gap-tolerance-ev 1e-5 \
  --state-tolerance 1e-10 \
  >"$run_root/alias-vs-no-overlap-prefix.stdout" \
  2>"$run_root/alias-vs-no-overlap-prefix.stderr"
comparison_exit_code=$?
set -e
test "$comparison_exit_code" -eq 2
printf '%s\n' "$comparison_exit_code" >"$run_root/comparison-exit-code.txt"

"$python" -B "$run_root/tools/validate_gate_a_overlap_diagnostic_v1.py" \
  "$run_root/alias-vs-no-overlap-prefix.json" \
  "$run_root/overlap-diagnostic-validation.json" \
  --tolerance-ha 2e-9 \
  >"$run_root/overlap-diagnostic-validation.stdout" \
  2>"$run_root/overlap-diagnostic-validation.stderr"
grep -Fq '"passed": true' "$run_root/overlap-diagnostic-validation.json"

cat >"$run_root/PROVENANCE.txt" <<EOF
gate=gate_a_no_overlap_failed_postcheck_v1
acceptance=false_diagnostic_only
runner_commit=$RUNNER_COMMIT
runner_sha256=$RUNNER_SHA256
source_failed_run=$failed_run
source_failure_exit_code=2
source_legacy_trace_sha256=$expected_trace_sha
alias_legacy_trace_sha256=$expected_alias_trace_sha
selected_iterations=0:1
comparison_exit_code=$comparison_exit_code
numeric_equivalence_tolerance_ha=2e-9
overlap_conclusion=alias_does_not_explain_rejected_symmetry_prefix
next_controlled_factor=use_shrink_abfs_true_vs_false
symmetry=on
headwing=off
hartree=off
band=off
EOF

(
  cd "$run_root"
  find . -type f ! -name OUTPUT_SHA256SUMS.txt \
    ! -name DIAGNOSTIC_COMPLETE -print0 | LC_ALL=C sort -z | \
    xargs -0 sha256sum >OUTPUT_SHA256SUMS.txt
  sha256sum --check --quiet OUTPUT_SHA256SUMS.txt
)
touch "$run_root/DIAGNOSTIC_COMPLETE"
cat "$run_root/PROVENANCE.txt"
cat "$run_root/overlap-diagnostic-validation.json"
