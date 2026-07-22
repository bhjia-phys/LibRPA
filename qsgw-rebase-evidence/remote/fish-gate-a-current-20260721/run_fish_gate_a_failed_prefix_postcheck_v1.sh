#!/usr/bin/env bash
set -euo pipefail

: "${RUNNER_COMMIT:?RUNNER_COMMIT must identify the clean checkout}"
: "${RUNNER_SHA256:?RUNNER_SHA256 must identify this exact runner}"
: "${RUN_TAG:?RUN_TAG must make the diagnostic archive immutable}"

source_root=$(cd "$(dirname "$0")/../../.." && pwd)
run_root=/home/bhj/ai-runs/librpa-qsgw-gate-a-failed-prefix-postcheck-$RUN_TAG
failed_run=/home/bhj/ai-runs/librpa-qsgw-gate-a-current-20260722-1b5387e2-v1
legacy_trace=$failed_run/no-mix-miniter2/legacy/qsgw_oracle_matrices.dat
current_run=/home/bhj/ai-runs/librpa-qsgw-gate2-current-20260722-aca53374-v1/candidate-qsgw
gate2_postcheck=/home/bhj/ai-runs/librpa-qsgw-gate2-current-postcheck-20260722-2ad6b353-v1
tool_source=$source_root/qsgw-rebase-evidence/remote/fish-gate-a-current-20260721
base_comparator=$source_root/qsgw-rebase-evidence/remote/fish-gate-a-symmetry-20260720/compare_qsgw_component_traces-v4-c3daf072.py
current_parser=$source_root/regression_tests/backend/comparisons/cmp_qsgw.py
python=${PYTHON:-python3}

expected_legacy_trace_sha=99eabc038a2d3f566267a5a80a2fb7101011bea256eb2dcc90e05d443d7b69be
expected_current_matrix_sha=85a8e04bd29780630a6720832898e06e18953810c2b183f87e5be4a9af93af37
expected_current_eigenvalue_sha=1ee2b4e9afe21987a95bfc3315e9945b0dd9dfa69d8f41e1734f4e4607e009f8
expected_current_iteration_sha=e7ae3485578958d1ead4bead993e27b508bbe32f7ff7ab688f1b82d5d7837ef9
expected_failed_sha=400d0dde45f3156aa4a27e635e276d0551e05e7e3e7fdcb738a92fa8225e10c9
expected_gate2_postcheck_provenance_sha=79d50e4f5ea14b624ff3aebc2c43802c1ee01e3b97445401467048b57e8f0526

test ! -e "$run_root"
test -e "$source_root/.git"
test "$(git -C "$source_root" rev-parse HEAD)" = "$RUNNER_COMMIT"
git -C "$source_root" diff --exit-code
git -C "$source_root" diff --cached --exit-code
test -f "$legacy_trace"
test -f "$current_run/qsgw_matrices.dat"
test -f "$current_run/qsgw_eigenvalues.dat"
test -f "$current_run/qsgw_iterations.dat"
test -e "$gate2_postcheck/GREEN_CONFIRMED"
test ! -e "$gate2_postcheck/FAILED"
test "$(sha256sum "$legacy_trace" | awk '{print $1}')" = "$expected_legacy_trace_sha"
test "$(sha256sum "$current_run/qsgw_matrices.dat" | awk '{print $1}')" = "$expected_current_matrix_sha"
test "$(sha256sum "$current_run/qsgw_eigenvalues.dat" | awk '{print $1}')" = "$expected_current_eigenvalue_sha"
test "$(sha256sum "$current_run/qsgw_iterations.dat" | awk '{print $1}')" = "$expected_current_iteration_sha"
test "$(sha256sum "$failed_run/FAILED" | awk '{print $1}')" = "$expected_failed_sha"
test "$(sha256sum "$gate2_postcheck/PROVENANCE.txt" | awk '{print $1}')" = "$expected_gate2_postcheck_provenance_sha"
test "$(cat "$failed_run/FAILED")" = $'failed_utc=2026-07-22T00:17:51Z\nexit_code=143'
test -f "$base_comparator"
test -f "$current_parser"

mkdir -p "$run_root/tools"
cp "$tool_source/compare_qsgw_legacy_v4_current_v6.py" "$run_root/tools/"
cp "$tool_source/test_compare_qsgw_legacy_v4_current_v6.py" "$run_root/tools/"
cp "$tool_source/summarize_qsgw_trace_components_v1.py" "$run_root/tools/"
cp "$tool_source/test_summarize_qsgw_trace_components_v1.py" "$run_root/tools/"
cp -a "$tool_source/fixtures" "$run_root/tools/"
cp "$base_comparator" "$run_root/tools/base_comparator.py"
cp "$current_parser" "$run_root/tools/cmp_qsgw.py"
cp "$failed_run/FAILED" "$run_root/source-FAILED"
cp "$failed_run/no-mix-miniter2/legacy/homo_lumo_vs_iterations.dat" \
  "$run_root/legacy-homo-lumo.dat"
cp "$current_run/qsgw_iterations.dat" "$run_root/current-iterations.dat"
printf '%s\n' "$RUNNER_SHA256" >"$run_root/runner-sha256.txt"

(
  cd "$source_root"
  "$python" -B -m unittest \
    qsgw-rebase-evidence/remote/fish-gate-a-current-20260721/test_compare_qsgw_legacy_v4_current_v6.py \
    qsgw-rebase-evidence/remote/fish-gate-a-current-20260721/test_summarize_qsgw_trace_components_v1.py
) >"$run_root/unit-tests.stdout" 2>"$run_root/unit-tests.stderr"

"$python" -B "$run_root/tools/summarize_qsgw_trace_components_v1.py" \
  "$legacy_trace" "$run_root/legacy-component-summary.json" \
  --iterations 0:1 >"$run_root/legacy-component-summary.stdout" \
  2>"$run_root/legacy-component-summary.stderr"
"$python" -B "$run_root/tools/summarize_qsgw_trace_components_v1.py" \
  "$current_run/qsgw_matrices.dat" "$run_root/current-component-summary.json" \
  --iterations 0:1 >"$run_root/current-component-summary.stdout" \
  2>"$run_root/current-component-summary.stderr"

set +e
"$python" -B "$run_root/tools/compare_qsgw_legacy_v4_current_v6.py" \
  "$legacy_trace" \
  "$current_run/qsgw_matrices.dat" \
  "$current_run/qsgw_eigenvalues.dat" \
  "$current_run/qsgw_iterations.dat" \
  "$run_root/legacy-current-prefix-comparison.json" \
  --base-comparator "$run_root/tools/base_comparator.py" \
  --current-contract-parser "$run_root/tools/cmp_qsgw.py" \
  --iterations 0:1 \
  --expected-mode none \
  --expected-legacy-beta 1 \
  --expected-current-beta 0.2 \
  --allow-legacy-iteration-prefix \
  --frequency-tolerance 1e-10 \
  --matrix-max-abs-tolerance-ha 1e-8 \
  --matrix-relative-tolerance 1e-8 \
  --eigenvalue-tolerance-ha 1e-6 \
  --gap-tolerance-ev 1e-5 \
  --degeneracy-tolerance-ha 1e-8 \
  --state-tolerance 1e-10 \
  >"$run_root/legacy-current-prefix-comparison.stdout" \
  2>"$run_root/legacy-current-prefix-comparison.stderr"
comparison_exit_code=$?
set -e
printf '%s\n' "$comparison_exit_code" >"$run_root/comparison-exit-code.txt"

"$python" -B - "$run_root/legacy-component-summary.json" \
  "$run_root/current-component-summary.json" \
  "$run_root/component-maxima.tsv" <<'PY'
import json
import sys
from pathlib import Path

legacy = json.loads(Path(sys.argv[1]).read_text())
current = json.loads(Path(sys.argv[2]).read_text())
output = Path(sys.argv[3])

def rows(report):
    return {
        (row["iteration"], row["channel"], row["component"]): row
        for row in report["groups"]
    }

left = rows(legacy)
right = rows(current)
keys = sorted(set(left) | set(right))
lines = ["iteration\tchannel\tcomponent\tlegacy_max_abs\tcurrent_max_abs"]
for key in keys:
    lines.append(
        "\t".join(
            [
                str(key[0]),
                str(key[1]),
                key[2],
                str(left.get(key, {}).get("max_abs", "missing")),
                str(right.get(key, {}).get("max_abs", "missing")),
            ]
        )
    )
output.write_text("\n".join(lines) + "\n")
PY

cat >"$run_root/PROVENANCE.txt" <<EOF
gate=gate_a_failed_prefix_postcheck_v1
acceptance=false_diagnostic_only
runner_commit=$RUNNER_COMMIT
runner_sha256=$RUNNER_SHA256
source_failed_run=$failed_run
source_failure_exit_code=143
legacy_trace_sha256=$expected_legacy_trace_sha
current_gate2_run=$current_run
current_matrix_sha256=$expected_current_matrix_sha
current_eigenvalue_sha256=$expected_current_eigenvalue_sha
current_iteration_sha256=$expected_current_iteration_sha
selected_iterations=0:1
legacy_declared_iterations=0:2
comparison_exit_code=$comparison_exit_code
purpose=locate_first_component_divergence_without_rerunning_numerics
EOF

(
  cd "$run_root"
  find . -type f ! -name OUTPUT_SHA256SUMS.txt ! -name DIAGNOSTIC_COMPLETE \
    -print0 | LC_ALL=C sort -z | xargs -0 sha256sum >OUTPUT_SHA256SUMS.txt
  sha256sum --check --quiet OUTPUT_SHA256SUMS.txt
)
touch "$run_root/DIAGNOSTIC_COMPLETE"
cat "$run_root/PROVENANCE.txt"
cat "$run_root/component-maxima.tsv"
exit 0
