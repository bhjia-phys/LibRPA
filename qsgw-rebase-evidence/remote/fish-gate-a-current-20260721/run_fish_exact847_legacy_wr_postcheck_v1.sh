#!/usr/bin/env bash
set -euo pipefail

: "${RUNNER_COMMIT:?RUNNER_COMMIT must identify the clean runner checkout}"
: "${RUNNER_SOURCE:?RUNNER_SOURCE must identify the clean runner checkout}"
: "${RUNNER_SHA256:?RUNNER_SHA256 must identify this exact runner}"
: "${RUN_ID:?RUN_ID must name a fresh immutable postcheck directory}"

[[ "$RUNNER_COMMIT" =~ ^[0-9a-f]{40}$ ]]
[[ "$RUNNER_SHA256" =~ ^[0-9a-f]{64}$ ]]
case "$RUN_ID" in
  *[!A-Za-z0-9._-]*|'') echo "RUN_ID contains unsafe characters" >&2; exit 2 ;;
esac

base=/home/bhj/ai-runs
run_root=$base/$RUN_ID
source_run=$base/librpa-qsgw-gate-a1-exact847-legacy-wr-diagnostic-20260722-e5407096-v3
source_work=$source_run/diagnostic
accepted_current=$base/librpa-qsgw-gate-a1-exact847-current-sigcrf-compare-20260722-087f29fb-v1
accepted_current_work=$accepted_current/candidate
observer_run=$base/librpa-qsgw-gate-a1-exact847-component-observer-20260722-17720999-v1
component_dir=$observer_run/legacy/librpa.d/qsgw_legacy_components/iter_00001
current_dir=$RUNNER_SOURCE/qsgw-rebase-evidence/remote/fish-gate-a-current-20260721
symmetry_dir=$RUNNER_SOURCE/qsgw-rebase-evidence/remote/fish-gate-a-symmetry-20260720
tools_dir=$run_root/tools
python=$base/librpa-qsgw-gate0-20260715T1731-7d69a18c/venv/bin/python

expected_source_failed_sha=409a4c28da09eb9213a773f4da20062a810f5909a3e63907759cd8b8bd8684d6
expected_source_stdout_sha=94ec6e0b0b37f2e08559efe49916c16138d1ec3da812e3c4d157758d0b807772
expected_source_trace_sha=71cca2b54b24860e42228ecbce32a4215308120ad456d15210f3a58c11ae9cda
expected_source_sigcrf_json_sha=64af0a75a2631e90cb7977c16520013ecbdd6438c344de6ea67ae053324ea695
expected_source_sigcrf_tree_sha=58bb7917a440c048e2640aa39e2f1216507b2cef9f8bd474ac7cb23db0093ae1
expected_current_trace_sha=12ed429051692b63a46c13e481ea52811cefde9f9d236f0568d9ff7d10e1468e
expected_component_tree_sha=e1065cedb703c2eba9ee32aca3303815e0c539553cebabb1c3b333eb9275f4e3

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

test -e "$source_run/FAILED"
test ! -e "$source_run/DIAGNOSTIC_COMPLETE"
test "$(sha256sum "$source_run/FAILED" | awk '{print $1}')" = \
  "$expected_source_failed_sha"
test "$(sha256sum "$source_work/librpa.stdout" | awk '{print $1}')" = \
  "$expected_source_stdout_sha"
test "$(sha256sum "$source_work/qsgw_matrices.dat" | awk '{print $1}')" = \
  "$expected_source_trace_sha"
test "$(sha256sum "$source_run/exact847-legacy-wr-sigcrf-comparison.json" | awk '{print $1}')" = \
  "$expected_source_sigcrf_json_sha"
grep -Fq 'DIAGNOSTIC: GW symmetry accumulates irreducible-sector' \
  "$source_work/librpa.stdout"
grep -Fq 'QSGW completed iterations: 1' "$source_work/librpa.stdout"
grep -Fq 'libRPA finished successfully' "$source_work/librpa.stdout"
test -s "$source_work/qsgw_iterations.dat"
test "$(find "$source_work" -maxdepth 1 -type f -name 'SigcRF*' | wc -l)" -eq 16
source_sigcrf_tree_sha=$(
  cd "$source_work"
  find . -maxdepth 1 -type f -name 'SigcRF*' -print0 \
    | LC_ALL=C sort -z | xargs -0 sha256sum | sha256sum | awk '{print $1}'
)
test "$source_sigcrf_tree_sha" = "$expected_source_sigcrf_tree_sha"

test -e "$accepted_current/DIAGNOSTIC_COMPLETE"
test "$(sha256sum "$accepted_current_work/qsgw_matrices.dat" | awk '{print $1}')" = \
  "$expected_current_trace_sha"
test "$(find "$accepted_current_work" -maxdepth 1 -type f -name 'SigcRF*' | wc -l)" -eq 16
test -f "$component_dir/metadata.txt"
component_tree_sha=$(
  cd "$component_dir"
  find . -type f -print0 | LC_ALL=C sort -z | xargs -0 sha256sum | sha256sum \
    | awk '{print $1}'
)
test "$component_tree_sha" = "$expected_component_tree_sha"

mkdir -p "$run_root" "$tools_dir"
cp "$0" "$run_root/"
for tool in \
  compare_exact847_sigcrf_v1.py \
  compare_exact847_component_dump_v1.py \
  compare_legacy_h0_candidate_trace_v2.py \
  diagnose_exact847_component_parity_v1.py \
  classify_exact847_legacy_wr_diagnostic_v1.py; do
  cp "$current_dir/$tool" "$tools_dir/$tool"
done
for tool in \
  compare_legacy_band0_native_outputs_v1.py \
  compare_legacy_h0_candidate_trace_v1.py; do
  cp "$symmetry_dir/$tool" "$tools_dir/$tool"
done

PYTHONPATH="$tools_dir" "$python" -B \
  "$tools_dir/compare_exact847_component_dump_v1.py" \
  "$component_dir" "$source_work/qsgw_matrices.dat" \
  "$run_root/exact847-legacy-wr-component-comparison.json" \
  --iteration 1 --n-frequencies 16 --n-spins 1 --n-kpoints 8 --n-bands 44 \
  >"$run_root/component-comparison.stdout" \
  2>"$run_root/component-comparison.stderr"
test ! -s "$run_root/component-comparison.stderr"
grep -Fq '"diagnostic_complete": true' \
  "$run_root/exact847-legacy-wr-component-comparison.json"

"$python" -B "$tools_dir/compare_exact847_sigcrf_v1.py" \
  "$accepted_current_work" "$source_work" \
  "$run_root/current-vs-legacy-wr-sigcrf-comparison.json" \
  >"$run_root/current-vs-legacy-wr-sigcrf.stdout" \
  2>"$run_root/current-vs-legacy-wr-sigcrf.stderr"
test ! -s "$run_root/current-vs-legacy-wr-sigcrf.stderr"
grep -Fq '"diagnostic_complete": true' \
  "$run_root/current-vs-legacy-wr-sigcrf-comparison.json"

"$python" -B "$tools_dir/classify_exact847_legacy_wr_diagnostic_v1.py" \
  "$source_run/exact847-legacy-wr-sigcrf-comparison.json" \
  "$run_root/exact847-legacy-wr-component-comparison.json" \
  "$run_root/exact847-legacy-wr-classification.json" \
  >"$run_root/classification.stdout" \
  2>"$run_root/classification.stderr"
test ! -s "$run_root/classification.stderr"
grep -Fq '"diagnostic_complete": true' \
  "$run_root/exact847-legacy-wr-classification.json"

cat >"$run_root/PROVENANCE.txt" <<EOF
gate=gate_a1_exact847_legacy_wr_postcheck_v1
acceptance=false_diagnostic_only
runner_commit=$RUNNER_COMMIT
runner_sha256=$RUNNER_SHA256
source_run=$source_run
source_status=failed_postcheck_after_successful_librpa
source_trace_sha256=$expected_source_trace_sha
source_sigcrf_tree_sha256=$source_sigcrf_tree_sha
accepted_current_run=$accepted_current
legacy_component_observer_run=$observer_run
legacy_component_tree_sha256=$component_tree_sha
classification_sha256=$(sha256sum "$run_root/exact847-legacy-wr-classification.json" | awk '{print $1}')
completed_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
EOF

(
  cd "$run_root"
  find . -type f ! -name OUTPUT_SHA256SUMS.txt \
    -print0 | LC_ALL=C sort -z | xargs -0 sha256sum >OUTPUT_SHA256SUMS.txt
)
touch "$run_root/POSTCHECK_COMPLETE"
run_succeeded=1
