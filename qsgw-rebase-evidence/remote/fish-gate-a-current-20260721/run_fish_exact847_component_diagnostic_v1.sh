#!/usr/bin/env bash
set -euo pipefail

: "${RUNNER_COMMIT:?RUNNER_COMMIT must identify the clean runner checkout}"
: "${RUNNER_SOURCE:?RUNNER_SOURCE must identify the clean runner checkout}"
: "${RUNNER_SHA256:?RUNNER_SHA256 must identify this exact runner}"
: "${RUN_ID:?RUN_ID must name a fresh immutable diagnostic directory}"

base=/home/bhj/ai-runs
run_root=$base/$RUN_ID
legacy_run=$base/librpa-qsgw-gate-a1-exact847-fullcoul-nohead-shrinkchi-off-miniter2-20260722-v3
legacy=$legacy_run/legacy
legacy_checkpoint=$legacy/librpa.d/qsgw_checkpoints
candidate_run=$base/librpa-qsgw-gate-a1-exact847-candidate-one-update-20260722-69c33c2f-v3
candidate=$candidate_run/candidate
python=$base/librpa-qsgw-gate0-20260715T1731-7d69a18c/venv/bin/python

expected_checkpoint_matrix_shas=(
  0e28d870136a05e67f012b22b31ddac6af31e4ee6ffef3cddc2dc65eabc92a03
  2c35d317cdff6a3f189d1da8177616fa7cca09731d71cb2d74a0fc06f1cf3cca
  0836c9dd0e82dc2235c3d2134ed678f7483d3534c2455f83547171454539603a
  525a185516d4d9965b00137de39465d90b38ab01893bfc45c834966e8ffb3cf2
  9c6c58dc7d707f87676783f4356998c54c1a893ede6adb21defc3a4aa2822fe8
  f6ab1d381cba4a0236c4a8ec4ff7b2a8b5ac5ee833128d6f49ae5cac63629804
  acf93703141588102afa9b18b5e31bd29e726743030034e89ac120e50070911a
  70b9efe5e82eb3cb44cf0e830225e43367291c496eeea9dcf83b2f205930f37b
)

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
test -e "$legacy_run/FAILED"
test ! -e "$legacy_run/COMPLETE"
test -e "$candidate_run/FAILED"
test ! -e "$candidate_run/COMPLETE"
test "$(sha256sum "$legacy/librpa.stdout" | awk '{print $1}')" = \
  e7642746b53937792a974e791f5e6ca93c3e63b48584709126022ffd9e7184ac
test "$(sha256sum "$candidate/qsgw_matrices.dat" | awk '{print $1}')" = \
  e8f96767491dca9ac2f379598142cbf093b132c2faf9dcf22f3e97c6f271de1e
test "$(cat "$legacy_checkpoint/latest_iteration.txt")" = 1
for ik in $(seq 1 8); do
  printf -v name 'H0_GW_spin_01_k_%06d.bin' "$ik"
  test "$(sha256sum "$legacy_checkpoint/iter_00001/$name" | awk '{print $1}')" = \
    "${expected_checkpoint_matrix_shas[$((ik - 1))]}"
done

symmetry_dir=$RUNNER_SOURCE/qsgw-rebase-evidence/remote/fish-gate-a-symmetry-20260720
current_dir=$RUNNER_SOURCE/qsgw-rebase-evidence/remote/fish-gate-a-current-20260721
tools_dir=$run_root/tools
mkdir -p "$tools_dir"
cp "$0" "$run_root/"
cp "$current_dir/diagnose_exact847_component_parity_v1.py" "$tools_dir/"
cp "$current_dir/compare_legacy_h0_candidate_trace_v2.py" "$tools_dir/"
cp "$symmetry_dir/compare_legacy_h0_candidate_trace_v1.py" "$tools_dir/"
cp "$symmetry_dir/compare_legacy_band0_native_outputs_v1.py" "$tools_dir/"
test "$(sha256sum "$tools_dir/diagnose_exact847_component_parity_v1.py" | awk '{print $1}')" = \
  e943471dad91b02c5a269a28aa80c2b4142e4535d2050b4540f9f8a8c0f269eb

PYTHONPATH="$tools_dir" "$python" -B \
  "$tools_dir/diagnose_exact847_component_parity_v1.py" \
  "$legacy_checkpoint" "$legacy/librpa.stdout" \
  "$candidate/qsgw_matrices.dat" "$run_root/component-diagnostic.json" \
  --iteration 1 --n-spins 1 --n-kpoints 8 --n-bands 44 \
  >"$run_root/diagnostic.stdout" 2>"$run_root/diagnostic.stderr"
test ! -s "$run_root/diagnostic.stderr"
grep -Fq '"diagnostic_complete": true' "$run_root/component-diagnostic.json"

cat >"$run_root/PROVENANCE.txt" <<EOF
gate=gate_a1_exact847_component_diagnostic_v1
acceptance=false_diagnostic_only
runner_commit=$RUNNER_COMMIT
runner_sha256=$RUNNER_SHA256
legacy_run=$legacy_run
legacy_run_is_green=false
legacy_checkpoint_iteration=1
legacy_stdout_sha256=e7642746b53937792a974e791f5e6ca93c3e63b48584709126022ffd9e7184ac
candidate_run=$candidate_run
candidate_run_is_green=false
candidate_matrix_sha256=e8f96767491dca9ac2f379598142cbf093b132c2faf9dcf22f3e97c6f271de1e
symmetry=on
headwing=off
hartree=off
iteration=1
completed_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
EOF

(
  cd "$run_root"
  find . -type f ! -name OUTPUT_SHA256SUMS.txt \
    ! -name DIAGNOSTIC_COMPLETE ! -name FAILED -print0 \
    | LC_ALL=C sort -z | xargs -0 sha256sum >OUTPUT_SHA256SUMS.txt
  sha256sum --check --quiet OUTPUT_SHA256SUMS.txt
)
touch "$run_root/DIAGNOSTIC_COMPLETE"
cat "$run_root/PROVENANCE.txt"
cat "$run_root/component-diagnostic.json"
run_succeeded=1
trap - EXIT
