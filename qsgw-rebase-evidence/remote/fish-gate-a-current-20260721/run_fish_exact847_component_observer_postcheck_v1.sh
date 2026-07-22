#!/usr/bin/env bash
set -euo pipefail

: "${RUNNER_COMMIT:?RUNNER_COMMIT must identify the clean runner checkout}"
: "${RUNNER_SOURCE:?RUNNER_SOURCE must identify the clean runner checkout}"
: "${RUNNER_SHA256:?RUNNER_SHA256 must identify this exact postcheck runner}"
: "${RUN_ID:?RUN_ID must name a fresh immutable postcheck directory}"

base=/home/bhj/ai-runs
run_root=$base/$RUN_ID
source_run=$base/librpa-qsgw-gate-a1-exact847-component-observer-20260722-17720999-v1
source_work=$source_run/legacy
component_dir=$source_work/librpa.d/qsgw_legacy_components/iter_00001
source_checkpoint=$source_work/librpa.d/qsgw_checkpoints
frozen_run=$base/librpa-qsgw-gate-a1-exact847-fullcoul-nohead-shrinkchi-off-miniter2-20260722-v3
frozen_checkpoint=$frozen_run/legacy/librpa.d/qsgw_checkpoints
candidate_run=$base/librpa-qsgw-gate-a1-exact847-candidate-one-update-20260722-69c33c2f-v3
candidate_trace=$candidate_run/candidate/qsgw_matrices.dat
python=$base/librpa-qsgw-gate0-20260715T1731-7d69a18c/venv/bin/python
current_dir=$RUNNER_SOURCE/qsgw-rebase-evidence/remote/fish-gate-a-current-20260721
symmetry_dir=$RUNNER_SOURCE/qsgw-rebase-evidence/remote/fish-gate-a-symmetry-20260720

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
test ! -e "$source_run/COMPLETE"
test "$(sha256sum "$source_run/FAILED" | awk '{print $1}')" = \
  a5c2a0575e154653b950c04df5743975ebbc23502c74375216e2028b2a63e901
test "$(sha256sum "$source_run/PROVENANCE.txt" | awk '{print $1}')" = \
  976ce29b87fe2e48a11367ffc11079e638336ae6f9be0ee8dab19700b7239780
test "$(sha256sum "$source_work/librpa.stdout" | awk '{print $1}')" = \
  72346e10b4be04b8927ce5278cae1656b3d66a2d640d2dae9bac994f8b6f77ec
test "$(sha256sum "$source_work/librpa.stderr" | awk '{print $1}')" = \
  0cb18c37eb3f24e1fe03482852987d5aa34220d9755b07af197c800486421d90
test "$(sha256sum "$source_work/runtime.txt" | awk '{print $1}')" = \
  f024e18866bfd0fa5862e6ca2cb2dbe625a3c1b58afecb3c1bd965dc66b956f2
test "$(sha256sum "$source_work/homo_lumo_vs_iterations.dat" | awk '{print $1}')" = \
  8157cd3154c5515a8bbe3ede38731cf40c0bb952d47879004f8934d5d0addeb7
test "$(grep -c '^HF file not found: hf_exchange_spin_01_kpt_' \
  "$source_work/librpa.stderr")" -eq 8
grep -Fq 'libRPA finished successfully' "$source_work/librpa.stdout"
grep -Fq 'Iteration 1: HOMO =' "$source_work/librpa.stdout"
test "$(awk 'NF {last=$1} END {print last}' \
  "$source_work/homo_lumo_vs_iterations.dat")" = 1
test -f "$component_dir/metadata.txt"
test "$(find "$component_dir" -maxdepth 1 -type f -name '*.bin' | wc -l)" -eq 168
component_tree_sha=$(
  cd "$component_dir"
  find . -type f -print0 | LC_ALL=C sort -z | xargs -0 sha256sum | sha256sum \
    | awk '{print $1}'
)
test "$component_tree_sha" = \
  e1065cedb703c2eba9ee32aca3303815e0c539553cebabb1c3b333eb9275f4e3
test "$(sha256sum "$candidate_trace" | awk '{print $1}')" = \
  e8f96767491dca9ac2f379598142cbf093b132c2faf9dcf22f3e97c6f271de1e
observer_exe=/tmp/librpa-qsgw-exact847-component-observer-17720999-v1/build/chi0_main.exe
test "$(sha256sum "$observer_exe" | awk '{print $1}')" = \
  1f638533ace7b4bef6f7d94c0f79632fb2ccbf771199a5587f3b9184ca6ad39b

tools_dir=$run_root/tools
mkdir -p "$tools_dir" "$run_root/source"
cp "$0" "$run_root/"
cp "$current_dir/compare_exact847_checkpoints_v1.py" "$tools_dir/"
cp "$current_dir/compare_exact847_component_dump_v1.py" "$tools_dir/"
cp "$current_dir/diagnose_exact847_component_parity_v1.py" "$tools_dir/"
cp "$current_dir/compare_legacy_h0_candidate_trace_v2.py" "$tools_dir/"
cp "$symmetry_dir/compare_legacy_h0_candidate_trace_v1.py" "$tools_dir/"
cp "$symmetry_dir/compare_legacy_band0_native_outputs_v1.py" "$tools_dir/"
cp "$source_run/FAILED" "$run_root/source/source-run-FAILED"
cp "$source_run/PROVENANCE.txt" "$run_root/source/source-run-PROVENANCE.txt"
cp "$source_work/librpa.in" "$run_root/source/"
cp "$source_work/runtime.txt" "$run_root/source/"
cp "$source_work/homo_lumo_vs_iterations.dat" "$run_root/source/"

PYTHONPATH="$tools_dir" "$python" -B \
  "$tools_dir/compare_exact847_checkpoints_v1.py" \
  "$frozen_checkpoint" "$source_checkpoint" \
  "$run_root/observer-checkpoint-comparison.json" \
  --iteration 1 --n-spins 1 --n-kpoints 8 --n-bands 44 \
  >"$run_root/checkpoint-comparison.stdout" \
  2>"$run_root/checkpoint-comparison.stderr"
test ! -s "$run_root/checkpoint-comparison.stderr"
grep -Fq '"diagnostic_complete": true' \
  "$run_root/observer-checkpoint-comparison.json"

PYTHONPATH="$tools_dir" "$python" -B \
  "$tools_dir/compare_exact847_component_dump_v1.py" \
  "$component_dir" "$candidate_trace" \
  "$run_root/legacy-candidate-component-comparison.json" \
  --iteration 1 --n-frequencies 16 --n-spins 1 --n-kpoints 8 --n-bands 44 \
  >"$run_root/component-comparison.stdout" \
  2>"$run_root/component-comparison.stderr"
test ! -s "$run_root/component-comparison.stderr"
grep -Fq '"diagnostic_complete": true' \
  "$run_root/legacy-candidate-component-comparison.json"

checkpoint_parity=false
if grep -Fq '"numerical_parity_passed": true' \
  "$run_root/observer-checkpoint-comparison.json"; then
  checkpoint_parity=true
fi
cat >"$run_root/PROVENANCE.txt" <<EOF
gate=gate_a1_exact847_component_observer_postcheck_v1
acceptance=false_diagnostic_only
runner_commit=$RUNNER_COMMIT
runner_sha256=$RUNNER_SHA256
source_run=$source_run
source_run_failed_postcondition=true
source_failure_reason=expected_exact847_hf_fallback_stderr_was_rejected
source_numerical_execution=completed_successfully
source_component_tree_sha256=$component_tree_sha
observer_executable_sha256=1f638533ace7b4bef6f7d94c0f79632fb2ccbf771199a5587f3b9184ca6ad39b
frozen_legacy_run=$frozen_run
candidate_run=$candidate_run
candidate_trace_sha256=e8f96767491dca9ac2f379598142cbf093b132c2faf9dcf22f3e97c6f271de1e
observer_checkpoint_numerical_parity=$checkpoint_parity
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
cat "$run_root/observer-checkpoint-comparison.json"
cat "$run_root/legacy-candidate-component-comparison.json"
run_succeeded=1
trap - EXIT
