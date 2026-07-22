#!/usr/bin/env bash
set -euo pipefail

: "${RUNNER_COMMIT:?RUNNER_COMMIT must identify the clean runner checkout}"
: "${RUNNER_SOURCE:?RUNNER_SOURCE must identify the clean runner checkout}"
: "${RUNNER_SHA256:?RUNNER_SHA256 must identify this exact runner}"
: "${RUN_ID:?RUN_ID must name a fresh immutable postcheck directory}"

base=/home/bhj/ai-runs
run_root=$base/$RUN_ID
source_run=$base/librpa-qsgw-gate-a1-exact847-candidate-one-update-20260722-69c33c2f-v3
candidate=$source_run/candidate
input_overlay=$source_run/input-overlay-state-basis
legacy_run=$base/librpa-qsgw-gate-a1-exact847-fullcoul-nohead-shrinkchi-off-miniter2-20260722-v3
legacy_checkpoint=$legacy_run/legacy/librpa.d/qsgw_checkpoints
legacy_history=$legacy_run/legacy/homo_lumo_vs_iterations.dat
python=/home/bhj/ai-runs/librpa-qsgw-gate0-20260715T1731-7d69a18c/venv/bin/python

expected_source_hashes=(
  2bd0ac98137062760e461b6acfa21b6bac6c71518fc6b651f2e2c276bca87660
  c0c258bed2f8820462df64c7edb1c1cf38439af1198c27018ae944352d5cc066
  4f86b95949a2e647a84e7df40dfa770f93c4d92c1c13fa2f1fe588fbff74ddcc
  c23a8618d079c9f06c0abd0774ea844f81c521443175293b8472b36ccd7e9bec
  ca8257f0a51dd367800de41fe849bc2a86b35b2e7a1b4b1051137bdd3c89f286
  bf4028c527a4ea7dd9a6a2a2babfc6b5ffd5d0d4848b142f2836b681d1881a72
  910a95d7f1bf9d25d3ea66c15b3945fc0ebfcc16e09994caa3499482995d39ee
  e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
  d4fcfc476e55f6f0ed53e8024ac41fa5bdddea60a3528ede77066bc6dbc40423
  e8f96767491dca9ac2f379598142cbf093b132c2faf9dcf22f3e97c6f271de1e
  6cdc4de20467297afa003959bac0fe091a361b782b16539a648fe590ad8e9ae1
  3627de49ac9f4d5b05d925b2cbe97e214d00848e4f0868fef2d6259793b8ccfc
)
source_files=(
  "$source_run/FAILED"
  "$source_run/PROVENANCE.txt"
  "$source_run/PARAMETER_MAPPING.txt"
  "$source_run/shared-reader-input-audit.json"
  "$source_run/vxc-state-basis-upgrade.json"
  "$candidate/librpa.in"
  "$candidate/librpa.stdout"
  "$candidate/librpa.stderr"
  "$candidate/runtime.txt"
  "$candidate/qsgw_matrices.dat"
  "$candidate/qsgw_eigenvalues.dat"
  "$candidate/qsgw_iterations.dat"
)
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
test -e "$source_run/FAILED"
test ! -e "$source_run/COMPLETE"
test ! -e "$source_run/ONE_UPDATE_PARITY_GREEN"
test "${#source_files[@]}" -eq "${#expected_source_hashes[@]}"
for index in "${!source_files[@]}"; do
  test "$(sha256sum "${source_files[$index]}" | awk '{print $1}')" = \
    "${expected_source_hashes[$index]}"
done
grep -Fq 'libRPA finished successfully' "$candidate/librpa.stdout"
test ! -s "$candidate/librpa.stderr"
test "$(awk 'NF && $1 !~ /^#/ {last=$1} END {print last}' \
  "$candidate/qsgw_iterations.dat")" = 1
grep -Fqx '# qsgw_contract_version 6' "$candidate/qsgw_matrices.dat"
grep -Fqx '# qsgw_input_contract_sha256 c99e51f99ef96e2ac396116c552a58d6ca52c4ffa309997648679884a008e6fe' \
  "$candidate/qsgw_matrices.dat"
test "$(sha256sum "$input_overlay/qsgw_input.contract" | awk '{print $1}')" = \
  c99e51f99ef96e2ac396116c552a58d6ca52c4ffa309997648679884a008e6fe
test "$(sha256sum "$input_overlay/qsgw_vxc_scf.manifest" | awk '{print $1}')" = \
  9048470bccadacd8ba04ad3a0467d9a7958830bafdb0174cbb08a857569064cc
grep -Fqx 'basis state' "$input_overlay/qsgw_vxc_scf.manifest"
grep -Fqx 'gauge mf0_state' "$input_overlay/qsgw_vxc_scf.manifest"

test -e "$legacy_run/FAILED"
test ! -e "$legacy_run/COMPLETE"
test "$(sha256sum "$legacy_run/FAILED" | awk '{print $1}')" = \
  acea8cb23bf445e18da415193e099532413036a0ce9c28393835b38fa1cc5ddf
test "$(cat "$legacy_checkpoint/latest_iteration.txt")" = 1
test "$(sha256sum "$legacy_history" | awk '{print $1}')" = \
  8157cd3154c5515a8bbe3ede38731cf40c0bb952d47879004f8934d5d0addeb7
grep -Fqx '1 4.08114 4.08114 4.08114' "$legacy_history"
for ik in $(seq 1 8); do
  printf -v name 'H0_GW_spin_01_k_%06d.bin' "$ik"
  test "$(sha256sum "$legacy_checkpoint/iter_00001/$name" | awk '{print $1}')" = \
    "${expected_checkpoint_matrix_shas[$((ik - 1))]}"
done

symmetry_dir=$RUNNER_SOURCE/qsgw-rebase-evidence/remote/fish-gate-a-symmetry-20260720
current_dir=$RUNNER_SOURCE/qsgw-rebase-evidence/remote/fish-gate-a-current-20260721
tools_dir=$run_root/tools
mkdir -p "$tools_dir" "$run_root/source"
cp "$0" "$run_root/"
cp "$symmetry_dir/observer-tools-v1/compare_qsgw_component_traces.py" \
  "$tools_dir/compare_qsgw_component_traces_v4.py"
cp "$RUNNER_SOURCE/regression_tests/backend/comparisons/cmp_qsgw.py" \
  "$tools_dir/cmp_qsgw_v6.py"
cp "$current_dir/compare_qsgw_component_traces_v6_adapter.py" \
  "$tools_dir/compare_qsgw_component_traces.py"
cp "$symmetry_dir/observer-tools-v1/validate_qsgw_trace_closure-v3-4a5de94e.py" \
  "$tools_dir/validate_qsgw_trace_closure.py"
cp "$symmetry_dir/observer-tools-v1/validate_qsgw_fixed_basis.py" \
  "$tools_dir/validate_qsgw_fixed_basis.py"
cp "$symmetry_dir/observer-tools-v1/validate_qsgw_initial_state-v1.py" \
  "$tools_dir/validate_qsgw_initial_state.py"
cp "$symmetry_dir/compare_legacy_band0_native_outputs_v1.py" "$tools_dir/"
cp "$symmetry_dir/compare_legacy_h0_candidate_trace_v1.py" "$tools_dir/"
cp "$current_dir/compare_legacy_h0_candidate_trace_v2.py" "$tools_dir/"
cp "$source_run/FAILED" "$run_root/source/source-run-FAILED"
cp "$source_run/PROVENANCE.txt" "$run_root/source/source-run-PROVENANCE.txt"
cp "$candidate/librpa.in" "$run_root/source/"
cp "$candidate/runtime.txt" "$run_root/source/"
cp "$candidate/qsgw_iterations.dat" "$run_root/source/"

PYTHONPATH="$tools_dir" "$python" -B \
  "$tools_dir/validate_qsgw_trace_closure.py" \
  "$candidate/qsgw_matrices.dat" "$run_root/candidate-closure.json" \
  --iterations 0:1 --channel 0
PYTHONPATH="$tools_dir" "$python" -B \
  "$tools_dir/validate_qsgw_fixed_basis.py" \
  "$candidate/qsgw_matrices.dat" "$candidate/qsgw_eigenvalues.dat" \
  "$input_overlay/band_out" "$run_root/candidate-fixed-basis.json" \
  --iterations 0:1 --channel 0
PYTHONPATH="$tools_dir" "$python" -B \
  "$tools_dir/validate_qsgw_initial_state.py" \
  "$candidate/qsgw_matrices.dat" "$candidate/qsgw_iterations.dat" \
  "$input_overlay/band_out" "$run_root/candidate-initial-state.json"
for report in candidate-closure.json candidate-fixed-basis.json \
  candidate-initial-state.json; do
  grep -Fq '"passed": true' "$run_root/$report"
done

set +e
PYTHONPATH="$tools_dir" "$python" -B \
  "$tools_dir/compare_legacy_h0_candidate_trace_v2.py" \
  "$legacy_checkpoint" "$candidate/qsgw_matrices.dat" \
  "$candidate/qsgw_eigenvalues.dat" \
  "$run_root/legacy-candidate-comparison.json" \
  --iterations 1 --n-spins 1 --n-kpoints 8 --n-bands 44 \
  --occupied-bands 4 \
  >"$run_root/comparator.stdout" 2>"$run_root/comparator.stderr"
comparison_rc=$?
set -e
test "$comparison_rc" -eq 0 -o "$comparison_rc" -eq 1
test -s "$run_root/legacy-candidate-comparison.json"
printf '%s\n' "$comparison_rc" >"$run_root/comparison-exit-code.txt"

parity=false
acceptance=false_diagnostic_only
marker=DIAGNOSTIC_COMPLETE
if grep -Fq '"parity_passed": true' \
  "$run_root/legacy-candidate-comparison.json"; then
  parity=true
  acceptance=true_one_update_parity
  marker=ONE_UPDATE_PARITY_GREEN
fi
cat >"$run_root/PROVENANCE.txt" <<EOF
gate=gate_a1_exact847_candidate_one_update_postcheck_v1
acceptance=$acceptance
runner_commit=$RUNNER_COMMIT
runner_sha256=$RUNNER_SHA256
source_failed_run=$source_run
source_failure_reason=observer_adapter_missing_private_matrix_groups_export
source_numerical_execution=completed_successfully
source_candidate_matrix_sha256=${expected_source_hashes[9]}
source_candidate_eigenvalue_sha256=${expected_source_hashes[10]}
source_candidate_iteration_sha256=${expected_source_hashes[11]}
corrected_input_contract_sha256=c99e51f99ef96e2ac396116c552a58d6ca52c4ffa309997648679884a008e6fe
corrected_vxc_manifest_sha256=9048470bccadacd8ba04ad3a0467d9a7958830bafdb0174cbb08a857569064cc
legacy_run=$legacy_run
legacy_run_is_green=false
legacy_checkpoint_iteration=1
comparison_exit_code=$comparison_rc
one_update_parity=$parity
symmetry=on
headwing=off
hartree=off
use_shrink_abfs=true
use_shrink_chi=false
iterations=0:1
completed_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
EOF

(
  cd "$run_root"
  find . -type f ! -name OUTPUT_SHA256SUMS.txt \
    ! -name ONE_UPDATE_PARITY_GREEN ! -name DIAGNOSTIC_COMPLETE \
    ! -name FAILED -print0 | LC_ALL=C sort -z | xargs -0 sha256sum \
    >OUTPUT_SHA256SUMS.txt
  sha256sum --check --quiet OUTPUT_SHA256SUMS.txt
)
touch "$run_root/$marker"
cat "$run_root/PROVENANCE.txt"
cat "$run_root/legacy-candidate-comparison.json"
run_succeeded=1
trap - EXIT
