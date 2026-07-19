#!/usr/bin/env bash
# Gate 6B8: internal-consistency acceptance of the frozen Gate 6B7 current
# (a25e4e43, Hartree delta_density) QSGW trace, replayed with the v4
# closure observer. No new physics run is performed.
#
# Background: Gate 6B7 (job 2350992) was rejected because the old v40
# Hartree-on oracle is not a valid same-input reference (its read_Vq_full
# path alters the legacy base G0W0 quantities; attribution evidence
# gate6b7-attribution-v3-2351376). The revised Hartree acceptance therefore
# verifies the current implementation directly on the frozen trace:
#   - Hamiltonian closure including delta_vh (<= 1e-10 Ha)
#   - Hermiticity including delta_vh (<= 1e-10 Ha)
#   - iteration-one delta_vh vanishes (<= 1e-10 Ha)
#   - post-reference Hartree response is nonzero (>= 1e-12 Ha)
#   - fixed-basis diagonalization / live-state updates (<= 1e-10)
#   - initial state against the frozen band_out (<= 1e-12)
# The legacy oracle trace closure is re-validated as an observer
# regression. All observers are hash-pinned; the frozen traces are bound
# by the SHA256 recorded in the Gate 6B7 postmortem (job 2351093).
#SBATCH --job-name=qsgw-g6b8-vh-closure
#SBATCH --partition=v100m3
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=/data/home/df_iopcas_bhj/ai-runs/librpa-qsgw-rebase-gates-20260715T1910-7d69a18c/logs/gate6b8-hartree-closure-replay-%j.out
#SBATCH --error=/data/home/df_iopcas_bhj/ai-runs/librpa-qsgw-rebase-gates-20260715T1910-7d69a18c/logs/gate6b8-hartree-closure-replay-%j.err

set -euo pipefail

root=/data/home/df_iopcas_bhj/ai-runs/librpa-qsgw-rebase-gates-20260715T1910-7d69a18c
failed_run=$root/runs/gate6b7-hartree-on-oracle-v40-vs-a25e4e43-mpi4-omp12-miniter2-2350992
oracle_run=$root/runs/gate6b6-legacy-full-reader-hartree-on-mpi4-omp12-miniter2-2348987
input_dir=$root/inputs/si-k444-gate6b1-mpi1-full-abfs-hartree-contracts-2348832/dataset
current_trace=$failed_run/current/qsgw_matrices.dat
current_eigenvalues=$failed_run/current/qsgw_eigenvalues.dat
current_iterations=$failed_run/current/qsgw_iterations.dat
old_trace=$oracle_run/old/qsgw_oracle_matrices.dat
closure_v4=$root/jobs/validate_qsgw_trace_closure-v4-860e8bcd.py
closure_v4_test=$root/jobs/test_validate_qsgw_trace_closure-v4-b6b98149.py
comparator_module=$root/jobs/compare_qsgw_component_traces-v6-hartree-1715ae0a.py
fixed_basis=$root/jobs/validate_qsgw_fixed_basis.py
initial_state=$root/jobs/validate_qsgw_initial_state-v1.py
python=/data/home/df_iopcas_bhj/software-stack/miniconda3/bin/python3
run_root=$root/runs/gate6b8-current-hartree-closure-replay-v1-$SLURM_JOB_ID
tool_dir=$run_root/tools

expected_closure_v4_sha=860e8bcd3c641ebab510d5b41ece95992bb960611855525772a48d58559ca3d5
expected_closure_v4_test_sha=b6b981492a95cfb6ba10db858ff74d18bdd38484bf9cb677993b65e64e3d0e86
expected_comparator_sha=1715ae0a7029bde53599c646c35259381bd1430a847678bd13c8f9600778718f
expected_fixed_basis_sha=569ecb1366bc6dd4584216bd42ac55905a4848b1a0b1bdf96c6236c782de7cb2
expected_initial_state_sha=6bbade9eaeb207b6cea9fa2f80d8cbcd0baeb8cbd760a2ffab5a0fe6f6d4868a
expected_current_trace_sha=f27fb95c7849fd75539d94b142c4b556cf9b74f4ab11e1b9c69492f9a289527a
expected_current_eigenvalues_sha=fa5f16cf120b9e9ca2c6cf3fe048bec76a6f6b15c2f75d35f655f03b47abbaff
expected_current_iterations_sha=0b5e8b7d814ea9e9d1e9051a95b571afec5fd4afda9d5c1eb4a97ee5414732a6
expected_old_trace_sha=829f2c1dbf891e4f2951fe796d22513447ad0b508a1156e1b49d233dfb22894b
expected_hartree_contract_sha=ea2d80953e1e03524f2c63f484a0cf333cbb59a2ed5c00f78eed141c8988edf6

record_failure() {
  local rc=$?
  if [[ -d ${run_root:-/nonexistent} ]]; then
    printf 'failed_utc=%s\nexit_code=%s\n' \
      "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$rc" >"$run_root/FAILED"
  fi
  exit "$rc"
}
trap record_failure ERR

test ! -e "$run_root"
mkdir -p "$tool_dir"
test -e "$failed_run/FAILED"
test ! -e "$failed_run/COMPLETE"
test -e "$oracle_run/COMPLETE"
test ! -e "$oracle_run/FAILED"

while read -r path expected; do
  observed=$(sha256sum "$path" | awk '{print $1}')
  if [[ $observed != "$expected" ]]; then
    printf 'hash_mismatch path=%s expected=%s observed=%s\n' \
      "$path" "$expected" "$observed" >&2
    exit 1
  fi
done <<EOF
$closure_v4 $expected_closure_v4_sha
$closure_v4_test $expected_closure_v4_test_sha
$comparator_module $expected_comparator_sha
$fixed_basis $expected_fixed_basis_sha
$initial_state $expected_initial_state_sha
$current_trace $expected_current_trace_sha
$current_eigenvalues $expected_current_eigenvalues_sha
$current_iterations $expected_current_iterations_sha
$old_trace $expected_old_trace_sha
$input_dir/qsgw_input.hartree.contract $expected_hartree_contract_sha
EOF

cp "$0" "$run_root/gate6b8-current-hartree-closure-replay-v1.slurm"
cp "$comparator_module" "$tool_dir/compare_qsgw_component_traces.py"
cp "$closure_v4" "$tool_dir/validate_qsgw_trace_closure.py"
cp "$closure_v4" "$tool_dir/validate_qsgw_trace_closure_v4.py"
cp "$closure_v4_test" "$tool_dir/test_validate_qsgw_trace_closure_v4.py"
cp "$fixed_basis" "$tool_dir/validate_qsgw_fixed_basis.py"
cp "$initial_state" "$tool_dir/validate_qsgw_initial_state.py"

cat >"$run_root/PROVENANCE.txt" <<EOF
gate=gate6b8-current-hartree-closure-replay-v1
final_acceptance=false_pending_observers
result=PENDING
acceptance_scope=current_implementation_internal_consistency_only
current_trace_job=2350992
current_trace_sha256=$expected_current_trace_sha
oracle_trace_job=2348987
oracle_trace_sha256=$expected_old_trace_sha
oracle_reference_status=invalid_same_input_reference_attribution_2351376
closure_observer_v4_sha256=$expected_closure_v4_sha
python=$($python --version 2>&1)
numpy=$($python -c 'import numpy; print(numpy.__version__)')
slurm_job_id=$SLURM_JOB_ID
slurm_node_list=$SLURM_JOB_NODELIST
started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
EOF

PYTHONPATH="$tool_dir" "$python" -B \
  "$tool_dir/test_validate_qsgw_trace_closure_v4.py" \
  >"$run_root/closure-v4-unit-test.stdout" \
  2>"$run_root/closure-v4-unit-test.stderr"

PYTHONPATH="$tool_dir" "$python" -B \
  "$tool_dir/validate_qsgw_trace_closure_v4.py" \
  "$current_trace" "$run_root/current-closure-validation.json" \
  --iterations 0:2 --channel 0 \
  --closure-tolerance-ha 1e-10 --hermiticity-tolerance-ha 1e-10 \
  --hartree-reference-tolerance-ha 1e-10 \
  --hartree-response-minimum-ha 1e-12 \
  >"$run_root/current-closure-validator.stdout" \
  2>"$run_root/current-closure-validator.stderr"
grep -Fq '"passed": true' "$run_root/current-closure-validation.json"

PYTHONPATH="$tool_dir" "$python" -B \
  "$tool_dir/validate_qsgw_trace_closure_v4.py" \
  "$old_trace" "$run_root/old-closure-validation.json" \
  --iterations 0:2 --channel 0 --legacy-contract \
  --closure-tolerance-ha 1e-10 --hermiticity-tolerance-ha 1e-10 \
  --hartree-reference-tolerance-ha 1e-10 \
  --hartree-response-minimum-ha 1e-12 \
  >"$run_root/old-closure-validator.stdout" \
  2>"$run_root/old-closure-validator.stderr"
grep -Fq '"passed": true' "$run_root/old-closure-validation.json"

PYTHONPATH="$tool_dir" "$python" -B \
  "$tool_dir/validate_qsgw_fixed_basis.py" \
  "$current_trace" "$current_eigenvalues" \
  "$input_dir/band_out" "$run_root/current-fixed-basis-validation.json" \
  --iterations 0:2 --channel 0 \
  --eigenvalue-tolerance-ha 1e-10 --invariant-tolerance 1e-10 \
  >"$run_root/current-fixed-basis-validator.stdout" \
  2>"$run_root/current-fixed-basis-validator.stderr"
grep -Fq '"passed": true' "$run_root/current-fixed-basis-validation.json"

PYTHONPATH="$tool_dir" "$python" -B \
  "$tool_dir/validate_qsgw_initial_state.py" \
  "$current_trace" "$current_iterations" \
  "$input_dir/band_out" "$run_root/current-initial-state-validation.json" \
  --efermi-tolerance-ha 1e-12 --occupation-tolerance 1e-12 \
  >"$run_root/current-initial-state-validator.stdout" \
  2>"$run_root/current-initial-state-validator.stderr"
grep -Fq '"passed": true' "$run_root/current-initial-state-validation.json"

printf 'final_acceptance=true\nresult=PASS\ncompleted_utc=%s\n' \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >>"$run_root/PROVENANCE.txt"
(
  cd "$run_root"
  find . -type f ! -name OUTPUT_SHA256SUMS.txt ! -name COMPLETE \
    ! -name FAILED -print0 | sort -z | xargs -0 sha256sum \
    >OUTPUT_SHA256SUMS.txt
  sha256sum --check --quiet OUTPUT_SHA256SUMS.txt
)
touch "$run_root/COMPLETE"
echo GATE6B8_CURRENT_HARTREE_CLOSURE_REPLAY=PASS
