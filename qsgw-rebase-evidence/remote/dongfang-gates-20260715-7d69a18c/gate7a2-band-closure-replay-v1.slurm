#!/usr/bin/env bash
# Gate 7A2: replay the frozen Gate 7A (job 2358253) qsgw_band trace with
# the v5 closure observer (accepts band operator_fourier and maps it to
# task=qsgw_band) and the fixed-basis observer. No new physics run.
# Acceptance (fail-closed):
#   A. observer unit tests
#   B. v5 closure on the grid channel (0:2, Hartree off)
#   C. v5 closure on the band channel (0:2, task=qsgw_band)
#   D. fixed-basis on the grid channel (0:2)
#SBATCH --job-name=qsgw-g7a2-band-closure
#SBATCH --partition=v100m3
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=/data/home/df_iopcas_bhj/ai-runs/librpa-qsgw-rebase-gates-20260715T1910-7d69a18c/logs/gate7a2-band-closure-replay-%j.out
#SBATCH --error=/data/home/df_iopcas_bhj/ai-runs/librpa-qsgw-rebase-gates-20260715T1910-7d69a18c/logs/gate7a2-band-closure-replay-%j.err

set -euo pipefail

root=/data/home/df_iopcas_bhj/ai-runs/librpa-qsgw-rebase-gates-20260715T1910-7d69a18c
gate7a_run=$root/runs/gate7a-qsgw-band-ad9fd6ed-miniter2-v3-2358253
input_dir=$root/inputs/si-k444-band-v2-2358129/dataset
closure_v5=$root/jobs/validate_qsgw_trace_closure-v5-1cf1fcd6.py
closure_v5_test=$root/jobs/test_validate_qsgw_trace_closure-v5-aaf5c7c3.py
comparator_module=$root/jobs/compare_qsgw_component_traces-v6-hartree-1715ae0a.py
fixed_basis=$root/jobs/validate_qsgw_fixed_basis.py
python=/data/home/df_iopcas_bhj/software-stack/miniconda3/bin/python3
run_root=$root/runs/gate7a2-band-closure-replay-v1-$SLURM_JOB_ID
tool_dir=$run_root/tools

expected_trace_sha=202fee60b1be7bf6d336cf6e0fa8378e75076b18e028019b12584e5ad02743a7
expected_eigenvalues_sha=5347ffee98d1111d6c78eecdcf44d78bf4ae63371ac87543c5c9a769b563ab6a
expected_iterations_sha=0bef1e0076f58c433c5ced73e27bc26f0884c5314c5b7c43ea3db93dd9a86fb5
expected_gate_script_sha=44b1aadb2bec45e92c4f1f74edbe9eabd061fc1e6ee598dd9e14234a00fea482
expected_band_initial_sha=faec7a4a36bc1419dff6c1fbe0f68b627c784895015dde87053cedf988ee5939
expected_grid_initial_sha=0785565daffc966fe003dcf6e81b77d41ddbac6bb2835910d4206ea819422be3
expected_closure_v5_sha=1cf1fcd6a409d4b7b84df2d03e36d2109d1fd37ad8afd9082d8353b1c4bbd7b6
expected_closure_v5_test_sha=aaf5c7c3877890511e0bc3532440a7f69a0bc2f745fdb689552f360094d4c776
expected_comparator_module_sha=1715ae0a7029bde53599c646c35259381bd1430a847678bd13c8f9600778718f
expected_fixed_basis_sha=569ecb1366bc6dd4584216bd42ac55905a4848b1a0b1bdf96c6236c782de7cb2
expected_band_contract_sha=94c1832de2c73fc3d8f44c93bf206a39f80b9e2233d40398af075b19dd40cdd3

record_failure() {
  local rc=$?
  if [[ -d ${run_root:-/nonexistent} ]]; then
    printf 'failed_utc=%s\nexit_code=%s\n' \
      "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$rc" >"$run_root/FAILED"
  fi
  exit "$rc"
}
trap record_failure ERR

assert_passed() {
  "$python" -c 'import json, sys; sys.exit(0 if json.load(open(sys.argv[1]))["passed"] else 1)' "$1"
}

test ! -e "$run_root"
mkdir -p "$tool_dir"
test -e "$gate7a_run/FAILED"
test ! -e "$gate7a_run/COMPLETE"

while read -r path expected; do
  observed=$(sha256sum "$path" | awk '{print $1}')
  if [[ $observed != "$expected" ]]; then
    printf 'hash_mismatch path=%s expected=%s observed=%s\n' \
      "$path" "$expected" "$observed" >&2
    exit 1
  fi
done <<EOF
$gate7a_run/current/qsgw_matrices.dat $expected_trace_sha
$gate7a_run/current/qsgw_eigenvalues.dat $expected_eigenvalues_sha
$gate7a_run/current/qsgw_iterations.dat $expected_iterations_sha
$gate7a_run/gate7a-qsgw-band-ad9fd6ed-miniter2-v3.slurm $expected_gate_script_sha
$gate7a_run/band-initial-state-validation.json $expected_band_initial_sha
$gate7a_run/grid-initial-state-validation.json $expected_grid_initial_sha
$closure_v5 $expected_closure_v5_sha
$closure_v5_test $expected_closure_v5_test_sha
$comparator_module $expected_comparator_module_sha
$fixed_basis $expected_fixed_basis_sha
$input_dir/qsgw_input.band.contract $expected_band_contract_sha
EOF
grep -Fq '"passed": true' "$gate7a_run/band-initial-state-validation.json"
grep -Fq '"passed": true' "$gate7a_run/grid-initial-state-validation.json"

cp "$0" "$run_root/gate7a2-band-closure-replay-v1.slurm"
cp "$closure_v5" "$tool_dir/validate_qsgw_trace_closure.py"
cp "$closure_v5" "$tool_dir/validate_qsgw_trace_closure_v5.py"
cp "$closure_v5_test" "$tool_dir/test_validate_qsgw_trace_closure_v5.py"
cp "$comparator_module" "$tool_dir/compare_qsgw_component_traces.py"
cp "$fixed_basis" "$tool_dir/validate_qsgw_fixed_basis.py"

cat >"$run_root/PROVENANCE.txt" <<EOF
gate=gate7a2-band-closure-replay-v1
final_acceptance=false_pending_observers
result=PENDING
gate7a_job=2358253
gate7a_trace_sha256=$expected_trace_sha
closure_observer_v5_sha256=$expected_closure_v5_sha
band_contract_sha256=$expected_band_contract_sha
slurm_job_id=$SLURM_JOB_ID
slurm_node_list=$SLURM_JOB_NODELIST
started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
EOF

PYTHONPATH="$tool_dir" "$python" -B \
  "$tool_dir/test_validate_qsgw_trace_closure_v5.py" \
  >"$run_root/closure-v5-unit-test.stdout" \
  2>"$run_root/closure-v5-unit-test.stderr"

PYTHONPATH="$tool_dir" "$python" -B \
  "$tool_dir/validate_qsgw_trace_closure_v5.py" \
  "$gate7a_run/current/qsgw_matrices.dat" \
  "$run_root/grid-closure-validation.json" \
  --iterations 0:2 --channel 0 \
  --closure-tolerance-ha 1e-10 --hermiticity-tolerance-ha 1e-10 \
  >"$run_root/grid-closure.stdout" \
  2>"$run_root/grid-closure.stderr"
assert_passed "$run_root/grid-closure-validation.json"

PYTHONPATH="$tool_dir" "$python" -B \
  "$tool_dir/validate_qsgw_trace_closure_v5.py" \
  "$gate7a_run/current/qsgw_matrices.dat" \
  "$run_root/band-closure-validation.json" \
  --iterations 0:2 --channel 1 \
  --closure-tolerance-ha 1e-10 --hermiticity-tolerance-ha 1e-10 \
  >"$run_root/band-closure.stdout" \
  2>"$run_root/band-closure.stderr"
assert_passed "$run_root/band-closure-validation.json"

PYTHONPATH="$tool_dir" "$python" -B \
  "$tool_dir/validate_qsgw_fixed_basis.py" \
  "$gate7a_run/current/qsgw_matrices.dat" \
  "$gate7a_run/current/qsgw_eigenvalues.dat" \
  "$input_dir/band_out" \
  "$run_root/grid-fixed-basis-validation.json" \
  --iterations 0:2 --channel 0 \
  --eigenvalue-tolerance-ha 1e-10 --invariant-tolerance 1e-10 \
  >"$run_root/grid-fixed-basis.stdout" \
  2>"$run_root/grid-fixed-basis.stderr"
assert_passed "$run_root/grid-fixed-basis-validation.json"

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
echo GATE7A2_BAND_CLOSURE_REPLAY=PASS
