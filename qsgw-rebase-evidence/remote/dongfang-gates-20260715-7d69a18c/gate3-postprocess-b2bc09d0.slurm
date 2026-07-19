#!/usr/bin/env bash
#SBATCH --job-name=qsgw-gate3-post
#SBATCH --partition=48cp2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/data/home/df_iopcas_bhj/ai-runs/librpa-qsgw-rebase-gates-20260715T1910-7d69a18c/logs/gate3-post-%j.out
#SBATCH --error=/data/home/df_iopcas_bhj/ai-runs/librpa-qsgw-rebase-gates-20260715T1910-7d69a18c/logs/gate3-post-%j.err

set -euo pipefail

root=/data/home/df_iopcas_bhj/ai-runs/librpa-qsgw-rebase-gates-20260715T1910-7d69a18c
source_root=$root/runs/gate3-old-new-direct-miniter10-b2bc09d0-2343109
frozen_root=$root/runs/gate3-old-new-direct-miniter10-ce1e1859-2342954
common_root=$root/inputs/si-k444-gate3-common-v3-2342408
input_dir=$common_root/dataset
run_root=$root/evidence/gate3-postprocess-b2bc09d0-$SLURM_JOB_ID
tool_dir=$run_root/tools
python=/data/home/df_iopcas_bhj/software-stack/miniconda3/bin/python3

comparator_source=$root/jobs/compare_qsgw_component_traces-v4-c3daf072.py
comparator_test_source=$root/jobs/test_compare_qsgw_legacy_v4_current_v5-v4-501ef472.py
closure_source=$root/jobs/validate_qsgw_trace_closure-v3-4a5de94e.py
closure_test_source=$root/jobs/test_validate_qsgw_trace_closure-v3-38de02fa.py
fixed_source=$root/jobs/validate_qsgw_fixed_basis.py
initial_source=$root/jobs/validate_qsgw_initial_state-v1.py
initial_test_source=$root/jobs/test_validate_qsgw_initial_state-v1.py

expected_source_provenance_sha=30626be29da1250f4093aff64cae0c430011340ec5d17fd7af6bda24e9a0f908
expected_source_script_sha=8811be1d259b85efd2f69a3fdcea2014368b6ee65ab396d1c8d20134a3daa03b
expected_source_miniter10_sha=5b727273ab0495811825bc5d9abb4a7609f3a255566b2fef304246f1ef0c36a1
expected_common_output_sha=3b7811c12df23a563e597cae2fb80f592c045d27c803a1afc7ead31d7d7439d4
expected_common_dataset_sha=df793f964077ca03de7ad4723975736d0d0fb85d7e548241f3a7a04677cc05f7
expected_common_provenance_sha=54e2a4b0f07883f4341d46e41a80fd8ae1ebac7e25d263699eda8819b00ce0d8
expected_frozen_output_sha=6b3c4cbe496cf0f4d2ca38c539444d3f56d48ee31ad84987cc3dbcdb51db3e4b
expected_frozen_provenance_sha=ddcb456b20a5d6dc4c48811e7421947d375d28807ef978e5f778b08efae9896c
expected_frozen_matrix_sha=eece0f666871076a61607a8c5c4c17993148c2a932c6ebdf00bd53984090f074
expected_frozen_homo_lumo_sha=0d5471827f4ec76161bc1a4af79b46eee746350401fa81d1a26b44940fa31b38
expected_comparator_sha=c3daf072f222083a7ebdb9cf45f154d4bef64474f76db05992479a66fe30ebbc
expected_comparator_test_sha=501ef472ec997a13313081f1f5945ef47e03f83ca16a7603770b32c66dc54ced
expected_closure_sha=4a5de94e6dbf590dded4a6ecd140aa4227fa61ffa0e73af17ec0f388610cffaf
expected_closure_test_sha=38de02fabc0dde41911e5152b0b08a9987b312b0cea29c69396bb9e392e9c745
expected_fixed_sha=569ecb1366bc6dd4584216bd42ac55905a4848b1a0b1bdf96c6236c782de7cb2
expected_initial_sha=6bbade9eaeb207b6cea9fa2f80d8cbcd0baeb8cbd760a2ffab5a0fe6f6d4868a
expected_initial_test_sha=82634e292a8fc1eb5ed454360ea2367e06687f0547da6503291c850a00e5b339

test ! -e "$run_root"
test ! -e "$source_root/COMPLETE"
test -e "$frozen_root/COMPLETE"
test "$(sha256sum "$source_root/PROVENANCE.txt" | awk '{print $1}')" = \
  "$expected_source_provenance_sha"
test "$(sha256sum "$source_root/gate3-old-new-direct-miniter10-b2bc09d0-v4.slurm" | awk '{print $1}')" = \
  "$expected_source_script_sha"
test "$(sha256sum "$source_root/old-new-comparison.json" | awk '{print $1}')" = \
  "$expected_source_miniter10_sha"
grep -Fq '"passed": true' "$source_root/old-new-comparison.json"
test "$(grep -c '^[0-9]' "$source_root/old/homo_lumo_vs_iterations.dat")" -eq 11
test "$(grep -c '^[0-9]' "$source_root/current/qsgw_iterations.dat")" -eq 11
test -e "$common_root/COMPLETE"
test "$(sha256sum "$common_root/OUTPUT_SHA256SUMS.txt" | awk '{print $1}')" = \
  "$expected_common_output_sha"
test "$(sha256sum "$common_root/DATASET_SHA256SUMS.txt" | awk '{print $1}')" = \
  "$expected_common_dataset_sha"
test "$(sha256sum "$common_root/PROVENANCE.txt" | awk '{print $1}')" = \
  "$expected_common_provenance_sha"
(
  cd "$common_root"
  sha256sum --check --quiet OUTPUT_SHA256SUMS.txt
  sha256sum --check --quiet DATASET_SHA256SUMS.txt
)
test "$(sha256sum "$frozen_root/OUTPUT_SHA256SUMS.txt" | awk '{print $1}')" = \
  "$expected_frozen_output_sha"
test "$(sha256sum "$frozen_root/PROVENANCE.txt" | awk '{print $1}')" = \
  "$expected_frozen_provenance_sha"
test "$(sha256sum "$frozen_root/old/qsgw_oracle_matrices.dat" | awk '{print $1}')" = \
  "$expected_frozen_matrix_sha"
test "$(sha256sum "$frozen_root/old/homo_lumo_vs_iterations.dat" | awk '{print $1}')" = \
  "$expected_frozen_homo_lumo_sha"
(
  cd "$frozen_root"
  sha256sum --check --quiet OUTPUT_SHA256SUMS.txt
)

while read -r path expected; do
  test "$(sha256sum "$path" | awk '{print $1}')" = "$expected"
done <<EOF
$comparator_source $expected_comparator_sha
$comparator_test_source $expected_comparator_test_sha
$closure_source $expected_closure_sha
$closure_test_source $expected_closure_test_sha
$fixed_source $expected_fixed_sha
$initial_source $expected_initial_sha
$initial_test_source $expected_initial_test_sha
EOF

mkdir -p "$tool_dir"
cp "$0" "$run_root/gate3-postprocess-b2bc09d0.slurm"
cp "$comparator_source" "$tool_dir/compare_qsgw_component_traces.py"
cp "$comparator_source" "$tool_dir/compare_qsgw_component_traces_v3.py"
cp "$comparator_test_source" \
  "$tool_dir/test_compare_qsgw_legacy_v4_current_v5_v3.py"
cp "$closure_source" "$tool_dir/validate_qsgw_trace_closure.py"
cp "$closure_source" "$tool_dir/validate_qsgw_trace_closure_v3.py"
cp "$closure_test_source" \
  "$tool_dir/test_validate_qsgw_trace_closure_v3.py"
cp "$fixed_source" "$tool_dir/validate_qsgw_fixed_basis.py"
cp "$initial_source" "$tool_dir/validate_qsgw_initial_state.py"
cp "$initial_test_source" \
  "$tool_dir/test_validate_qsgw_initial_state.py"
cp "$source_root/PROVENANCE.txt" "$run_root/source-PROVENANCE.txt"
cp "$source_root/old-new-comparison.json" \
  "$run_root/source-old-new-miniter10.json"
cp "$frozen_root/OUTPUT_SHA256SUMS.txt" \
  "$run_root/frozen-OUTPUT_SHA256SUMS.txt"
cp "$frozen_root/PROVENANCE.txt" "$run_root/frozen-PROVENANCE.txt"

sacct -j 2343109 \
  --format=JobID,JobName%32,State,Elapsed,ExitCode -n -P \
  >"$run_root/source-job.sacct"
grep -Fq '2343109|qsgw-gate3-min10|FAILED|' \
  "$run_root/source-job.sacct"
grep -Fq '2343109.1|hydra_bstrap_proxy|COMPLETED|' \
  "$run_root/source-job.sacct"
grep -Fq '2343109.2|hydra_bstrap_proxy|COMPLETED|' \
  "$run_root/source-job.sacct"

sha256sum \
  "$source_root/old/qsgw_oracle_matrices.dat" \
  "$source_root/old/homo_lumo_vs_iterations.dat" \
  "$source_root/current/qsgw_matrices.dat" \
  "$source_root/current/qsgw_eigenvalues.dat" \
  "$source_root/current/qsgw_iterations.dat" \
  "$frozen_root/old/qsgw_oracle_matrices.dat" \
  "$frozen_root/old/homo_lumo_vs_iterations.dat" \
  >"$run_root/SOURCE_RAW_SHA256SUMS.txt"

PYTHONPATH="$tool_dir" "$python" -B \
  "$tool_dir/test_compare_qsgw_legacy_v4_current_v5_v3.py" \
  >"$run_root/comparator-unit-test.stdout" \
  2>"$run_root/comparator-unit-test.stderr"
PYTHONPATH="$tool_dir" "$python" -B \
  "$tool_dir/test_validate_qsgw_trace_closure_v3.py" \
  >"$run_root/closure-unit-test.stdout" \
  2>"$run_root/closure-unit-test.stderr"
PYTHONPATH="$tool_dir" "$python" -B \
  "$tool_dir/test_validate_qsgw_initial_state.py" \
  >"$run_root/initial-state-unit-test.stdout" \
  2>"$run_root/initial-state-unit-test.stderr"

run_comparison() {
  local old_trace=$1
  local iterations=$2
  local output=$3
  shift 3
  PYTHONPATH="$tool_dir" "$python" -B \
    "$tool_dir/compare_qsgw_component_traces.py" \
    "$old_trace" \
    "$source_root/current/qsgw_matrices.dat" \
    "$run_root/$output.json" \
    --iterations "$iterations" \
    --channel 0 \
    --contract-mode legacy_v4_to_current_v5 \
    --current-eigenvalue-trace \
      "$source_root/current/qsgw_eigenvalues.dat" \
    --current-iteration-trace \
      "$source_root/current/qsgw_iterations.dat" \
    --expected-legacy-use-fullcoul-exx 0 \
    --frequency-tolerance 1e-10 \
    --matrix-max-abs-tolerance-ha 1e-8 \
    --matrix-relative-tolerance 1e-8 \
    --eigenvalue-tolerance 1e-6 \
    --gap-tolerance-ev 1e-5 \
    --degeneracy-tolerance 1e-8 \
    --state-tolerance 1e-10 \
    "$@" \
    >"$run_root/$output.stdout" \
    2>"$run_root/$output.stderr"
  grep -Fq '"passed": true' "$run_root/$output.json"
}

run_comparison \
  "$source_root/old/qsgw_oracle_matrices.dat" 0:10 old-new-comparison
run_comparison \
  "$source_root/old/qsgw_oracle_matrices.dat" 0:5 \
  old-new-miniter5-comparison --allow-iteration-prefix
run_comparison \
  "$frozen_root/old/qsgw_oracle_matrices.dat" 0:10 \
  frozen-old-vs-current-miniter10
run_comparison \
  "$frozen_root/old/qsgw_oracle_matrices.dat" 0:5 \
  frozen-old-vs-current-miniter5 --allow-iteration-prefix

PYTHONPATH="$tool_dir" "$python" -B \
  "$tool_dir/validate_qsgw_initial_state.py" \
  "$source_root/current/qsgw_matrices.dat" \
  "$source_root/current/qsgw_iterations.dat" \
  "$input_dir/band_out" \
  "$run_root/current-initial-state-validation.json" \
  --efermi-tolerance-ha 1e-12 \
  --occupation-tolerance 1e-12 \
  >"$run_root/current-initial-state-validator.stdout" \
  2>"$run_root/current-initial-state-validator.stderr"
grep -Fq '"passed": true' \
  "$run_root/current-initial-state-validation.json"

PYTHONPATH="$tool_dir" "$python" -B \
  "$tool_dir/validate_qsgw_trace_closure.py" \
  "$source_root/old/qsgw_oracle_matrices.dat" \
  "$run_root/old-closure-validation.json" \
  --iterations 0:10 \
  --channel 0 \
  --legacy-contract \
  --closure-tolerance-ha 1e-10 \
  --hermiticity-tolerance-ha 1e-10 \
  >"$run_root/old-closure-validator.stdout" \
  2>"$run_root/old-closure-validator.stderr"
grep -Fq '"passed": true' "$run_root/old-closure-validation.json"

PYTHONPATH="$tool_dir" "$python" -B \
  "$tool_dir/validate_qsgw_trace_closure.py" \
  "$source_root/current/qsgw_matrices.dat" \
  "$run_root/current-closure-validation.json" \
  --iterations 0:10 \
  --channel 0 \
  --closure-tolerance-ha 1e-10 \
  --hermiticity-tolerance-ha 1e-10 \
  >"$run_root/current-closure-validator.stdout" \
  2>"$run_root/current-closure-validator.stderr"
grep -Fq '"passed": true' "$run_root/current-closure-validation.json"

PYTHONPATH="$tool_dir" "$python" -B \
  "$tool_dir/validate_qsgw_fixed_basis.py" \
  "$source_root/current/qsgw_matrices.dat" \
  "$source_root/current/qsgw_eigenvalues.dat" \
  "$input_dir/band_out" \
  "$run_root/current-fixed-basis-validation.json" \
  --iterations 0:10 \
  --channel 0 \
  --eigenvalue-tolerance-ha 1e-10 \
  --invariant-tolerance 1e-10 \
  >"$run_root/current-fixed-basis-validator.stdout" \
  2>"$run_root/current-fixed-basis-validator.stderr"
grep -Fq '"passed": true' \
  "$run_root/current-fixed-basis-validation.json"

cat >"$run_root/PROVENANCE.txt" <<EOF
gate=gate3-postprocess-recovery
final_acceptance=true
source_numerical_job=2343109
source_numerical_job_state=FAILED_observer_contract_only
source_old_mpi_state=COMPLETED
source_current_mpi_state=COMPLETED
source_root=$source_root
source_provenance_sha256=$expected_source_provenance_sha
source_script_sha256=$expected_source_script_sha
source_original_miniter10_comparison_sha256=$expected_source_miniter10_sha
common_input_root=$common_root
common_input_output_manifest_sha256=$expected_common_output_sha
common_input_dataset_manifest_sha256=$expected_common_dataset_sha
common_input_provenance_sha256=$expected_common_provenance_sha
frozen_old_job=2342954
frozen_old_root=$frozen_root
frozen_old_output_manifest_sha256=$expected_frozen_output_sha
frozen_old_provenance_sha256=$expected_frozen_provenance_sha
frozen_old_matrix_sha256=$expected_frozen_matrix_sha
frozen_old_homo_lumo_sha256=$expected_frozen_homo_lumo_sha
candidate_commit=b2bc09d00ab49a2ff39c46c72d83ab89bc5ddac6
upstream_commit=95c4c08009aa6752a6d386289abe1fb4358489ca
comparator_sha256=$expected_comparator_sha
comparator_test_sha256=$expected_comparator_test_sha
iteration_selection=exact_0:10,prefix_0:5
matrix_max_abs_tolerance_ha=1e-8
matrix_relative_tolerance=1e-8
eigenvalue_tolerance_ha=1e-6
gap_tolerance_ev=1e-5
closure_tolerance_ha=1e-10
hermiticity_tolerance_ha=1e-10
fixed_basis_eigenvalue_tolerance_ha=1e-10
fixed_basis_invariant_tolerance=1e-10
python=$($python --version 2>&1)
numpy=$($python -c 'import numpy; print(numpy.__version__)')
slurm_job_id=$SLURM_JOB_ID
slurm_node_list=$SLURM_JOB_NODELIST
completed_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
EOF

(
  cd "$run_root"
  find . -type f ! -name OUTPUT_SHA256SUMS.txt -print0 | sort -z | \
    xargs -0 sha256sum >OUTPUT_SHA256SUMS.txt
  sha256sum --check --quiet OUTPUT_SHA256SUMS.txt
)
touch "$run_root/COMPLETE"
echo GATE3_POSTPROCESS_RECOVERY=PASS
