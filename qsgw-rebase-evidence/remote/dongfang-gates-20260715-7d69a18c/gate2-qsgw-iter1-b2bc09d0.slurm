#!/usr/bin/env bash
#SBATCH --job-name=librpa-qsgw-iter1
#SBATCH --partition=48cp2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=12
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --time=01:00:00
#SBATCH --output=/data/home/df_iopcas_bhj/ai-runs/librpa-qsgw-rebase-gates-20260715T1910-7d69a18c/logs/gate2-%j.out
#SBATCH --error=/data/home/df_iopcas_bhj/ai-runs/librpa-qsgw-rebase-gates-20260715T1910-7d69a18c/logs/gate2-%j.err

set -euo pipefail

root=/data/home/df_iopcas_bhj/ai-runs/librpa-qsgw-rebase-gates-20260715T1910-7d69a18c
input_root=$root/inputs/si-k444-canonical-2341555
input_dir=$input_root/dataset
candidate_build=$root/build-candidate-b2bc09d0-v1
candidate_exe=$candidate_build/chi0_main.exe
gate1_root=$root/runs/gate1-g0w0-ab-b2bc09d0-2343096
upstream_g0w0=$gate1_root/upstream
trace_comparator=$root/jobs/compare_qsgw_trace_g0w0_sigc.py
sigc_reader=$root/jobs/summarize_librpa_sigc_matrix_dumps.py
component_reader=$root/jobs/compare_qsgw_component_traces-v3-9f9ec9be.py
closure_validator=$root/jobs/validate_qsgw_trace_closure-v3-4a5de94e.py
fixed_validator=$root/jobs/validate_qsgw_fixed_basis.py
initial_state_validator=$root/jobs/validate_qsgw_initial_state-v1.py
initial_state_test=$root/jobs/test_validate_qsgw_initial_state-v1.py
python=/data/home/df_iopcas_bhj/software-stack/miniconda3/bin/python3
run_root=$root/runs/gate2-qsgw-iter1-b2bc09d0-$SLURM_JOB_ID
tool_dir=$run_root/tools

expected_candidate_commit=b2bc09d00ab49a2ff39c46c72d83ab89bc5ddac6
expected_candidate_exe_sha=a898ef66390d0207158c0f24d1b311fabde1bb11f4a1cd89b6082aee75e54998
expected_candidate_build_manifest_sha=e647822680e91a4597ff2908ed48dcb0280ae2e081329bf623dc1ba3cf9a3761
expected_candidate_build_provenance_sha=3489aeae4e1632ab24483b1241252fbe43187b63946df75e15e9e5be4d77cb8c
expected_dataset_manifest_sha=ec2666cd54303b5ade8d392067cdf4c296a7ba0503f033710e4575784fea48cf
expected_freeze_output_manifest_sha=8d9bab71e5f5f04d05a83a979f9468b124543915d57e762431b80c0ed821745e
expected_contract_sha=25422eee1d5f8cd4db678c05d979bbd29cb516c8088157829d6c315da19ec12b
expected_gate1_output_manifest_sha=4cfa0d4632e3545cac1a1dda694d96ac7326ce87cd753e19013fddab6f2734e2
expected_gate1_provenance_sha=d4b7f11bce3297cb2cf8f82e476340133fcab5bb8a910ec13f585539e47b884f
expected_gate1_comparison_sha=b232ab43ed3db8678979f83bbe7655d4ad3326d10347ae6e82fedcc5b9cdf2c3
expected_trace_comparator_sha=f79f9c2cea50a941677ee8f8f18327bdf791b3c02df8b24bec7d9fdd0debbb63
expected_sigc_reader_sha=a2bf9dcec91dc01c221da0fbc84ac514b3a9ca7c83f065fcb260d7f32b5411fb
expected_component_reader_sha=9f9ec9beb0d08754d1f9a39cb528df9aa4000df7f3443479984ec2df6144bf89
expected_closure_validator_sha=4a5de94e6dbf590dded4a6ecd140aa4227fa61ffa0e73af17ec0f388610cffaf
expected_fixed_validator_sha=569ecb1366bc6dd4584216bd42ac55905a4848b1a0b1bdf96c6236c782de7cb2
expected_initial_state_validator_sha=6bbade9eaeb207b6cea9fa2f80d8cbcd0baeb8cbd760a2ffab5a0fe6f6d4868a
expected_initial_state_test_sha=82634e292a8fc1eb5ed454360ea2367e06687f0547da6503291c850a00e5b339

test ! -e "$run_root"
test -e "$candidate_build/COMPLETE"
test -e "$input_root/COMPLETE"
test -e "$gate1_root/COMPLETE"
test -x "$candidate_exe"
grep -Fqx "source_commit=$expected_candidate_commit" \
  "$candidate_build/build-provenance.txt"
grep -Fqx 'source_status=clean_git_archive_from_commit' \
  "$candidate_build/build-provenance.txt"
grep -Fqx 'final_acceptance=true' \
  "$candidate_build/build-provenance.txt"
test "$(sha256sum "$candidate_exe" | awk '{print $1}')" = \
  "$expected_candidate_exe_sha"
test "$(sha256sum "$candidate_build/SHA256SUMS.txt" | awk '{print $1}')" = \
  "$expected_candidate_build_manifest_sha"
test "$(sha256sum "$candidate_build/build-provenance.txt" | awk '{print $1}')" = \
  "$expected_candidate_build_provenance_sha"
test "$(sha256sum "$input_root/DATASET_SHA256SUMS.txt" | awk '{print $1}')" = \
  "$expected_dataset_manifest_sha"
test "$(sha256sum "$input_root/OUTPUT_SHA256SUMS.txt" | awk '{print $1}')" = \
  "$expected_freeze_output_manifest_sha"
test "$(sha256sum "$input_dir/qsgw_input.contract" | awk '{print $1}')" = \
  "$expected_contract_sha"
test "$(sha256sum "$gate1_root/OUTPUT_SHA256SUMS.txt" | awk '{print $1}')" = \
  "$expected_gate1_output_manifest_sha"
test "$(sha256sum "$gate1_root/PROVENANCE.txt" | awk '{print $1}')" = \
  "$expected_gate1_provenance_sha"
test "$(sha256sum "$gate1_root/g0w0-comparison.json" | awk '{print $1}')" = \
  "$expected_gate1_comparison_sha"
test "$(sha256sum "$trace_comparator" | awk '{print $1}')" = \
  "$expected_trace_comparator_sha"
test "$(sha256sum "$sigc_reader" | awk '{print $1}')" = \
  "$expected_sigc_reader_sha"
test "$(sha256sum "$component_reader" | awk '{print $1}')" = \
  "$expected_component_reader_sha"
test "$(sha256sum "$closure_validator" | awk '{print $1}')" = \
  "$expected_closure_validator_sha"
test "$(sha256sum "$fixed_validator" | awk '{print $1}')" = \
  "$expected_fixed_validator_sha"
test "$(sha256sum "$initial_state_validator" | awk '{print $1}')" = \
  "$expected_initial_state_validator_sha"
test "$(sha256sum "$initial_state_test" | awk '{print $1}')" = \
  "$expected_initial_state_test_sha"
grep -Fq '"passed": true' "$gate1_root/g0w0-comparison.json"
grep -Fqx 'n_scf_kpoints 64' "$input_dir/qsgw_input.contract"
grep -Fqx 'n_headwing_kpoints 0' "$input_dir/qsgw_input.contract"
grep -Fqx 'headwing_grid disabled' "$input_dir/qsgw_input.contract"
grep -Fqx 'hartree_update off' "$input_dir/qsgw_input.contract"
grep -Fqx 'band_update off' "$input_dir/qsgw_input.contract"
test "$(find "$input_dir" -type l | wc -l)" -eq 0
test "$(find "$upstream_g0w0" -maxdepth 1 \
  -name 'Sigc_fk_mn_kgrid_ispin_*_ik_*_ifreq_*.bin' | wc -l)" -eq 384

mkdir -p "$run_root" "$tool_dir"
cp "$0" "$run_root/gate2-qsgw-iter1-b2bc09d0.slurm"
cp "$candidate_build/build-provenance.txt" \
  "$run_root/candidate-build-provenance.txt"
cp "$candidate_build/SHA256SUMS.txt" \
  "$run_root/candidate-build-SHA256SUMS.txt"
cp "$input_root/DATASET_SHA256SUMS.txt" "$run_root/"
cp "$input_root/OUTPUT_SHA256SUMS.txt" \
  "$run_root/input-freeze-OUTPUT_SHA256SUMS.txt"
cp "$input_root/PROVENANCE.txt" "$run_root/input-freeze-PROVENANCE.txt"
cp "$input_root/contract-validation.json" \
  "$run_root/input-contract-validation.json"
cp "$input_dir/qsgw_input.contract" "$run_root/"
cp "$gate1_root/OUTPUT_SHA256SUMS.txt" \
  "$run_root/gate1-OUTPUT_SHA256SUMS.txt"
cp "$gate1_root/PROVENANCE.txt" "$run_root/gate1-PROVENANCE.txt"
cp "$gate1_root/g0w0-comparison.json" "$run_root/"
cp "$trace_comparator" "$run_root/"
cp "$sigc_reader" "$run_root/"
cp "$component_reader" "$tool_dir/compare_qsgw_component_traces.py"
cp "$closure_validator" "$tool_dir/validate_qsgw_trace_closure.py"
cp "$fixed_validator" "$tool_dir/validate_qsgw_fixed_basis.py"
cp "$initial_state_validator" \
  "$tool_dir/validate_qsgw_initial_state.py"
cp "$initial_state_test" \
  "$tool_dir/test_validate_qsgw_initial_state.py"

(
  cd "$input_root"
  sha256sum --check --quiet OUTPUT_SHA256SUMS.txt
  sha256sum --check --quiet DATASET_SHA256SUMS.txt
  printf 'frozen canonical Si-k444 payload and evidence manifests verified\n'
  cd "$gate1_root"
  sha256sum --check --quiet OUTPUT_SHA256SUMS.txt
  printf 'Gate 1 upstream/candidate G0W0 evidence manifest verified\n'
  cd "$candidate_build"
  sha256sum --check --quiet SHA256SUMS.txt
  printf 'clean candidate build manifest verified\n'
) >"$run_root/prerequisite-hash-check.log"

cat >"$run_root/librpa.in" <<EOF
task = qsgw
input_dir = $input_dir
output_dir = .
constants_choice = internal
nfreq = 6
tfgrid_type = minimax
n_params_anacon = -1
n_params_anacon_resample = -1
anacon_nfreq = -1
parallel_routing = libri
vq_threshold = 0
sqrt_coulomb_threshold = 0
use_scalapack_gw_wc = true
output_gw_sigc_ks_mat_kf = false
use_shrink_abfs = false
use_shrink_chi = false
use_pyatb = false
replace_w_head = false
option_dielect_func = 0
use_fullcoul_exx = false
use_fullcoul_eps = true
use_fullcoul_wc = false
use_symmetry_exx = false
use_symmetry_gw = false
use_symmetry_rpa = false
use_kpara_scf_eigvec = false
qsgw_input_contract = qsgw_input.contract
qsgw_mixer = none
qsgw_mixing_beta = 0.2
qsgw_min_iter = 1
qsgw_max_iter = 1
qsgw_write_iteration_matrices = true
qsgw_update_hartree = false
EOF

set +u
source /data/app/intel/oneapi-2024.2/setvars.sh --force \
  >"$run_root/oneapi-setvars.log" 2>&1
set -u
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-12}
export MKL_NUM_THREADS=$OMP_NUM_THREADS
export OPENBLAS_NUM_THREADS=$OMP_NUM_THREADS
export OMP_PROC_BIND=spread
export OMP_PLACES=cores
export I_MPI_PIN_DOMAIN=omp
export LIBRI_DETERMINISTIC_REDUCTION=1

{
  printf 'gate=canonical-si-k444-qsgw-iter1-clean-head\n'
  printf 'candidate_commit=%s\n' "$expected_candidate_commit"
  printf 'candidate_source_status=clean_git_archive_from_commit\n'
  printf 'final_acceptance=true\n'
  printf 'candidate_executable_sha256=%s\n' "$expected_candidate_exe_sha"
  printf 'candidate_build_manifest_sha256=%s\n' \
    "$expected_candidate_build_manifest_sha"
  printf 'candidate_build_provenance_sha256=%s\n' \
    "$expected_candidate_build_provenance_sha"
  printf 'input_root=%s\n' "$input_root"
  printf 'input_freeze_job_id=2341555\n'
  printf 'dataset_manifest_sha256=%s\n' "$expected_dataset_manifest_sha"
  printf 'qsgw_input_contract_sha256=%s\n' "$expected_contract_sha"
  printf 'gate1_run=%s\n' "$gate1_root"
  printf 'gate1_output_manifest_sha256=%s\n' \
    "$expected_gate1_output_manifest_sha"
  printf 'gate1_provenance_sha256=%s\n' \
    "$expected_gate1_provenance_sha"
  printf 'gate1_comparison_sha256=%s\n' \
    "$expected_gate1_comparison_sha"
  printf 'librpa_input_sha256=%s\n' \
    "$(sha256sum "$run_root/librpa.in" | awk '{print $1}')"
  printf 'crystal_symmetry=off\n'
  printf 'time_reversal_reduction=off_full_64_kpoint_input\n'
  printf 'headwing=off\n'
  printf 'hartree=off\n'
  printf 'band=off\n'
  printf 'mixing=none\n'
  printf 'iterations=0:1\n'
  printf 'mpi_ranks=%s\n' "${SLURM_NTASKS:-unknown}"
  printf 'omp_threads=%s\n' "$OMP_NUM_THREADS"
  printf 'LIBRI_DETERMINISTIC_REDUCTION=%s\n' \
    "$LIBRI_DETERMINISTIC_REDUCTION"
  printf 'python=%s\n' "$("$python" --version 2>&1)"
  printf 'numpy=%s\n' \
    "$("$python" -c 'import numpy; print(numpy.__version__)')"
  printf 'slurm_job_id=%s\n' "${SLURM_JOB_ID:-unknown}"
  printf 'slurm_node_list=%s\n' "${SLURM_JOB_NODELIST:-unknown}"
  printf 'started_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  mpirun -np "$SLURM_NTASKS" hostname | sort | uniq -c
} >"$run_root/PROVENANCE.txt"

cd "$run_root"
mpirun -np "$SLURM_NTASKS" "$candidate_exe" \
  >librpa.stdout 2>librpa.stderr
test -f qsgw_matrices.dat
test -f qsgw_eigenvalues.dat
test -f qsgw_iterations.dat
grep -Fq '# qsgw_contract_version 5' qsgw_matrices.dat
grep -Fq '# symmetry unsupported_full_bz_only' qsgw_matrices.dat
grep -Fq '# qsgw_mixer none' qsgw_matrices.dat
grep -Fq '# hartree disabled_stage1' qsgw_matrices.dat
grep -Fq '# band disabled_stage1' qsgw_matrices.dat

PYTHONPATH="$tool_dir" "$python" -B \
  "$tool_dir/test_validate_qsgw_initial_state.py" \
  >"$run_root/initial-state-unit-test.stdout" \
  2>"$run_root/initial-state-unit-test.stderr"

PYTHONPATH="$tool_dir" "$python" -B \
  "$tool_dir/validate_qsgw_initial_state.py" \
  "$run_root/qsgw_matrices.dat" \
  "$run_root/qsgw_iterations.dat" \
  "$input_dir/band_out" \
  "$run_root/initial-state-validation.json" \
  --efermi-tolerance-ha 1e-12 \
  --occupation-tolerance 1e-12 \
  >"$run_root/initial-state-validator.stdout" \
  2>"$run_root/initial-state-validator.stderr"
grep -Fq '"passed": true' \
  "$run_root/initial-state-validation.json"

"$python" "$trace_comparator" \
  "$run_root/qsgw_matrices.dat" \
  "$upstream_g0w0" \
  "$run_root/qsgw-iter1-vs-upstream-g0w0-sigc.json" \
  --iteration 1 \
  --channel 0 \
  --source kgrid \
  --max-abs-tolerance-ha 1e-10 \
  --relative-frobenius-tolerance 1e-10 \
  --input-contract "$input_dir/qsgw_input.contract" \
  >"$run_root/sigc-comparator.stdout" \
  2>"$run_root/sigc-comparator.stderr"
grep -Fq '"passed": true' \
  "$run_root/qsgw-iter1-vs-upstream-g0w0-sigc.json"

PYTHONPATH="$tool_dir" "$python" \
  "$tool_dir/validate_qsgw_trace_closure.py" \
  "$run_root/qsgw_matrices.dat" \
  "$run_root/closure-validation.json" \
  --iterations 0:1 \
  --channel 0 \
  --closure-tolerance-ha 1e-10 \
  --hermiticity-tolerance-ha 1e-10 \
  >"$run_root/closure-validator.stdout" \
  2>"$run_root/closure-validator.stderr"
grep -Fq '"passed": true' "$run_root/closure-validation.json"

PYTHONPATH="$tool_dir" "$python" \
  "$tool_dir/validate_qsgw_fixed_basis.py" \
  "$run_root/qsgw_matrices.dat" \
  "$run_root/qsgw_eigenvalues.dat" \
  "$input_dir/band_out" \
  "$run_root/fixed-basis-validation.json" \
  --iterations 0:1 \
  --channel 0 \
  --eigenvalue-tolerance-ha 1e-10 \
  --invariant-tolerance 1e-10 \
  >"$run_root/fixed-basis-validator.stdout" \
  2>"$run_root/fixed-basis-validator.stderr"
grep -Fq '"passed": true' "$run_root/fixed-basis-validation.json"

printf 'completed_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  >>"$run_root/PROVENANCE.txt"
sha256sum \
  "$run_root/PROVENANCE.txt" \
  "$run_root/librpa.in" \
  "$run_root/librpa.stdout" \
  "$run_root/librpa.stderr" \
  "$run_root/qsgw_matrices.dat" \
  "$run_root/qsgw_eigenvalues.dat" \
  "$run_root/qsgw_iterations.dat" \
  "$run_root/initial-state-validation.json" \
  "$run_root/qsgw-iter1-vs-upstream-g0w0-sigc.json" \
  "$run_root/closure-validation.json" \
  "$run_root/fixed-basis-validation.json" \
  "$run_root/sigc-comparator.stdout" \
  "$run_root/sigc-comparator.stderr" \
  "$run_root/closure-validator.stdout" \
  "$run_root/closure-validator.stderr" \
  "$run_root/fixed-basis-validator.stdout" \
  "$run_root/fixed-basis-validator.stderr" \
  "$run_root/initial-state-unit-test.stdout" \
  "$run_root/initial-state-unit-test.stderr" \
  "$run_root/initial-state-validator.stdout" \
  "$run_root/initial-state-validator.stderr" \
  "$run_root/prerequisite-hash-check.log" \
  "$run_root/input-freeze-OUTPUT_SHA256SUMS.txt" \
  "$run_root/input-freeze-PROVENANCE.txt" \
  "$run_root/input-contract-validation.json" \
  "$run_root/gate1-OUTPUT_SHA256SUMS.txt" \
  "$run_root/gate1-PROVENANCE.txt" \
  "$run_root/g0w0-comparison.json" \
  "$run_root/qsgw_input.contract" \
  "$run_root/candidate-build-SHA256SUMS.txt" \
  "$run_root/gate2-qsgw-iter1-b2bc09d0.slurm" \
  "$run_root/compare_qsgw_trace_g0w0_sigc.py" \
  "$run_root/summarize_librpa_sigc_matrix_dumps.py" \
  "$tool_dir/compare_qsgw_component_traces.py" \
  "$tool_dir/validate_qsgw_trace_closure.py" \
  "$tool_dir/validate_qsgw_fixed_basis.py" \
  "$tool_dir/validate_qsgw_initial_state.py" \
  "$tool_dir/test_validate_qsgw_initial_state.py" \
  >"$run_root/OUTPUT_SHA256SUMS.txt"
touch "$run_root/COMPLETE"
printf 'CANONICAL_SI_K444_QSGW_ITER1=PASS\n'
