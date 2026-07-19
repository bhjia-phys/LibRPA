#!/usr/bin/env bash
#SBATCH --job-name=qsgw-gate5a-b02
#SBATCH --partition=v100m3
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:8
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --time=04:00:00
#SBATCH --output=/data/home/df_iopcas_bhj/ai-runs/librpa-qsgw-rebase-gates-20260715T1910-7d69a18c/logs/gate5a-linear-beta02-%j.out
#SBATCH --error=/data/home/df_iopcas_bhj/ai-runs/librpa-qsgw-rebase-gates-20260715T1910-7d69a18c/logs/gate5a-linear-beta02-%j.err

set -euo pipefail

target_iter=5
mixing_beta=0.2
root=/data/home/df_iopcas_bhj/ai-runs/librpa-qsgw-rebase-gates-20260715T1910-7d69a18c
common_root=$root/inputs/si-k444-gate3-common-v3-2342408
input_dir=$common_root/dataset
gate4b_root=$root/evidence/gate4b-head-wing-pass-a25e4e43-2348159
old_build=/data/home/df_iopcas_bhj/qsgw-builds/old-new-a03a4c0-majorfix-v14-20260713
old_exe=$old_build/old/build/chi0_main.exe
candidate_build=$root/build-candidate-a25e4e43-v1
candidate_exe=$candidate_build/chi0_main.exe
comparator_source=$root/jobs/compare_qsgw_component_traces-v5-linear-56c8d808.py
comparator_test_source=$root/jobs/test_compare_qsgw_legacy_v4_current_v5-v5-linear-785e0c50.py
residual_source=$root/jobs/summarize_qsgw_linear_residuals-638f1195.py
residual_test_source=$root/jobs/test_summarize_qsgw_linear_residuals-2d2f19da.py
closure_source=$root/jobs/validate_qsgw_trace_closure-v3-4a5de94e.py
closure_test_source=$root/jobs/test_validate_qsgw_trace_closure-v3-38de02fa.py
fixed_source=$root/jobs/validate_qsgw_fixed_basis.py
initial_source=$root/jobs/validate_qsgw_initial_state-v1.py
initial_test_source=$root/jobs/test_validate_qsgw_initial_state-v1.py
python=/data/home/df_iopcas_bhj/software-stack/miniconda3/bin/python3
run_root=$root/runs/gate5a-linear-beta02-miniter${target_iter}-a25e4e43-$SLURM_JOB_ID
old_run=$run_root/old
current_run=$run_root/current
tool_dir=$run_root/tools

expected_old_commit=e08f4a130df7661e9ac355b9be45fb2bf9c3ed01
expected_old_exe_sha=a1292eff5364565d5b7463596882580a9a758b6e0e3e600ac5dfe67113bef788
expected_old_build_provenance_sha=8dfa0742e5618be14c27a96764ddf240af2ed8fc19662bac4d506bd44e77c96b
expected_candidate_commit=a25e4e4398d96bccdf44df00bfaad31c5e6035d6
expected_candidate_exe_sha=2072fc09a44a5f246513be152a06832dccf777da04b3dd8911b70e0f9a31591e
expected_candidate_build_manifest_sha=e0377a567a3504dbc61ece4e22f2afa392a20ed13640cc1ed5f4dd909dbfe31a
expected_candidate_build_provenance_sha=8041536b8d4f267544db223c5c800ce6ac4467a97782ccb302f2bed4d485b509
expected_common_dataset_manifest_sha=df793f964077ca03de7ad4723975736d0d0fb85d7e548241f3a7a04677cc05f7
expected_common_output_manifest_sha=3b7811c12df23a563e597cae2fb80f592c045d27c803a1afc7ead31d7d7439d4
expected_common_provenance_sha=54e2a4b0f07883f4341d46e41a80fd8ae1ebac7e25d263699eda8819b00ce0d8
expected_common_contract_sha=0dd83763717853fb64960f56354a2d7b7acd9c6759adcdff49dac9936635f110
expected_gate4b_output_sha=7948d59e04230faa6dd56457141892fe3c11757095833ffbd7672d8f20255b4a
expected_gate4b_provenance_sha=4c395471826bfb7bcc9349cf61a6f219a6bf5998475f26c83ace26f74340e8fa
expected_comparator_sha=56c8d808f5f4ec220b4ce57c4afbdae01d64065d5a327386a2cb2c4c69f67611
expected_comparator_test_sha=785e0c50edfb88160c60718a923774448f836828f9449859770f4c971a500b32
expected_residual_sha=638f1195ba6a639f5750eb7c6f78911daffb76a42222f4a1c0f215768d163bba
expected_residual_test_sha=2d2f19da10b6e42c1c2e108e9f092a59e2db757b63c0a8c8f0d8eacc960bd4e4
expected_closure_sha=4a5de94e6dbf590dded4a6ecd140aa4227fa61ffa0e73af17ec0f388610cffaf
expected_closure_test_sha=38de02fabc0dde41911e5152b0b08a9987b312b0cea29c69396bb9e392e9c745
expected_fixed_sha=569ecb1366bc6dd4584216bd42ac55905a4848b1a0b1bdf96c6236c782de7cb2
expected_initial_sha=6bbade9eaeb207b6cea9fa2f80d8cbcd0baeb8cbd760a2ffab5a0fe6f6d4868a
expected_initial_test_sha=82634e292a8fc1eb5ed454360ea2367e06687f0547da6503291c850a00e5b339

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
mkdir -p "$old_run" "$current_run" "$tool_dir"
printf 'preflight=started\n' >"$run_root/PREFLIGHT.txt"
test -e "$common_root/COMPLETE"
test -e "$gate4b_root/COMPLETE"
test -e "$candidate_build/COMPLETE"
test -x "$old_exe"
test -x "$candidate_exe"
printf 'preflight=sentinels_present\n' >>"$run_root/PREFLIGHT.txt"

while read -r path expected; do
  observed=$(sha256sum "$path" | awk '{print $1}')
  if [[ $observed != "$expected" ]]; then
    printf 'hash_mismatch path=%s expected=%s observed=%s\n' \
      "$path" "$expected" "$observed" >&2
    exit 1
  fi
done <<EOF
$old_exe $expected_old_exe_sha
$old_build/build-provenance.txt $expected_old_build_provenance_sha
$candidate_exe $expected_candidate_exe_sha
$candidate_build/SHA256SUMS.txt $expected_candidate_build_manifest_sha
$candidate_build/build-provenance.txt $expected_candidate_build_provenance_sha
$common_root/DATASET_SHA256SUMS.txt $expected_common_dataset_manifest_sha
$common_root/OUTPUT_SHA256SUMS.txt $expected_common_output_manifest_sha
$common_root/PROVENANCE.txt $expected_common_provenance_sha
$input_dir/qsgw_input.contract $expected_common_contract_sha
$gate4b_root/OUTPUT_SHA256SUMS.txt $expected_gate4b_output_sha
$gate4b_root/PROVENANCE.txt $expected_gate4b_provenance_sha
$comparator_source $expected_comparator_sha
$comparator_test_source $expected_comparator_test_sha
$residual_source $expected_residual_sha
$residual_test_source $expected_residual_test_sha
$closure_source $expected_closure_sha
$closure_test_source $expected_closure_test_sha
$fixed_source $expected_fixed_sha
$initial_source $expected_initial_sha
$initial_test_source $expected_initial_test_sha
EOF
printf 'preflight=hashes_verified\n' >>"$run_root/PREFLIGHT.txt"

grep -Fqx "old_oracle_commit=$expected_old_commit" \
  "$old_build/build-provenance.txt"
grep -Fqx "source_commit=$expected_candidate_commit" \
  "$candidate_build/build-provenance.txt"
grep -Fqx 'source_status=clean_linux_git_archive_from_commit' \
  "$candidate_build/build-provenance.txt"
grep -Fqx 'fish_gate0_result=59_of_59_pass' \
  "$candidate_build/build-provenance.txt"
grep -Fqx 'acceptance_gate=true' "$gate4b_root/PROVENANCE.txt"
grep -Fqx 'result=PASS' "$gate4b_root/PROVENANCE.txt"
printf 'preflight=provenance_verified\n' >>"$run_root/PREFLIGHT.txt"
grep -Fqx 'n_scf_kpoints 64' "$input_dir/qsgw_input.contract"
grep -Fqx 'n_headwing_kpoints 0' "$input_dir/qsgw_input.contract"
grep -Fqx 'headwing_grid disabled' "$input_dir/qsgw_input.contract"
grep -Fqx 'hartree_update off' "$input_dir/qsgw_input.contract"
grep -Fqx 'band_update off' "$input_dir/qsgw_input.contract"
test "$(awk 'NR == 2 {print $1, $2}' "$input_dir/bz_sampling_out")" = '64 64'
test "$(find "$input_dir" -maxdepth 1 -name 'KS_eigenvector_*.dat' | wc -l)" -eq 64
test "$(find "$input_dir" -maxdepth 1 -type f -name 'vxcs1k*_nao.txt' | wc -l)" -eq 64
test "$(find "$input_dir" -maxdepth 1 -type f -name 'sks1k*_nao.txt' | wc -l)" -eq 64
test "$(find "$input_dir" -type l | wc -l)" -eq 0
printf 'preflight=contracts_verified\n' >>"$run_root/PREFLIGHT.txt"

mkdir -p "$old_run" "$current_run" "$tool_dir"
cp "$0" "$run_root/gate5a-linear-beta02-miniter5-a25e4e43-v5.slurm"
cp "$old_build/build-provenance.txt" "$run_root/old-build-provenance.txt"
cp "$candidate_build/build-provenance.txt" "$run_root/candidate-build-provenance.txt"
cp "$candidate_build/SHA256SUMS.txt" "$run_root/candidate-build-SHA256SUMS.txt"
cp "$common_root/DATASET_SHA256SUMS.txt" "$run_root/"
cp "$common_root/OUTPUT_SHA256SUMS.txt" "$run_root/common-input-OUTPUT_SHA256SUMS.txt"
cp "$common_root/PROVENANCE.txt" "$run_root/common-input-PROVENANCE.txt"
cp "$input_dir/qsgw_input.contract" "$run_root/"
cp "$gate4b_root/OUTPUT_SHA256SUMS.txt" "$run_root/gate4b-OUTPUT_SHA256SUMS.txt"
cp "$gate4b_root/PROVENANCE.txt" "$run_root/gate4b-PROVENANCE.txt"
cp "$comparator_source" "$tool_dir/compare_qsgw_component_traces.py"
cp "$comparator_source" "$tool_dir/compare_qsgw_component_traces_v3.py"
cp "$comparator_test_source" "$tool_dir/test_compare_qsgw_legacy_v4_current_v5_v3.py"
cp "$residual_source" "$tool_dir/summarize_qsgw_linear_residuals.py"
cp "$residual_test_source" "$tool_dir/test_summarize_qsgw_linear_residuals.py"
cp "$closure_source" "$tool_dir/validate_qsgw_trace_closure.py"
cp "$closure_source" "$tool_dir/validate_qsgw_trace_closure_v3.py"
cp "$closure_test_source" "$tool_dir/test_validate_qsgw_trace_closure_v3.py"
cp "$fixed_source" "$tool_dir/validate_qsgw_fixed_basis.py"
cp "$initial_source" "$tool_dir/validate_qsgw_initial_state.py"
cp "$initial_test_source" "$tool_dir/test_validate_qsgw_initial_state.py"

cat >"$old_run/librpa.in" <<EOF
task = qsgw
input_dir = $input_dir
output_dir = .
nfreq = 6
tfgrid_type = minimax
n_params_anacon = -1
parallel_routing = libri
vq_threshold = 0
sqrt_coulomb_threshold = 0
use_scalapack_gw_wc = true
use_shrink_abfs = false
use_shrink_chi = false
use_pyatb = false
replace_w_head = false
option_dielect_func = 0
use_fullcoul_exx = false
use_fullcoul_wc = false
use_abacus_exx_symmetry = false
use_abacus_gw_symmetry = false
qsgw_iterative_headwing = false
max_iter = $target_iter
EOF

cat >"$current_run/librpa.in" <<EOF
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
qsgw_mixer = linear
qsgw_mixing_beta = $mixing_beta
qsgw_min_iter = $target_iter
qsgw_max_iter = $target_iter
qsgw_write_iteration_matrices = true
qsgw_update_hartree = false
EOF

cat >"$run_root/PARAMETER_MAPPING.txt" <<EOF
comparison=legacy_v4_to_current_v5
iterations=0:$target_iter
old_mixing=linear_beta_$mixing_beta
current_mixing=linear_beta_$mixing_beta
old_hartree=off
current_hartree=off
old_band=off
current_band=off
old_headwing=off
current_headwing=off
old_symmetry=off_full_bz_64
current_symmetry=off_full_bz_64
common_input=$input_dir
common_input_manifest_sha256=$expected_common_dataset_manifest_sha
common_contract_sha256=$expected_common_contract_sha
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
base_ld_library_path=${LD_LIBRARY_PATH:-}

cat >"$run_root/PROVENANCE.txt" <<EOF
gate=gate5a-linear-mixing-beta02-miniter5
final_acceptance=false_pending_observers
old_commit=$expected_old_commit
old_executable=$old_exe
old_executable_sha256=$expected_old_exe_sha
current_commit=$expected_candidate_commit
current_executable=$candidate_exe
current_executable_sha256=$expected_candidate_exe_sha
common_input=$input_dir
common_input_manifest_sha256=$expected_common_dataset_manifest_sha
common_contract_sha256=$expected_common_contract_sha
gate4b_dependency_job=2348159
gate4b_output_manifest_sha256=$expected_gate4b_output_sha
target_iteration=$target_iter
mixing=legacy_linear_beta_$mixing_beta,current_linear_beta_$mixing_beta
crystal_symmetry=off
time_reversal_reduction=off_full_64_kpoint_input
headwing=off
hartree=off
band=off
mpi_ranks=${SLURM_NTASKS:-unknown}
omp_threads=$OMP_NUM_THREADS
LIBRI_DETERMINISTIC_REDUCTION=$LIBRI_DETERMINISTIC_REDUCTION
python=$($python --version 2>&1)
numpy=$($python -c 'import numpy; print(numpy.__version__)')
slurm_job_id=$SLURM_JOB_ID
slurm_node_list=$SLURM_JOB_NODELIST
started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
EOF
sha256sum "$old_run/librpa.in" "$current_run/librpa.in" \
  "$input_dir/band_out" "$input_dir/stru_out" \
  "$input_dir/vxc_out.dat" "$input_dir/qsgw_input.contract" \
  >>"$run_root/PROVENANCE.txt"

PYTHONPATH="$tool_dir" "$python" -B \
  "$tool_dir/test_compare_qsgw_legacy_v4_current_v5_v3.py" \
  >"$run_root/comparator-unit-test.stdout" \
  2>"$run_root/comparator-unit-test.stderr"
PYTHONPATH="$tool_dir" "$python" -B \
  "$tool_dir/test_summarize_qsgw_linear_residuals.py" \
  >"$run_root/residual-unit-test.stdout" \
  2>"$run_root/residual-unit-test.stderr"
PYTHONPATH="$tool_dir" "$python" -B \
  "$tool_dir/test_validate_qsgw_trace_closure_v3.py" \
  >"$run_root/closure-unit-test.stdout" \
  2>"$run_root/closure-unit-test.stderr"
PYTHONPATH="$tool_dir" "$python" -B \
  "$tool_dir/test_validate_qsgw_initial_state.py" \
  >"$run_root/initial-state-unit-test.stdout" \
  2>"$run_root/initial-state-unit-test.stderr"

(
  cd "$old_run"
  export QSGW_ORACLE_TRACE=1
  export QSGW_ORACLE_UPDATE_HARTREE=0
  export QSGW_HROUND_SCALE=0
  export LIBRPA_QSGW_MIXING_BETA=$mixing_beta
  export LD_LIBRARY_PATH="$old_build/old/build/qsgw:$old_build/old/build/src:$base_ld_library_path"
  mpirun -np "$SLURM_NTASKS" "$old_exe" >librpa.stdout 2>librpa.stderr
)

test -s "$old_run/qsgw_oracle_matrices.dat"
test -s "$old_run/homo_lumo_vs_iterations.dat"
grep -Fqx '# qsgw_contract_version 4' "$old_run/qsgw_oracle_matrices.dat"
grep -Fqx '# qsgw_mixer linear' "$old_run/qsgw_oracle_matrices.dat"
grep -Fqx '# qsgw_update_hartree 0' "$old_run/qsgw_oracle_matrices.dat"
grep -Fqx '# use_symmetry_gw 0' "$old_run/qsgw_oracle_matrices.dat"
grep -Fqx '# use_symmetry_exx 0' "$old_run/qsgw_oracle_matrices.dat"
grep -Fqx '# replace_w_head 0' "$old_run/qsgw_oracle_matrices.dat"
grep -Fqx '# option_dielect_func 0' "$old_run/qsgw_oracle_matrices.dat"
test "$(awk 'NF && $1 !~ /^#/ {last=$1} END {print last}' \
  "$old_run/homo_lumo_vs_iterations.dat")" = "$target_iter"

(
  cd "$current_run"
  unset QSGW_ORACLE_TRACE QSGW_ORACLE_UPDATE_HARTREE QSGW_HROUND_SCALE
  unset LIBRPA_QSGW_MIXING_BETA
  export LD_LIBRARY_PATH="$base_ld_library_path"
  mpirun -np "$SLURM_NTASKS" "$candidate_exe" >librpa.stdout 2>librpa.stderr
)

test -s "$current_run/qsgw_matrices.dat"
test -s "$current_run/qsgw_eigenvalues.dat"
test -s "$current_run/qsgw_iterations.dat"
for trace in "$current_run/qsgw_matrices.dat" \
  "$current_run/qsgw_eigenvalues.dat" "$current_run/qsgw_iterations.dat"
do
  grep -Fqx '# qsgw_contract_version 5' "$trace"
  grep -Fqx '# fixed_basis immutable_mf0' "$trace"
  grep -Fqx '# live_update eigenvalues_wfc' "$trace"
  grep -Fqx '# symmetry unsupported_full_bz_only' "$trace"
  grep -Fqx '# headwing disabled_stage1' "$trace"
  grep -Fqx '# hartree disabled_stage1' "$trace"
  grep -Fqx '# band disabled_stage1' "$trace"
  grep -Fqx '# qsgw_mixer linear' "$trace"
  grep -Fqx "# qsgw_input_contract_sha256 $expected_common_contract_sha" "$trace"
done

"$python" - "$current_run/qsgw_iterations.dat" \
  "$run_root/current-linear-convergence.tsv" "$target_iter" "$mixing_beta" <<'PY'
import math
import pathlib
import sys

source = pathlib.Path(sys.argv[1])
target = pathlib.Path(sys.argv[2])
target_iter = int(sys.argv[3])
beta = float(sys.argv[4])
rows = []
for line_number, line in enumerate(source.read_text().splitlines(), 1):
    if not line.strip() or line.lstrip().startswith("#"):
        continue
    fields = line.split()
    if len(fields) < 16:
        raise SystemExit(f"iteration trace line {line_number} has too few fields")
    iteration = int(fields[0])
    numeric = [float(fields[index]) for index in range(1, 7)]
    if not all(math.isfinite(value) for value in numeric):
        raise SystemExit(f"iteration trace line {line_number} is non-finite")
    requested, applied = int(fields[7]), int(fields[8])
    observed_beta, fallback = float(fields[9]), int(fields[10])
    if iteration == 0:
        if (requested, applied) != (-1, -1):
            raise SystemExit("iteration zero unexpectedly has a mixing decision")
    elif (requested, applied) != (0, 0):
        raise SystemExit(f"iteration {iteration} is not linear mixing")
    if not math.isclose(observed_beta, beta, rel_tol=0.0, abs_tol=1.0e-15):
        raise SystemExit(f"iteration {iteration} beta mismatch")
    if fallback != 0:
        raise SystemExit(f"iteration {iteration} unexpectedly fell back")
    rows.append((iteration, *numeric, requested, applied, observed_beta, fallback))
if [row[0] for row in rows] != list(range(target_iter + 1)):
    raise SystemExit("iteration trace does not contain exactly 0..target_iter")
header = (
    "iter\tmax_delta_eV\tresidual_l2_Ha\tresidual_max_Ha\t"
    "efermi_eV\tgap_eV\telectron_count\trequested_mode\t"
    "applied_mode\tbeta\tfallback\n"
)
target.write_text(header + "".join("\t".join(map(str, row)) + "\n" for row in rows))
PY

PYTHONPATH="$tool_dir" "$python" -B \
  "$tool_dir/compare_qsgw_component_traces.py" \
  "$old_run/qsgw_oracle_matrices.dat" \
  "$current_run/qsgw_matrices.dat" \
  "$run_root/old-new-comparison.json" \
  --iterations "0:$target_iter" \
  --channel 0 \
  --contract-mode legacy_v4_to_current_v5 \
  --current-eigenvalue-trace "$current_run/qsgw_eigenvalues.dat" \
  --current-iteration-trace "$current_run/qsgw_iterations.dat" \
  --expected-legacy-use-fullcoul-exx 0 \
  --expected-mixing-beta "$mixing_beta" \
  --expected-current-mixer linear \
  --frequency-tolerance 1e-10 \
  --matrix-max-abs-tolerance-ha 1e-8 \
  --matrix-relative-tolerance 1e-8 \
  --eigenvalue-tolerance 1e-6 \
  --gap-tolerance-ev 1e-5 \
  --degeneracy-tolerance 1e-8 \
  --state-tolerance 1e-10 \
  >"$run_root/old-new-comparator.stdout" \
  2>"$run_root/old-new-comparator.stderr"
grep -Fq '"passed": true' "$run_root/old-new-comparison.json"

PYTHONPATH="$tool_dir" "$python" -B \
  "$tool_dir/summarize_qsgw_linear_residuals.py" \
  "$old_run/qsgw_oracle_matrices.dat" \
  "$current_run/qsgw_matrices.dat" \
  "$current_run/qsgw_iterations.dat" \
  "$run_root/linear-residual-comparison.json" \
  "$run_root/linear-residual-trajectory.csv" \
  --iterations "0:$target_iter" \
  --absolute-tolerance-ha 1e-8 \
  --relative-tolerance 1e-8 \
  >"$run_root/residual-observer.stdout" \
  2>"$run_root/residual-observer.stderr"
grep -Fq '"passed": true' "$run_root/linear-residual-comparison.json"

PYTHONPATH="$tool_dir" "$python" -B \
  "$tool_dir/validate_qsgw_initial_state.py" \
  "$current_run/qsgw_matrices.dat" \
  "$current_run/qsgw_iterations.dat" \
  "$input_dir/band_out" \
  "$run_root/current-initial-state-validation.json" \
  --efermi-tolerance-ha 1e-12 \
  --occupation-tolerance 1e-12 \
  >"$run_root/current-initial-state-validator.stdout" \
  2>"$run_root/current-initial-state-validator.stderr"
grep -Fq '"passed": true' "$run_root/current-initial-state-validation.json"

PYTHONPATH="$tool_dir" "$python" -B \
  "$tool_dir/validate_qsgw_trace_closure.py" \
  "$old_run/qsgw_oracle_matrices.dat" \
  "$run_root/old-closure-validation.json" \
  --iterations "0:$target_iter" --channel 0 --legacy-contract \
  --closure-tolerance-ha 1e-10 --hermiticity-tolerance-ha 1e-10 \
  >"$run_root/old-closure-validator.stdout" \
  2>"$run_root/old-closure-validator.stderr"
grep -Fq '"passed": true' "$run_root/old-closure-validation.json"

PYTHONPATH="$tool_dir" "$python" -B \
  "$tool_dir/validate_qsgw_trace_closure.py" \
  "$current_run/qsgw_matrices.dat" \
  "$run_root/current-closure-validation.json" \
  --iterations "0:$target_iter" --channel 0 \
  --closure-tolerance-ha 1e-10 --hermiticity-tolerance-ha 1e-10 \
  >"$run_root/current-closure-validator.stdout" \
  2>"$run_root/current-closure-validator.stderr"
grep -Fq '"passed": true' "$run_root/current-closure-validation.json"

"$python" - "$run_root/old-closure-validation.json" \
  "$run_root/current-closure-validation.json" <<'PY'
import json
import sys
for path in sys.argv[1:]:
    report = json.load(open(path))
    value = report["linear_mixing_max_abs_ha"]
    if value is None or not value <= 1.0e-10:
        raise SystemExit(f"linear mixing recurrence did not close: {path}")
PY

PYTHONPATH="$tool_dir" "$python" -B \
  "$tool_dir/validate_qsgw_fixed_basis.py" \
  "$current_run/qsgw_matrices.dat" \
  "$current_run/qsgw_eigenvalues.dat" \
  "$input_dir/band_out" \
  "$run_root/current-fixed-basis-validation.json" \
  --iterations "0:$target_iter" --channel 0 \
  --eigenvalue-tolerance-ha 1e-10 --invariant-tolerance 1e-10 \
  >"$run_root/current-fixed-basis-validator.stdout" \
  2>"$run_root/current-fixed-basis-validator.stderr"
grep -Fq '"passed": true' "$run_root/current-fixed-basis-validation.json"

printf 'final_acceptance=true\ncompleted_utc=%s\n' \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >>"$run_root/PROVENANCE.txt"
(
  cd "$run_root"
  find old -maxdepth 1 -type f -print0 | sort -z | xargs -0 sha256sum \
    >OLD_OUTPUT_SHA256SUMS.txt
  find current -maxdepth 1 -type f -print0 | sort -z | xargs -0 sha256sum \
    >CURRENT_OUTPUT_SHA256SUMS.txt
  sha256sum --check --quiet OLD_OUTPUT_SHA256SUMS.txt
  sha256sum --check --quiet CURRENT_OUTPUT_SHA256SUMS.txt
  find . -type f ! -name OUTPUT_SHA256SUMS.txt ! -name COMPLETE \
    ! -name FAILED -print0 | sort -z | xargs -0 sha256sum \
    >OUTPUT_SHA256SUMS.txt
  sha256sum --check --quiet OUTPUT_SHA256SUMS.txt
)
touch "$run_root/COMPLETE"
echo GATE5A_LINEAR_BETA02_MINITER5=PASS
