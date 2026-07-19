#!/usr/bin/env bash
# Gate 7A: first end-to-end qsgw_band run of the new implementation
# (commit ad9fd6ed) on the frozen Si k444 full-BZ dataset with the frozen
# band input contract (job 2356667). No Hartree, no head-wing (phase-1
# scope). Acceptance (fail-closed):
#   A. observer unit tests
#   B. trace contract headers (v5, band operator_fourier, hartree/head off)
#   C. iteration-0 band-channel initial state vs band_KS_eigenvalue
#      (eigenvalues + occupations, <= 1e-12)
#   D. iteration-0 grid initial state vs band_out (<= 1e-12)
#   E. v4 closure on the grid channel (0:2, Hartree off)
#   F. fixed-basis validation on the grid channel (0:2)
# The band-channel closure is deferred: the v5 stage-one trace contract
# does not declare a task field, so the closure observer rejects channel 1
# regardless of physics (observer-side gap; band external consistency is
# covered by the initial-state observer here and by the upstream g0w0_band
# comparison in Gate 7B).
#SBATCH --job-name=qsgw-g7a-band
#SBATCH --partition=v100m3
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:8
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --time=03:00:00
#SBATCH --output=/data/home/df_iopcas_bhj/ai-runs/librpa-qsgw-rebase-gates-20260715T1910-7d69a18c/logs/gate7a-qsgw-band-%j.out
#SBATCH --error=/data/home/df_iopcas_bhj/ai-runs/librpa-qsgw-rebase-gates-20260715T1910-7d69a18c/logs/gate7a-qsgw-band-%j.err

set -euo pipefail

target_iter=2
mixing_beta=0.2
root=/data/home/df_iopcas_bhj/ai-runs/librpa-qsgw-rebase-gates-20260715T1910-7d69a18c
input_root=$root/inputs/si-k444-band-v1-2356667
input_dir=$input_root/dataset
candidate_build=$root/build-candidate-ad9fd6ed-v1
candidate_exe=$candidate_build/chi0_main.exe
band_state_observer=$root/jobs/validate_qsgw_band_initial_state-v1-45d350bc.py
band_state_observer_test=$root/jobs/test_validate_qsgw_band_initial_state-v1-7af1032b.py
closure_v4=$root/jobs/validate_qsgw_trace_closure-v4-860e8bcd.py
comparator_module=$root/jobs/compare_qsgw_component_traces-v6-hartree-1715ae0a.py
fixed_basis=$root/jobs/validate_qsgw_fixed_basis.py
initial_state=$root/jobs/validate_qsgw_initial_state-v1.py
python=/data/home/df_iopcas_bhj/software-stack/miniconda3/bin/python3
run_root=$root/runs/gate7a-qsgw-band-ad9fd6ed-miniter2-$SLURM_JOB_ID
current_run=$run_root/current
tool_dir=$run_root/tools

expected_candidate_commit=ad9fd6ed65552b5ad2b3e80483aaf56538d15805
expected_candidate_exe_sha=74648f0ccf0ab84bec2c096a57b1ef03187d0e701027b0b2abea6183df2e46da
expected_input_manifest_sha=b279a3d2eea04e8ea65245277a2fee668e84fa1a1e229fb8d006fa0af04ba79a
expected_band_contract_sha=765c42f124c1c1344780a3499562764a5f4c05dc37ff6db17555663fd68274c9
expected_band_manifest_sha=fcc8f5841790ccb4ba29bd787d9500b76a6dfc85699c2ed79ff317c88d08b60a
expected_band_state_observer_sha=45d350bc4cc15295db196d86e6a63bafcdc6e88ada05cab21051e58da7683d74
expected_band_state_observer_test_sha=7af1032b5eeb3e91c9b54b2617cf47602e6da4bfeebecd9201bcff1a52ec7ed2
expected_closure_v4_sha=860e8bcd3c641ebab510d5b41ece95992bb960611855525772a48d58559ca3d5
expected_comparator_module_sha=1715ae0a7029bde53599c646c35259381bd1430a847678bd13c8f9600778718f
expected_fixed_basis_sha=569ecb1366bc6dd4584216bd42ac55905a4848b1a0b1bdf96c6236c782de7cb2
expected_initial_state_sha=6bbade9eaeb207b6cea9fa2f80d8cbcd0baeb8cbd760a2ffab5a0fe6f6d4868a

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
mkdir -p "$current_run" "$tool_dir"
test -e "$input_root/COMPLETE"
test -e "$candidate_build/COMPLETE"

while read -r path expected; do
  observed=$(sha256sum "$path" | awk '{print $1}')
  if [[ $observed != "$expected" ]]; then
    printf 'hash_mismatch path=%s expected=%s observed=%s\n' \
      "$path" "$expected" "$observed" >&2
    exit 1
  fi
done <<EOF
$candidate_exe $expected_candidate_exe_sha
$input_root/DATASET_SHA256SUMS.txt $expected_input_manifest_sha
$input_dir/qsgw_input.band.contract $expected_band_contract_sha
$input_dir/qsgw_vxc_band.v2.manifest $expected_band_manifest_sha
$band_state_observer $expected_band_state_observer_sha
$band_state_observer_test $expected_band_state_observer_test_sha
$closure_v4 $expected_closure_v4_sha
$comparator_module $expected_comparator_module_sha
$fixed_basis $expected_fixed_basis_sha
$initial_state $expected_initial_state_sha
EOF
grep -Fqx "source_commit=$expected_candidate_commit" \
  "$candidate_build/build-provenance.txt"
grep -Fqx 'fish_gate0_result=60_of_60_pass' \
  "$candidate_build/build-provenance.txt"
(
  cd "$input_dir"
  sha256sum --check --quiet DATASET_SHA256SUMS.txt
)

cp "$0" "$run_root/gate7a-qsgw-band-ad9fd6ed-miniter2-v1.slurm"
cp "$candidate_build/build-provenance.txt" "$run_root/candidate-build-provenance.txt"
cp "$input_root/PROVENANCE.txt" "$run_root/input-PROVENANCE.txt"
cp "$input_root/DATASET_SHA256SUMS.txt" "$run_root/input-DATASET_SHA256SUMS.txt"
cp "$input_dir/qsgw_input.band.contract" "$run_root/"
cp "$input_dir/qsgw_vxc_band.v2.manifest" "$run_root/"
cp "$band_state_observer" "$tool_dir/validate_qsgw_band_initial_state_v1.py"
cp "$band_state_observer_test" \
  "$tool_dir/test_validate_qsgw_band_initial_state_v1.py"
cp "$closure_v4" "$tool_dir/validate_qsgw_trace_closure.py"
cp "$closure_v4" "$tool_dir/validate_qsgw_trace_closure_v4.py"
cp "$comparator_module" "$tool_dir/compare_qsgw_component_traces.py"
cp "$fixed_basis" "$tool_dir/validate_qsgw_fixed_basis.py"
cp "$initial_state" "$tool_dir/validate_qsgw_initial_state.py"

cat >"$current_run/librpa.in" <<EOF
task = qsgw_band
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
qsgw_input_contract = qsgw_input.band.contract
qsgw_mixer = linear
qsgw_mixing_beta = $mixing_beta
qsgw_min_iter = $target_iter
qsgw_max_iter = $target_iter
qsgw_write_iteration_matrices = true
qsgw_update_hartree = false
EOF

band_contract_sha=$(sha256sum "$input_dir/qsgw_input.band.contract" \
  | awk '{print $1}')

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
gate=gate7a-qsgw-band-ad9fd6ed-miniter2
final_acceptance=false_pending_observers
result=PENDING
candidate_commit=$expected_candidate_commit
candidate_executable_sha256=$expected_candidate_exe_sha
band_input_root=$input_root
band_input_dataset_manifest_sha256=$expected_input_manifest_sha
band_contract_sha256=$band_contract_sha
target_iteration=$target_iter
mixing=linear_beta_$mixing_beta
use_shrink_abfs=false
crystal_symmetry=off
headwing=off
hartree=off
band_update=operator_fourier
n_band_kpoints=201
mpi_ranks=${SLURM_NTASKS:-unknown}
omp_threads=$OMP_NUM_THREADS
LIBRI_DETERMINISTIC_REDUCTION=$LIBRI_DETERMINISTIC_REDUCTION
slurm_job_id=$SLURM_JOB_ID
slurm_node_list=$SLURM_JOB_NODELIST
started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
EOF

(
  cd "$current_run"
  unset QSGW_ORACLE_TRACE QSGW_ORACLE_UPDATE_HARTREE QSGW_HROUND_SCALE
  unset LIBRPA_QSGW_MIXING_BETA LIBRPA_QSGW_HARTREE_DUMP_DIR
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
  grep -Fqx '# band fixed_reference_operator_fourier_live' "$trace"
  grep -Fqx '# qsgw_mixer linear' "$trace"
  grep -Fqx "# qsgw_input_contract_sha256 $band_contract_sha" "$trace"
done

# ---- Acceptance A: observer unit tests ----
PYTHONPATH="$tool_dir" "$python" -B \
  "$tool_dir/test_validate_qsgw_band_initial_state_v1.py" \
  >"$run_root/band-state-unit-test.stdout" \
  2>"$run_root/band-state-unit-test.stderr"

# ---- Acceptance C: iteration-0 band initial state ----
awk '$3 == "occupation" && $2 == 1 && $1 == 0 {print $1, $2, $4, $5, 0, 0, 0, $9, $10}' \
  "$current_run/qsgw_matrices.dat" \
  >"$run_root/band-occupation-iter0.txt"
PYTHONPATH="$tool_dir" "$python" -B \
  "$tool_dir/validate_qsgw_band_initial_state_v1.py" \
  "$current_run/qsgw_eigenvalues.dat" \
  "$input_dir" \
  201 \
  "$run_root/band-initial-state-validation.json" \
  --iterations 0 \
  --eigenvalue-tolerance-ha 1e-12 \
  --occupation-tolerance 1e-12 \
  --occupation-trace "$run_root/band-occupation-iter0.txt" \
  >"$run_root/band-initial-state.stdout" \
  2>"$run_root/band-initial-state.stderr"
assert_passed "$run_root/band-initial-state-validation.json"

# ---- Acceptance D: iteration-0 grid initial state ----
PYTHONPATH="$tool_dir" "$python" -B \
  "$tool_dir/validate_qsgw_initial_state.py" \
  "$current_run/qsgw_matrices.dat" "$current_run/qsgw_iterations.dat" \
  "$input_dir/band_out" "$run_root/grid-initial-state-validation.json" \
  --efermi-tolerance-ha 1e-12 --occupation-tolerance 1e-12 \
  >"$run_root/grid-initial-state.stdout" \
  2>"$run_root/grid-initial-state.stderr"
assert_passed "$run_root/grid-initial-state-validation.json"

# ---- Acceptance E: v4 closure on the grid channel ----
PYTHONPATH="$tool_dir" "$python" -B \
  "$tool_dir/validate_qsgw_trace_closure_v4.py" \
  "$current_run/qsgw_matrices.dat" \
  "$run_root/grid-closure-validation.json" \
  --iterations 0:2 --channel 0 \
  --closure-tolerance-ha 1e-10 --hermiticity-tolerance-ha 1e-10 \
  >"$run_root/grid-closure.stdout" \
  2>"$run_root/grid-closure.stderr"
assert_passed "$run_root/grid-closure-validation.json"

# ---- Acceptance F: fixed basis on the grid channel ----
PYTHONPATH="$tool_dir" "$python" -B \
  "$tool_dir/validate_qsgw_fixed_basis.py" \
  "$current_run/qsgw_matrices.dat" "$current_run/qsgw_eigenvalues.dat" \
  "$input_dir/band_out" "$run_root/grid-fixed-basis-validation.json" \
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
echo GATE7A_QSGW_BAND=PASS
