#!/usr/bin/env bash
#SBATCH --job-name=qsgw-g6b7-vh-ab
#SBATCH --partition=v100m3
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:8
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --time=03:00:00
#SBATCH --output=/data/home/df_iopcas_bhj/ai-runs/librpa-qsgw-rebase-gates-20260715T1910-7d69a18c/logs/gate6b7-hartree-ab-%j.out
#SBATCH --error=/data/home/df_iopcas_bhj/ai-runs/librpa-qsgw-rebase-gates-20260715T1910-7d69a18c/logs/gate6b7-hartree-ab-%j.err

set -euo pipefail

target_iter=2
mixing_beta=0.2
root=/data/home/df_iopcas_bhj/ai-runs/librpa-qsgw-rebase-gates-20260715T1910-7d69a18c
gate5b_root=$root/runs/gate5b-linear-beta02-miniter10-a25e4e43-2348654
gate6a2_root=$root/runs/gate6a2-full-abfs-baseline-miniter2-a25e4e43-2348804
input_root=$root/inputs/si-k444-gate6b1-mpi1-full-abfs-hartree-contracts-2348832
input_dir=$input_root/dataset
old_build=/data/home/df_iopcas_bhj/qsgw-builds/legacy-hartree-full-reader-oracle-v40-20260717
old_exe=$old_build/build/chi0_main.exe
oracle_run=$root/runs/gate6b6-legacy-full-reader-hartree-on-mpi4-omp12-miniter2-2348987
oracle_recheck=$root/artifacts/gate6b6-output-manifest-recheck-2348987-2349011
candidate_build=$root/build-candidate-a25e4e43-v1
candidate_exe=$candidate_build/chi0_main.exe
comparator_source=$root/jobs/compare_qsgw_component_traces-v6-hartree-1715ae0a.py
comparator_test_source=$root/jobs/test_compare_qsgw_legacy_v4_current_v5-v6-hartree-576f860d.py
hartree_validator_source=$root/jobs/validate_qsgw_hartree_trace-v1-c0512c35.py
hartree_validator_test_source=$root/jobs/test_validate_qsgw_hartree_trace-v1-96cdfee1.py
residual_source=$root/jobs/summarize_qsgw_linear_residuals-638f1195.py
residual_test_source=$root/jobs/test_summarize_qsgw_linear_residuals-2d2f19da.py
closure_source=$root/jobs/validate_qsgw_trace_closure-v3-4a5de94e.py
closure_test_source=$root/jobs/test_validate_qsgw_trace_closure-v3-38de02fa.py
fixed_source=$root/jobs/validate_qsgw_fixed_basis.py
initial_source=$root/jobs/validate_qsgw_initial_state-v1.py
initial_test_source=$root/jobs/test_validate_qsgw_initial_state-v1.py
python=/data/home/df_iopcas_bhj/software-stack/miniconda3/bin/python3
run_root=$root/runs/gate6b7-hartree-on-oracle-v40-vs-a25e4e43-mpi4-omp12-miniter${target_iter}-$SLURM_JOB_ID
old_run=$oracle_run/old
current_run=$run_root/current
tool_dir=$run_root/tools

expected_old_commit=e08f4a130df7661e9ac355b9be45fb2bf9c3ed01
expected_old_exe_sha=413cbc964bd81e0245fde4d40b09c5d5edc257bd805972db4fe5929f22189e4e
expected_old_build_provenance_sha=ba1881eadc624395f620e123f162b0c263569f4c3b41ea7bb0b8f7dcffb8996e
expected_old_source_provenance_sha=0bcd2f55600ffa4fdfd8cbbfdcda56c93e388f1de0930f4217b97e216de74cae
expected_old_output_manifest_sha=1add0e5f772941f7a0edce2ce078ad52993d17f324fe86d1b19276947a27a6f2
expected_candidate_commit=a25e4e4398d96bccdf44df00bfaad31c5e6035d6
expected_candidate_exe_sha=2072fc09a44a5f246513be152a06832dccf777da04b3dd8911b70e0f9a31591e
expected_candidate_build_manifest_sha=e0377a567a3504dbc61ece4e22f2afa392a20ed13640cc1ed5f4dd909dbfe31a
expected_candidate_build_provenance_sha=8041536b8d4f267544db223c5c800ce6ac4467a97782ccb302f2bed4d485b509
expected_input_dataset_manifest_sha=6e183ba7e91eb03311836bf5d2e6f0bf2c209338679fae1fc9c9cc627423bba1
expected_input_provenance_sha=532d4106269f299a7346e2ba30178844e9cd82cd48439dc6b61a4459e46de2cb
expected_gate5b_output_sha=822a6ff869f222ac4596585cbc49c2ff2bbfb6bd84fb8c064404223bba725d77
expected_gate5b_provenance_sha=4a51ff4d9a0bad374d62d56daf15915152e5f1ba2b6569a3b4a67429815ed11b
expected_gate6a2_output_sha=5f877cb8729ee6dfe94f832de3406961b59031494a9b6c2937773901be871768
expected_gate6a2_provenance_sha=780541f86c8ab38daa59213efa4754963aadb7cf3d19d6624d534bcce75710a8
expected_gate6a2_input_dataset_manifest_sha=6e183ba7e91eb03311836bf5d2e6f0bf2c209338679fae1fc9c9cc627423bba1
expected_gate6a2_off_contract_sha=3ae9f9d514f47a4d2b87fd8709d77c10e690348763b44446be5b7074e00c0e17
expected_gate6a2_hartree_contract_sha=ea2d80953e1e03524f2c63f484a0cf333cbb59a2ed5c00f78eed141c8988edf6
expected_oracle_output_manifest_sha=11758525d80a1025eac9dbe35a6794d52444960899bff90482a2f344c327e78f
expected_oracle_provenance_sha=35be3d726dffb010c92158ba0847aeed4b06b460bec9d27a5dac8045c66c8ddf
expected_oracle_closure_sha=ae15401792cfbf028bb1964bcf8299a481c0a819f92b9b77604341d6eee5fbd9
expected_oracle_script_sha=ed6068b4e40122bbf1dae494340872f751467ee0a26e5976e3d14d2a33bd7ade
expected_recheck_output_manifest_sha=2d0649f62192d3c30f2c4f7fc311b7f6057e06aa99c9559709f80a7c260e0bc7
expected_recheck_provenance_sha=728cfa2ca1e73b2fe620b68172b63e09dded1d53aae8cf0996905ad5dd229e7d
expected_comparator_sha=1715ae0a7029bde53599c646c35259381bd1430a847678bd13c8f9600778718f
expected_comparator_test_sha=576f860db9a00e1653f4503e6845716b24d35ca425fa392eb96703faa1833065
expected_hartree_validator_sha=c0512c356d8a681cf2aa1ee5114c0b13a24e13b435a758610056b7c34536206a
expected_hartree_validator_test_sha=96cdfee1fdf055efb5f9069d726b3456ad94904d09f54378a16dfa8108b488fb
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
mkdir -p "$current_run" "$tool_dir"
printf 'preflight=started\n' >"$run_root/PREFLIGHT.txt"
test -e "$input_root/COMPLETE"
test -e "$gate5b_root/COMPLETE"
test -e "$gate6a2_root/COMPLETE"
test -e "$old_build/COMPLETE"
test -e "$oracle_run/COMPLETE"
test ! -e "$oracle_run/FAILED"
test -e "$oracle_recheck/COMPLETE"
test ! -e "$oracle_recheck/FAILED"
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
$old_build/source-provenance.txt $expected_old_source_provenance_sha
$old_build/OUTPUT_SHA256SUMS.txt $expected_old_output_manifest_sha
$candidate_exe $expected_candidate_exe_sha
$candidate_build/SHA256SUMS.txt $expected_candidate_build_manifest_sha
$candidate_build/build-provenance.txt $expected_candidate_build_provenance_sha
$input_root/DATASET_SHA256SUMS.txt $expected_input_dataset_manifest_sha
$input_root/PROVENANCE.txt $expected_input_provenance_sha
$input_dir/qsgw_input.off.contract $expected_gate6a2_off_contract_sha
$input_dir/qsgw_input.hartree.contract $expected_gate6a2_hartree_contract_sha
$oracle_run/OUTPUT_SHA256SUMS.txt $expected_oracle_output_manifest_sha
$oracle_run/PROVENANCE.txt $expected_oracle_provenance_sha
$oracle_run/old-closure-validation.json $expected_oracle_closure_sha
$oracle_run/gate6b6-legacy-full-reader-hartree-on-mpi4-omp12-miniter2-v2.slurm $expected_oracle_script_sha
$oracle_recheck/OUTPUT_SHA256SUMS.txt $expected_recheck_output_manifest_sha
$oracle_recheck/PROVENANCE.txt $expected_recheck_provenance_sha
$gate5b_root/OUTPUT_SHA256SUMS.txt $expected_gate5b_output_sha
$gate5b_root/PROVENANCE.txt $expected_gate5b_provenance_sha
$gate6a2_root/OUTPUT_SHA256SUMS.txt $expected_gate6a2_output_sha
$gate6a2_root/PROVENANCE.txt $expected_gate6a2_provenance_sha
$comparator_source $expected_comparator_sha
$comparator_test_source $expected_comparator_test_sha
$hartree_validator_source $expected_hartree_validator_sha
$hartree_validator_test_source $expected_hartree_validator_test_sha
$residual_source $expected_residual_sha
$residual_test_source $expected_residual_test_sha
$closure_source $expected_closure_sha
$closure_test_source $expected_closure_test_sha
$fixed_source $expected_fixed_sha
$initial_source $expected_initial_sha
$initial_test_source $expected_initial_test_sha
EOF
printf 'preflight=hashes_verified\n' >>"$run_root/PREFLIGHT.txt"

grep -Fqx 'legacy_oracle_adapter=true' \
  "$old_build/source-provenance.txt"
grep -Fqx \
  'adapter_scope=task_qsgw_cut_coulomb_reader_only_when_hartree_env_enabled' \
  "$old_build/source-provenance.txt"
grep -Fqx 'hartree_off_reader_behavior=unchanged' \
  "$old_build/source-provenance.txt"
grep -Fqx 'source_output_manifest_recheck=PASS' \
  "$oracle_recheck/PROVENANCE.txt"
grep -Fq '"passed": true' "$oracle_run/old-closure-validation.json"
grep -Fqx "source_commit=$expected_candidate_commit" \
  "$candidate_build/build-provenance.txt"
grep -Fqx 'source_status=clean_linux_git_archive_from_commit' \
  "$candidate_build/build-provenance.txt"
grep -Fqx 'final_acceptance=true' "$gate5b_root/PROVENANCE.txt"
grep -Fqx 'hartree=off' "$gate5b_root/PROVENANCE.txt"
grep -Fqx 'final_acceptance=true' "$gate6a2_root/PROVENANCE.txt"
grep -Fqx 'result=PASS' "$gate6a2_root/PROVENANCE.txt"
grep -Fqx 'hartree=off' "$gate6a2_root/PROVENANCE.txt"
grep -Fqx 'use_shrink_abfs=false' "$gate6a2_root/PROVENANCE.txt"
printf 'preflight=provenance_verified\n' >>"$run_root/PREFLIGHT.txt"

grep -Fqx 'n_scf_kpoints 64' "$input_dir/qsgw_input.off.contract"
grep -Fqx 'n_headwing_kpoints 0' "$input_dir/qsgw_input.off.contract"
grep -Fqx 'headwing_grid disabled' "$input_dir/qsgw_input.off.contract"
grep -Fqx 'hartree_update off' "$input_dir/qsgw_input.off.contract"
grep -Fqx 'band_update off' "$input_dir/qsgw_input.off.contract"
grep -Fqx 'hartree_update delta_density' \
  "$input_dir/qsgw_input.hartree.contract"
grep -Fq 'hartree_ri_coefficients ' \
  "$input_dir/qsgw_input.hartree.contract"
grep -Fq 'hartree_coulomb ' "$input_dir/qsgw_input.hartree.contract"
grep -Fq 'hartree_aux_basis ' "$input_dir/qsgw_input.hartree.contract"
test "$(grep -Ec '^hartree_ri_coefficients [0-9a-f]{64} Cs_data_[0-9]+\.txt$' \
  "$input_dir/qsgw_input.hartree.contract")" -eq 4
test "$(grep -Ec '^hartree_aux_basis [0-9a-f]{64} basis_aux_out$' \
  "$input_dir/qsgw_input.hartree.contract")" -eq 1
if grep -Eq '^hartree_(ri_coefficients|aux_basis) [0-9a-f]{64} (Cs_shrinked_data_|shrink_sinvS_)' \
  "$input_dir/qsgw_input.hartree.contract"
then
  echo 'Hartree contract incorrectly binds shrink files in full-ABF mode' >&2
  exit 1
fi
test "$(awk 'NR == 2 {print $1, $2}' "$input_dir/bz_sampling_out")" = '64 64'
test "$(find "$input_dir" -maxdepth 1 -name 'KS_eigenvector_*.dat' | wc -l)" -eq 64
test "$(find "$input_dir" -maxdepth 1 -type f -name 'Cs_shrinked_data_*.txt' | wc -l)" -eq 4
test "$(find "$input_dir" -maxdepth 1 -type f -name 'shrink_sinvS_*.txt' | wc -l)" -eq 4
test "$(find "$input_dir" -maxdepth 1 -type f -name 'Cs_data_*.txt' | wc -l)" -eq 4
test "$(find "$input_dir" -maxdepth 1 -type f -name 'coulomb_cut_*.txt' | wc -l)" -eq 4
test "$(find "$input_dir" -type l | wc -l)" -eq 0

(
  cd "$input_root"
  sha256sum --check --quiet DATASET_SHA256SUMS.txt
)
off_contract_sha=$(sha256sum "$input_dir/qsgw_input.off.contract" | awk '{print $1}')
hartree_contract_sha=$(sha256sum "$input_dir/qsgw_input.hartree.contract" | awk '{print $1}')
dataset_manifest_sha=$(sha256sum "$input_root/DATASET_SHA256SUMS.txt" | awk '{print $1}')
test "$dataset_manifest_sha" = "$expected_input_dataset_manifest_sha"
test "$off_contract_sha" = "$expected_gate6a2_off_contract_sha"
test "$hartree_contract_sha" = "$expected_gate6a2_hartree_contract_sha"
(
  cd "$oracle_run"
  sha256sum --check --quiet OLD_OUTPUT_SHA256SUMS.txt
)
printf 'preflight=input_contracts_frozen\n' >>"$run_root/PREFLIGHT.txt"

cp "$0" \
  "$run_root/gate6b7-hartree-on-oracle-v40-vs-a25e4e43-mpi4-omp12-miniter2-v1.slurm"
cp "$old_build/build-provenance.txt" "$run_root/old-build-provenance.txt"
cp "$old_build/source-provenance.txt" "$run_root/old-source-provenance.txt"
cp "$old_build/OUTPUT_SHA256SUMS.txt" "$run_root/old-build-OUTPUT_SHA256SUMS.txt"
cp "$candidate_build/build-provenance.txt" "$run_root/candidate-build-provenance.txt"
cp "$candidate_build/SHA256SUMS.txt" "$run_root/candidate-build-SHA256SUMS.txt"
cp "$input_root/DATASET_SHA256SUMS.txt" "$run_root/"
cp "$input_root/PROVENANCE.txt" "$run_root/input-PROVENANCE.txt"
cp "$input_root/OUTPUT_SHA256SUMS.txt" "$run_root/input-OUTPUT_SHA256SUMS.txt"
cp "$input_dir/qsgw_input.off.contract" "$run_root/"
cp "$input_dir/qsgw_input.hartree.contract" "$run_root/"
cp "$gate5b_root/OUTPUT_SHA256SUMS.txt" "$run_root/gate5b-OUTPUT_SHA256SUMS.txt"
cp "$gate5b_root/PROVENANCE.txt" "$run_root/gate5b-PROVENANCE.txt"
cp "$gate6a2_root/OUTPUT_SHA256SUMS.txt" "$run_root/gate6a2-OUTPUT_SHA256SUMS.txt"
cp "$gate6a2_root/PROVENANCE.txt" "$run_root/gate6a2-PROVENANCE.txt"
cp "$oracle_run/OUTPUT_SHA256SUMS.txt" "$run_root/oracle-OUTPUT_SHA256SUMS.txt"
cp "$oracle_run/OLD_OUTPUT_SHA256SUMS.txt" \
  "$run_root/oracle-OLD_OUTPUT_SHA256SUMS.txt"
cp "$oracle_run/PROVENANCE.txt" "$run_root/oracle-PROVENANCE.txt"
cp "$oracle_run/old-closure-validation.json" \
  "$run_root/oracle-old-closure-validation.json"
cp "$oracle_recheck/OUTPUT_SHA256SUMS.txt" \
  "$run_root/oracle-recheck-OUTPUT_SHA256SUMS.txt"
cp "$oracle_recheck/PROVENANCE.txt" "$run_root/oracle-recheck-PROVENANCE.txt"
cp "$comparator_source" "$tool_dir/compare_qsgw_component_traces.py"
cp "$comparator_source" "$tool_dir/compare_qsgw_component_traces_v3.py"
cp "$comparator_test_source" "$tool_dir/test_compare_qsgw_legacy_v4_current_v5_v3.py"
cp "$hartree_validator_source" "$tool_dir/validate_qsgw_hartree_trace_v1.py"
cp "$hartree_validator_test_source" "$tool_dir/test_validate_qsgw_hartree_trace_v1.py"
cp "$residual_source" "$tool_dir/summarize_qsgw_linear_residuals.py"
cp "$residual_test_source" "$tool_dir/test_summarize_qsgw_linear_residuals.py"
cp "$closure_source" "$tool_dir/validate_qsgw_trace_closure.py"
cp "$closure_source" "$tool_dir/validate_qsgw_trace_closure_v3.py"
cp "$closure_test_source" "$tool_dir/test_validate_qsgw_trace_closure_v3.py"
cp "$fixed_source" "$tool_dir/validate_qsgw_fixed_basis.py"
cp "$initial_source" "$tool_dir/validate_qsgw_initial_state.py"
cp "$initial_test_source" "$tool_dir/test_validate_qsgw_initial_state.py"

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
qsgw_input_contract = qsgw_input.hartree.contract
qsgw_mixer = linear
qsgw_mixing_beta = $mixing_beta
qsgw_min_iter = $target_iter
qsgw_max_iter = $target_iter
qsgw_write_iteration_matrices = true
qsgw_update_hartree = true
qsgw_hartree_coulomb = truncated
qsgw_hartree_normalization = legacy_extra_inverse_nk
EOF

cat >"$run_root/PARAMETER_MAPPING.txt" <<EOF
comparison=legacy_v4_to_current_v5
iterations=0:$target_iter
old_mixing=linear_beta_$mixing_beta
current_mixing=linear_beta_$mixing_beta
old_use_shrink_abfs=false
current_use_shrink_abfs=false
old_hartree=delta_density
current_hartree=delta_density
old_hartree_coulomb=truncated
current_hartree_coulomb=truncated
old_hartree_normalization=legacy_extra_inverse_nk
current_hartree_normalization=legacy_extra_inverse_nk
old_band=off
current_band=off
old_headwing=off
current_headwing=off
old_symmetry=off_full_bz_64
current_symmetry=off_full_bz_64
input=$input_dir
input_dataset_manifest_sha256=$dataset_manifest_sha
off_contract_sha256=$off_contract_sha
hartree_contract_sha256=$hartree_contract_sha
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
gate=gate6b7-hartree-on-oracle-v40-vs-a25e4e43-mpi4-omp12-miniter2
final_acceptance=false_pending_observers
result=PENDING
old_commit=$expected_old_commit
old_executable=$old_exe
old_executable_sha256=$expected_old_exe_sha
old_hartree_build=legacy_v40_full_reader_adapter
old_oracle_job=2348987
old_oracle_output_manifest_sha256=$expected_oracle_output_manifest_sha
old_oracle_recheck_job=2349011
old_oracle_recheck_manifest_sha256=$expected_recheck_output_manifest_sha
current_commit=$expected_candidate_commit
current_executable=$candidate_exe
current_executable_sha256=$expected_candidate_exe_sha
gate5b_job=2348654
gate5b_output_manifest_sha256=$expected_gate5b_output_sha
gate6a2_job=2348804
gate6a2_output_manifest_sha256=$expected_gate6a2_output_sha
input=$input_dir
input_dataset_manifest_sha256=$dataset_manifest_sha
off_contract_sha256=$off_contract_sha
hartree_contract_sha256=$hartree_contract_sha
target_iteration=$target_iter
mixing=legacy_linear_beta_$mixing_beta,current_linear_beta_$mixing_beta
use_shrink_abfs=false
crystal_symmetry=off
time_reversal_reduction=off_full_64_kpoint_input
headwing=off
hartree=delta_density
hartree_coulomb=truncated
hartree_normalization=legacy_extra_inverse_nk
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
  "$input_dir/qsgw_input.off.contract" \
  "$input_dir/qsgw_input.hartree.contract" \
  >>"$run_root/PROVENANCE.txt"

PYTHONPATH="$tool_dir" "$python" -B \
  "$tool_dir/test_compare_qsgw_legacy_v4_current_v5_v3.py" \
  >"$run_root/comparator-unit-test.stdout" \
  2>"$run_root/comparator-unit-test.stderr"
PYTHONPATH="$tool_dir" "$python" -B \
  "$tool_dir/test_validate_qsgw_hartree_trace_v1.py" \
  >"$run_root/hartree-validator-unit-test.stdout" \
  2>"$run_root/hartree-validator-unit-test.stderr"
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

test -s "$old_run/qsgw_oracle_matrices.dat"
test -s "$old_run/homo_lumo_vs_iterations.dat"
grep -Fqx '# qsgw_contract_version 4' "$old_run/qsgw_oracle_matrices.dat"
grep -Fqx '# qsgw_mixer linear' "$old_run/qsgw_oracle_matrices.dat"
grep -Fqx '# qsgw_update_hartree 1' "$old_run/qsgw_oracle_matrices.dat"
grep -Fqx '# use_shrink_abfs 0' "$old_run/qsgw_oracle_matrices.dat"
grep -Fqx '# use_symmetry_gw 0' "$old_run/qsgw_oracle_matrices.dat"
grep -Fqx '# use_symmetry_exx 0' "$old_run/qsgw_oracle_matrices.dat"
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
  grep -Fqx '# hartree delta_density' "$trace"
  grep -Fqx '# hartree_coulomb truncated' "$trace"
  grep -Fqx '# hartree_normalization legacy_extra_inverse_nk' "$trace"
  grep -Fqx '# band disabled_stage1' "$trace"
  grep -Fqx '# qsgw_mixer linear' "$trace"
  grep -Fqx "# qsgw_input_contract_sha256 $hartree_contract_sha" "$trace"
done

PYTHONPATH="$tool_dir" "$python" -B \
  "$tool_dir/compare_qsgw_component_traces.py" \
  "$old_run/qsgw_oracle_matrices.dat" \
  "$current_run/qsgw_matrices.dat" \
  "$run_root/old-new-comparison.json" \
  --iterations "0:$target_iter" --channel 0 \
  --contract-mode legacy_v4_to_current_v5 \
  --current-eigenvalue-trace "$current_run/qsgw_eigenvalues.dat" \
  --current-iteration-trace "$current_run/qsgw_iterations.dat" \
  --expected-legacy-use-fullcoul-exx 0 \
  --expected-use-shrink-abfs 0 \
  --expected-hartree 1 \
  --expected-mixing-beta "$mixing_beta" \
  --expected-current-mixer linear \
  --frequency-tolerance 1e-10 \
  --matrix-max-abs-tolerance-ha 1e-8 \
  --matrix-relative-tolerance 1e-8 \
  --eigenvalue-tolerance 1e-6 --gap-tolerance-ev 1e-5 \
  --degeneracy-tolerance 1e-8 --state-tolerance 1e-10 \
  >"$run_root/old-new-comparator.stdout" \
  2>"$run_root/old-new-comparator.stderr"
grep -Fq '"passed": true' "$run_root/old-new-comparison.json"

PYTHONPATH="$tool_dir" "$python" -B \
  "$tool_dir/validate_qsgw_hartree_trace_v1.py" \
  "$old_run/qsgw_oracle_matrices.dat" \
  "$current_run/qsgw_matrices.dat" \
  "$current_run/qsgw_iterations.dat" \
  "$input_dir/bz_sampling_out" \
  "$run_root/hartree-validation.json" \
  --iterations "0:$target_iter" --channel 0 \
  --initial-delta-tolerance-ha 1e-10 \
  --charge-tolerance 1e-10 \
  --matrix-max-abs-tolerance-ha 1e-8 \
  --matrix-relative-tolerance 1e-8 \
  --hermiticity-tolerance-ha 1e-10 \
  >"$run_root/hartree-validator.stdout" \
  2>"$run_root/hartree-validator.stderr"
grep -Fq '"passed": true' "$run_root/hartree-validation.json"

PYTHONPATH="$tool_dir" "$python" -B \
  "$tool_dir/summarize_qsgw_linear_residuals.py" \
  "$old_run/qsgw_oracle_matrices.dat" \
  "$current_run/qsgw_matrices.dat" \
  "$current_run/qsgw_iterations.dat" \
  "$run_root/linear-residual-comparison.json" \
  "$run_root/linear-residual-trajectory.csv" \
  --iterations "0:$target_iter" \
  --absolute-tolerance-ha 1e-8 --relative-tolerance 1e-8 \
  >"$run_root/residual-observer.stdout" \
  2>"$run_root/residual-observer.stderr"
grep -Fq '"passed": true' "$run_root/linear-residual-comparison.json"

PYTHONPATH="$tool_dir" "$python" -B \
  "$tool_dir/validate_qsgw_initial_state.py" \
  "$current_run/qsgw_matrices.dat" "$current_run/qsgw_iterations.dat" \
  "$input_dir/band_out" "$run_root/current-initial-state-validation.json" \
  --efermi-tolerance-ha 1e-12 --occupation-tolerance 1e-12 \
  >"$run_root/current-initial-state-validator.stdout" \
  2>"$run_root/current-initial-state-validator.stderr"
grep -Fq '"passed": true' "$run_root/current-initial-state-validation.json"

PYTHONPATH="$tool_dir" "$python" -B \
  "$tool_dir/validate_qsgw_trace_closure.py" \
  "$old_run/qsgw_oracle_matrices.dat" \
  "$run_root/old-closure-validation.json" \
  --iterations "0:$target_iter" --channel 0 --legacy-contract \
  --closure-tolerance-ha 1e-10 --hermiticity-tolerance-ha 1e-10 \
  --hartree-reference-tolerance-ha 1e-10 \
  --hartree-response-minimum-ha 1e-12 \
  >"$run_root/old-closure-validator.stdout" \
  2>"$run_root/old-closure-validator.stderr"
grep -Fq '"passed": true' "$run_root/old-closure-validation.json"

PYTHONPATH="$tool_dir" "$python" -B \
  "$tool_dir/validate_qsgw_trace_closure.py" \
  "$current_run/qsgw_matrices.dat" \
  "$run_root/current-closure-validation.json" \
  --iterations "0:$target_iter" --channel 0 \
  --closure-tolerance-ha 1e-10 --hermiticity-tolerance-ha 1e-10 \
  --hartree-reference-tolerance-ha 1e-10 \
  --hartree-response-minimum-ha 1e-12 \
  >"$run_root/current-closure-validator.stdout" \
  2>"$run_root/current-closure-validator.stderr"
grep -Fq '"passed": true' "$run_root/current-closure-validation.json"

PYTHONPATH="$tool_dir" "$python" -B \
  "$tool_dir/validate_qsgw_fixed_basis.py" \
  "$current_run/qsgw_matrices.dat" "$current_run/qsgw_eigenvalues.dat" \
  "$input_dir/band_out" "$run_root/current-fixed-basis-validation.json" \
  --iterations "0:$target_iter" --channel 0 \
  --eigenvalue-tolerance-ha 1e-10 --invariant-tolerance 1e-10 \
  >"$run_root/current-fixed-basis-validator.stdout" \
  2>"$run_root/current-fixed-basis-validator.stderr"
grep -Fq '"passed": true' "$run_root/current-fixed-basis-validation.json"

printf 'final_acceptance=true\nresult=PASS\ncompleted_utc=%s\n' \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >>"$run_root/PROVENANCE.txt"
(
  cd "$run_root"
  find current -maxdepth 1 -type f -print0 | sort -z | xargs -0 sha256sum \
    >CURRENT_OUTPUT_SHA256SUMS.txt
  sha256sum --check --quiet CURRENT_OUTPUT_SHA256SUMS.txt
  find . -type f ! -name OUTPUT_SHA256SUMS.txt ! -name COMPLETE \
    ! -name FAILED -print0 | sort -z | xargs -0 sha256sum \
    >OUTPUT_SHA256SUMS.txt
  sha256sum --check --quiet OUTPUT_SHA256SUMS.txt
)
touch "$run_root/COMPLETE"
echo GATE6B7_HARTREE_ON_ORACLE_V40_VS_A25E4E43_MPI4_OMP12_MINITER2=PASS
