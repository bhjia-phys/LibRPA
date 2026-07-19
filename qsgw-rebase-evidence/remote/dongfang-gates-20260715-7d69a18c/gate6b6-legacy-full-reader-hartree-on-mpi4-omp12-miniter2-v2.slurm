#!/usr/bin/env bash
#SBATCH --job-name=qsgw-g6b6-oldvh2
#SBATCH --partition=v100m3
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:8
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --time=01:00:00
#SBATCH --output=/data/home/df_iopcas_bhj/ai-runs/librpa-qsgw-rebase-gates-20260715T1910-7d69a18c/logs/gate6b6-legacy-full-reader-hartree-on-%j.out
#SBATCH --error=/data/home/df_iopcas_bhj/ai-runs/librpa-qsgw-rebase-gates-20260715T1910-7d69a18c/logs/gate6b6-legacy-full-reader-hartree-on-%j.err

set -euo pipefail

root=/data/home/df_iopcas_bhj/ai-runs/librpa-qsgw-rebase-gates-20260715T1910-7d69a18c
target_iter=2
input_root=$root/inputs/si-k444-gate6b1-mpi1-full-abfs-hartree-contracts-2348832
input_dir=$input_root/dataset
adapter_build=/data/home/df_iopcas_bhj/qsgw-builds/legacy-hartree-full-reader-oracle-v40-20260717
adapter_exe=$adapter_build/build/chi0_main.exe
isolation_artifact=$root/artifacts/gate6b5-hartree-off-numerical-postprocess-2348914-2348976
comparator_source=$root/jobs/compare_qsgw_component_traces-v6-hartree-1715ae0a.py
closure_source=$root/jobs/validate_qsgw_trace_closure-v3-4a5de94e.py
closure_test_source=$root/jobs/test_validate_qsgw_trace_closure-v3-38de02fa.py
python=/data/home/df_iopcas_bhj/software-stack/miniconda3/bin/python3
run_root=$root/runs/gate6b6-legacy-full-reader-hartree-on-mpi4-omp12-miniter${target_iter}-$SLURM_JOB_ID
old_run=$run_root/old
tool_dir=$run_root/tools

expected_adapter_exe_sha=413cbc964bd81e0245fde4d40b09c5d5edc257bd805972db4fe5929f22189e4e
expected_adapter_librpa_sha=1e35789c999ff8f8b6745a2d85fe90a4e0d433373cad97bed62910a6fab411d1
expected_adapter_libqsgw_sha=7e35e904131e192c8dd7b0296d932a5db466d0a1cc6842865e231c457eea8b45
expected_adapter_build_provenance_sha=ba1881eadc624395f620e123f162b0c263569f4c3b41ea7bb0b8f7dcffb8996e
expected_adapter_source_provenance_sha=0bcd2f55600ffa4fdfd8cbbfdcda56c93e388f1de0930f4217b97e216de74cae
expected_adapter_output_manifest_sha=1add0e5f772941f7a0edce2ce078ad52993d17f324fe86d1b19276947a27a6f2
expected_input_dataset_manifest_sha=6e183ba7e91eb03311836bf5d2e6f0bf2c209338679fae1fc9c9cc627423bba1
expected_input_provenance_sha=532d4106269f299a7346e2ba30178844e9cd82cd48439dc6b61a4459e46de2cb
expected_hartree_contract_sha=ea2d80953e1e03524f2c63f484a0cf333cbb59a2ed5c00f78eed141c8988edf6
expected_isolation_comparison_sha=4e3e48a8b65563093be1cbfa2754ca7fe7023ad3dd1af60190eaeebd70387a50
expected_isolation_output_manifest_sha=2903a6d1205898799ee344f23f4960e9fdb6a5b9aa3584a44cd5346182e8df8f
expected_isolation_provenance_sha=fff6ed186ac30543c66426ccaac7a983829060c7e2ec6f689f314583c60ed1d5
expected_comparator_sha=1715ae0a7029bde53599c646c35259381bd1430a847678bd13c8f9600778718f
expected_closure_sha=4a5de94e6dbf590dded4a6ecd140aa4227fa61ffa0e73af17ec0f388610cffaf
expected_closure_test_sha=38de02fabc0dde41911e5152b0b08a9987b312b0cea29c69396bb9e392e9c745

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
mkdir -p "$old_run" "$tool_dir"
test -e "$input_root/COMPLETE"
test -e "$adapter_build/COMPLETE"
test -e "$isolation_artifact/COMPLETE"
test ! -e "$isolation_artifact/FAILED"
test -x "$adapter_exe"

while read -r path expected; do
  observed=$(sha256sum "$path" | awk '{print $1}')
  if [[ $observed != "$expected" ]]; then
    printf 'hash_mismatch path=%s expected=%s observed=%s\n' \
      "$path" "$expected" "$observed" >&2
    exit 1
  fi
done <<EOF
$adapter_exe $expected_adapter_exe_sha
$adapter_build/build/src/librpa.so.0.3.0 $expected_adapter_librpa_sha
$adapter_build/build/qsgw/libqsgw.so.0.3.0 $expected_adapter_libqsgw_sha
$adapter_build/build-provenance.txt $expected_adapter_build_provenance_sha
$adapter_build/source-provenance.txt $expected_adapter_source_provenance_sha
$adapter_build/OUTPUT_SHA256SUMS.txt $expected_adapter_output_manifest_sha
$input_root/DATASET_SHA256SUMS.txt $expected_input_dataset_manifest_sha
$input_root/PROVENANCE.txt $expected_input_provenance_sha
$input_dir/qsgw_input.hartree.contract $expected_hartree_contract_sha
$isolation_artifact/comparison.json $expected_isolation_comparison_sha
$isolation_artifact/OUTPUT_SHA256SUMS.txt $expected_isolation_output_manifest_sha
$isolation_artifact/PROVENANCE.txt $expected_isolation_provenance_sha
$comparator_source $expected_comparator_sha
$closure_source $expected_closure_sha
$closure_test_source $expected_closure_test_sha
EOF

(
  cd "$input_root"
  sha256sum --check --quiet DATASET_SHA256SUMS.txt
)
(
  cd "$isolation_artifact"
  sha256sum --check --quiet OUTPUT_SHA256SUMS.txt
)
grep -Fq '"passed": true' "$isolation_artifact/comparison.json"
grep -Fqx 'full_bz_grid=4x4x4_64_kpoints' "$input_root/PROVENANCE.txt"
grep -Fqx 'crystal_symmetry=off' "$input_root/PROVENANCE.txt"
grep -Fqx 'time_reversal_reduction=off_full_64_kpoint_input' \
  "$input_root/PROVENANCE.txt"
grep -Fqx 'use_shrink_abfs=false' "$input_root/PROVENANCE.txt"
grep -Fqx 'hartree_update delta_density' \
  "$input_dir/qsgw_input.hartree.contract"
grep -Fqx 'hartree_coulomb=truncated' "$input_root/PROVENANCE.txt"
grep -Fqx 'hartree_normalization=legacy_extra_inverse_nk' \
  "$input_root/PROVENANCE.txt"
grep -Fqx 'legacy_oracle_adapter=true' "$adapter_build/source-provenance.txt"
grep -Fqx \
  'adapter_scope=task_qsgw_cut_coulomb_reader_only_when_hartree_env_enabled' \
  "$adapter_build/source-provenance.txt"
grep -Fqx 'hartree_off_reader_behavior=unchanged' \
  "$adapter_build/source-provenance.txt"

cp "$0" \
  "$run_root/gate6b6-legacy-full-reader-hartree-on-mpi4-omp12-miniter2-v2.slurm"
cp "$adapter_build/build-provenance.txt" "$run_root/adapter-build-provenance.txt"
cp "$adapter_build/source-provenance.txt" "$run_root/adapter-source-provenance.txt"
cp "$adapter_build/OUTPUT_SHA256SUMS.txt" \
  "$run_root/adapter-build-OUTPUT_SHA256SUMS.txt"
cp "$input_root/DATASET_SHA256SUMS.txt" "$run_root/input-DATASET_SHA256SUMS.txt"
cp "$input_root/PROVENANCE.txt" "$run_root/input-PROVENANCE.txt"
cp "$input_dir/qsgw_input.hartree.contract" "$run_root/"
cp "$isolation_artifact/comparison.json" "$run_root/gate6b5-isolation-comparison.json"
cp "$isolation_artifact/OUTPUT_SHA256SUMS.txt" \
  "$run_root/gate6b5-isolation-OUTPUT_SHA256SUMS.txt"
cp "$isolation_artifact/PROVENANCE.txt" "$run_root/gate6b5-isolation-PROVENANCE.txt"
cp "$comparator_source" "$tool_dir/compare_qsgw_component_traces.py"
cp "$closure_source" "$tool_dir/validate_qsgw_trace_closure.py"
cp "$closure_source" "$tool_dir/validate_qsgw_trace_closure_v3.py"
cp "$closure_test_source" "$tool_dir/test_validate_qsgw_trace_closure_v3.py"

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
gate=gate6b6-legacy-full-reader-hartree-on-mpi4-omp12-miniter2
final_acceptance=false_legacy_oracle_smoke_only
benchmark_acceptance=false
result=PENDING
adapter_executable=$adapter_exe
adapter_executable_sha256=$expected_adapter_exe_sha
adapter_scope=full_reader_only_when_QSGW_ORACLE_UPDATE_HARTREE_is_true
adapter_isolation_artifact=$isolation_artifact
adapter_isolation_comparison_sha256=$expected_isolation_comparison_sha
parser_observer_sha256=$expected_comparator_sha
closure_observer_sha256=$expected_closure_sha
input=$input_dir
input_dataset_manifest_sha256=$expected_input_dataset_manifest_sha
hartree_contract_sha256=$expected_hartree_contract_sha
target_iteration=$target_iter
mixing=legacy_linear_beta_0.2
use_shrink_abfs=false
crystal_symmetry=off
time_reversal_reduction=off_full_64_kpoint_input
headwing=off
hartree=delta_density
hartree_coulomb=truncated
hartree_normalization=legacy_extra_inverse_nk
band=off
mpi_ranks=$SLURM_NTASKS
omp_threads=$OMP_NUM_THREADS
LIBRI_DETERMINISTIC_REDUCTION=$LIBRI_DETERMINISTIC_REDUCTION
python=$($python --version 2>&1)
slurm_job_id=$SLURM_JOB_ID
slurm_node_list=$SLURM_JOB_NODELIST
started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
EOF
sha256sum "$old_run/librpa.in" "$input_dir/band_out" \
  "$input_dir/stru_out" "$input_dir/qsgw_input.hartree.contract" \
  >>"$run_root/PROVENANCE.txt"

PYTHONPATH="$tool_dir" "$python" -B \
  "$tool_dir/test_validate_qsgw_trace_closure_v3.py" \
  >"$run_root/closure-unit-test.stdout" \
  2>"$run_root/closure-unit-test.stderr"

(
  cd "$old_run"
  export QSGW_ORACLE_TRACE=1
  export QSGW_ORACLE_UPDATE_HARTREE=1
  export QSGW_HROUND_SCALE=0
  export LIBRPA_QSGW_MIXING_BETA=0.2
  export LD_LIBRARY_PATH="$adapter_build/build/qsgw:$adapter_build/build/src:$base_ld_library_path"
  mpirun -np "$SLURM_NTASKS" "$adapter_exe" >librpa.stdout 2>librpa.stderr
)

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

PYTHONPATH="$tool_dir" "$python" -B \
  "$tool_dir/validate_qsgw_trace_closure_v3.py" \
  "$old_run/qsgw_oracle_matrices.dat" \
  "$run_root/old-closure-validation.json" \
  --iterations "0:$target_iter" --channel 0 --legacy-contract \
  --closure-tolerance-ha 1e-10 --hermiticity-tolerance-ha 1e-10 \
  --hartree-reference-tolerance-ha 1e-10 \
  --hartree-response-minimum-ha 1e-12 \
  >"$run_root/old-closure-validator.stdout" \
  2>"$run_root/old-closure-validator.stderr"
grep -Fq '"passed": true' "$run_root/old-closure-validation.json"

printf 'oracle_smoke=PASS\nresult=PASS\ncompleted_utc=%s\n' \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >>"$run_root/PROVENANCE.txt"
(
  cd "$run_root"
  find old -maxdepth 1 -type f -print0 | sort -z | xargs -0 sha256sum \
    >OLD_OUTPUT_SHA256SUMS.txt
  sha256sum --check --quiet OLD_OUTPUT_SHA256SUMS.txt
  find . -type f ! -name OUTPUT_SHA256SUMS.txt ! -name COMPLETE \
    ! -name FAILED -print0 | sort -z | xargs -0 sha256sum \
    >OUTPUT_SHA256SUMS.txt
  sha256sum --check --quiet OUTPUT_SHA256SUMS.txt
)
touch "$run_root/COMPLETE"
echo GATE6B6_LEGACY_FULL_READER_HARTREE_ON_MPI4_OMP12_MINITER2=PASS
