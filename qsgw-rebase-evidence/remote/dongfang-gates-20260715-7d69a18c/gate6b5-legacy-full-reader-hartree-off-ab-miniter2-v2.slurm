#!/usr/bin/env bash
#SBATCH --job-name=qsgw-g6b5-offab2
#SBATCH --partition=v100m3
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:8
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --time=01:00:00
#SBATCH --output=/data/home/df_iopcas_bhj/ai-runs/librpa-qsgw-rebase-gates-20260715T1910-7d69a18c/logs/gate6b5-legacy-full-reader-off-ab-%j.out
#SBATCH --error=/data/home/df_iopcas_bhj/ai-runs/librpa-qsgw-rebase-gates-20260715T1910-7d69a18c/logs/gate6b5-legacy-full-reader-off-ab-%j.err

set -euo pipefail

root=/data/home/df_iopcas_bhj/ai-runs/librpa-qsgw-rebase-gates-20260715T1910-7d69a18c
target_iter=2
input_root=$root/inputs/si-k444-gate6b1-mpi1-full-abfs-hartree-contracts-2348832
input_dir=$input_root/dataset
old_build=/data/home/df_iopcas_bhj/qsgw-builds/legacy-hartree-oracle-v36-20260717
old_exe=$old_build/build/chi0_main.exe
adapter_build=/data/home/df_iopcas_bhj/qsgw-builds/legacy-hartree-full-reader-oracle-v40-20260717
adapter_exe=$adapter_build/build/chi0_main.exe
run_root=$root/runs/gate6b5-legacy-full-reader-hartree-off-ab-mpi4-omp12-miniter${target_iter}-$SLURM_JOB_ID
old_run=$run_root/old
adapter_run=$run_root/adapter

expected_old_exe_sha=97aca5a73f50c4d1f7e48018a2c7ac3b5fb712401ebb8ed79fcf23a0e6128c05
expected_old_build_provenance_sha=c733c00c5da083436179d9bf64345effa0ef4398d6e4826ad073d9b45a9db4e0
expected_old_source_provenance_sha=fb05da6aee4c6f0d8bb671523d015c44808e08b9cad6a9da9aa47df9eb0f6938
expected_old_output_manifest_sha=e323a42a09db2b41ab70f77e18a0406043a7c397d79871792df9a317559f535c
expected_adapter_exe_sha=413cbc964bd81e0245fde4d40b09c5d5edc257bd805972db4fe5929f22189e4e
expected_adapter_build_provenance_sha=ba1881eadc624395f620e123f162b0c263569f4c3b41ea7bb0b8f7dcffb8996e
expected_adapter_source_provenance_sha=0bcd2f55600ffa4fdfd8cbbfdcda56c93e388f1de0930f4217b97e216de74cae
expected_adapter_output_manifest_sha=1add0e5f772941f7a0edce2ce078ad52993d17f324fe86d1b19276947a27a6f2
expected_input_dataset_manifest_sha=6e183ba7e91eb03311836bf5d2e6f0bf2c209338679fae1fc9c9cc627423bba1
expected_input_provenance_sha=532d4106269f299a7346e2ba30178844e9cd82cd48439dc6b61a4459e46de2cb
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
mkdir -p "$old_run" "$adapter_run"
test -e "$input_root/COMPLETE"
test -e "$old_build/COMPLETE"
test -e "$adapter_build/COMPLETE"
test -x "$old_exe"
test -x "$adapter_exe"

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
$adapter_exe $expected_adapter_exe_sha
$adapter_build/build-provenance.txt $expected_adapter_build_provenance_sha
$adapter_build/source-provenance.txt $expected_adapter_source_provenance_sha
$adapter_build/OUTPUT_SHA256SUMS.txt $expected_adapter_output_manifest_sha
$input_root/DATASET_SHA256SUMS.txt $expected_input_dataset_manifest_sha
$input_root/PROVENANCE.txt $expected_input_provenance_sha
$input_dir/qsgw_input.hartree.contract $expected_hartree_contract_sha
EOF

(
  cd "$input_root"
  sha256sum --check --quiet DATASET_SHA256SUMS.txt
)
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

cp "$0" "$run_root/gate6b5-legacy-full-reader-hartree-off-ab-miniter2-v2.slurm"
cp "$old_build/build-provenance.txt" "$run_root/old-build-provenance.txt"
cp "$old_build/source-provenance.txt" "$run_root/old-source-provenance.txt"
cp "$old_build/OUTPUT_SHA256SUMS.txt" "$run_root/old-build-OUTPUT_SHA256SUMS.txt"
cp "$adapter_build/build-provenance.txt" "$run_root/adapter-build-provenance.txt"
cp "$adapter_build/source-provenance.txt" "$run_root/adapter-source-provenance.txt"
cp "$adapter_build/OUTPUT_SHA256SUMS.txt" "$run_root/adapter-build-OUTPUT_SHA256SUMS.txt"
cp "$input_root/DATASET_SHA256SUMS.txt" "$run_root/input-DATASET_SHA256SUMS.txt"
cp "$input_root/PROVENANCE.txt" "$run_root/input-PROVENANCE.txt"
cp "$input_dir/qsgw_input.hartree.contract" "$run_root/"

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
cp "$old_run/librpa.in" "$adapter_run/librpa.in"

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
gate=gate6b5-legacy-full-reader-hartree-off-ab-mpi4-omp12-miniter2
final_acceptance=false_adapter_isolation_only
benchmark_acceptance=false
result=PENDING
old_executable=$old_exe
old_executable_sha256=$expected_old_exe_sha
old_hartree_build=real_legacy_hartree_cpp_cvclr
adapter_executable=$adapter_exe
adapter_executable_sha256=$expected_adapter_exe_sha
adapter_scope=full_reader_only_when_QSGW_ORACLE_UPDATE_HARTREE_is_true
input=$input_dir
input_dataset_manifest_sha256=$expected_input_dataset_manifest_sha
hartree_contract_sha256=$expected_hartree_contract_sha
target_iteration=$target_iter
use_shrink_abfs=false
crystal_symmetry=off
time_reversal_reduction=off_full_64_kpoint_input
headwing=off
hartree=off
hartree_coulomb=truncated
hartree_normalization=legacy_extra_inverse_nk
band=off
mpi_ranks=$SLURM_NTASKS
omp_threads=$OMP_NUM_THREADS
LIBRI_DETERMINISTIC_REDUCTION=$LIBRI_DETERMINISTIC_REDUCTION
slurm_job_id=$SLURM_JOB_ID
slurm_node_list=$SLURM_JOB_NODELIST
started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
EOF
sha256sum "$old_run/librpa.in" "$adapter_run/librpa.in" "$input_dir/band_out" \
  "$input_dir/stru_out" "$input_dir/qsgw_input.hartree.contract" \
  >>"$run_root/PROVENANCE.txt"

(
  cd "$old_run"
  export QSGW_ORACLE_TRACE=1
  export QSGW_ORACLE_UPDATE_HARTREE=0
  export QSGW_HROUND_SCALE=0
  export LIBRPA_QSGW_MIXING_BETA=0.2
  export LD_LIBRARY_PATH="$old_build/build/qsgw:$old_build/build/src:$base_ld_library_path"
  mpirun -np "$SLURM_NTASKS" "$old_exe" >librpa.stdout 2>librpa.stderr
)

test -s "$old_run/qsgw_oracle_matrices.dat"
test -s "$old_run/homo_lumo_vs_iterations.dat"
grep -Fqx '# qsgw_contract_version 4' "$old_run/qsgw_oracle_matrices.dat"
grep -Fqx '# qsgw_update_hartree 0' "$old_run/qsgw_oracle_matrices.dat"
test "$(awk 'NF && $1 !~ /^#/ {last=$1} END {print last}' \
  "$old_run/homo_lumo_vs_iterations.dat")" = "$target_iter"

(
  cd "$adapter_run"
  export QSGW_ORACLE_TRACE=1
  export QSGW_ORACLE_UPDATE_HARTREE=0
  export QSGW_HROUND_SCALE=0
  export LIBRPA_QSGW_MIXING_BETA=0.2
  export LD_LIBRARY_PATH="$adapter_build/build/qsgw:$adapter_build/build/src:$base_ld_library_path"
  mpirun -np "$SLURM_NTASKS" "$adapter_exe" >librpa.stdout 2>librpa.stderr
)

test -s "$adapter_run/qsgw_oracle_matrices.dat"
test -s "$adapter_run/homo_lumo_vs_iterations.dat"
grep -Fqx '# qsgw_contract_version 4' "$adapter_run/qsgw_oracle_matrices.dat"
grep -Fqx '# qsgw_update_hartree 0' "$adapter_run/qsgw_oracle_matrices.dat"
test "$(awk 'NF && $1 !~ /^#/ {last=$1} END {print last}' \
  "$adapter_run/homo_lumo_vs_iterations.dat")" = "$target_iter"

cmp "$old_run/qsgw_oracle_matrices.dat" "$adapter_run/qsgw_oracle_matrices.dat"
cmp "$old_run/homo_lumo_vs_iterations.dat" "$adapter_run/homo_lumo_vs_iterations.dat"
{
  sha256sum "$old_run/qsgw_oracle_matrices.dat" \
    "$adapter_run/qsgw_oracle_matrices.dat"
  sha256sum "$old_run/homo_lumo_vs_iterations.dat" \
    "$adapter_run/homo_lumo_vs_iterations.dat"
} >"$run_root/TRACE_EQUALITY_SHA256SUMS.txt"

printf 'adapter_isolation=PASS_bitwise_equal_hartree_off\nresult=PASS\ncompleted_utc=%s\n' \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >>"$run_root/PROVENANCE.txt"
(
  cd "$run_root"
  find old -maxdepth 1 -type f -print0 | sort -z | xargs -0 sha256sum \
    >OLD_OUTPUT_SHA256SUMS.txt
  find adapter -maxdepth 1 -type f -print0 | sort -z | xargs -0 sha256sum \
    >ADAPTER_OUTPUT_SHA256SUMS.txt
  sha256sum --check --quiet OLD_OUTPUT_SHA256SUMS.txt
  sha256sum --check --quiet ADAPTER_OUTPUT_SHA256SUMS.txt
  find . -type f ! -name OUTPUT_SHA256SUMS.txt ! -name COMPLETE \
    ! -name FAILED -print0 | sort -z | xargs -0 sha256sum \
    >OUTPUT_SHA256SUMS.txt
  sha256sum --check --quiet OUTPUT_SHA256SUMS.txt
)
touch "$run_root/COMPLETE"
echo GATE6B5_LEGACY_FULL_READER_HARTREE_OFF_AB=PASS
