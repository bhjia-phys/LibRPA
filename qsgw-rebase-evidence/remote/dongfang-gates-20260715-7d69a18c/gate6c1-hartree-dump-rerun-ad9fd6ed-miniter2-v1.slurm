#!/usr/bin/env bash
# Gate 6C1: rerun the candidate QSGW Hartree-on miniter2 with the
# env-gated Hartree pipeline dump enabled (commit ad9fd6ed), producing the
# intermediates for the independent recomputation observer. The run uses
# the same frozen Si k444 full-ABF input, the same librpa.in contract as
# the frozen Gate 6B7 current run (job 2350992, commit a25e4e43), and the
# same MPI=4/OMP=12 layout. The dump is a no-op numerically; the follow-up
# acceptance job compares this run's trace against the frozen Gate 6B7
# trace before using the dumps as observer input.
#SBATCH --job-name=qsgw-g6c1-vh-dump
#SBATCH --partition=v100m3
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:8
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --time=01:30:00
#SBATCH --output=/data/home/df_iopcas_bhj/ai-runs/librpa-qsgw-rebase-gates-20260715T1910-7d69a18c/logs/gate6c1-hartree-dump-rerun-%j.out
#SBATCH --error=/data/home/df_iopcas_bhj/ai-runs/librpa-qsgw-rebase-gates-20260715T1910-7d69a18c/logs/gate6c1-hartree-dump-rerun-%j.err

set -euo pipefail

target_iter=2
mixing_beta=0.2
root=/data/home/df_iopcas_bhj/ai-runs/librpa-qsgw-rebase-gates-20260715T1910-7d69a18c
input_root=$root/inputs/si-k444-gate6b1-mpi1-full-abfs-hartree-contracts-2348832
input_dir=$input_root/dataset
frozen_run=$root/runs/gate6b7-hartree-on-oracle-v40-vs-a25e4e43-mpi4-omp12-miniter2-2350992
candidate_build=$root/build-candidate-ad9fd6ed-v1
candidate_exe=$candidate_build/chi0_main.exe
python=/data/home/df_iopcas_bhj/software-stack/miniconda3/bin/python3
run_root=$root/runs/gate6c1-hartree-dump-rerun-ad9fd6ed-miniter2-$SLURM_JOB_ID
current_run=$run_root/current
dump_dir=$run_root/hartree-dump

expected_candidate_commit=ad9fd6ed65552b5ad2b3e80483aaf56538d15805
expected_candidate_exe_sha=74648f0ccf0ab84bec2c096a57b1ef03187d0e701027b0b2abea6183df2e46da
expected_input_dataset_manifest_sha=6e183ba7e91eb03311836bf5d2e6f0bf2c209338679fae1fc9c9cc627423bba1
expected_input_provenance_sha=532d4106269f299a7346e2ba30178844e9cd82cd48439dc6b61a4459e46de2cb
expected_off_contract_sha=3ae9f9d514f47a4d2b87fd8709d77c10e690348763b44446be5b7074e00c0e17
expected_hartree_contract_sha=ea2d80953e1e03524f2c63f484a0cf333cbb59a2ed5c00f78eed141c8988edf6
expected_frozen_trace_sha=f27fb95c7849fd75539d94b142c4b556cf9b74f4ab11e1b9c69492f9a289527a

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
mkdir -p "$current_run"
test -e "$input_root/COMPLETE"
test -e "$candidate_build/COMPLETE"
test -e "$frozen_run/FAILED"
test ! -e "$frozen_run/COMPLETE"
test -s "$frozen_run/current/qsgw_matrices.dat"

while read -r path expected; do
  observed=$(sha256sum "$path" | awk '{print $1}')
  if [[ $observed != "$expected" ]]; then
    printf 'hash_mismatch path=%s expected=%s observed=%s\n' \
      "$path" "$expected" "$observed" >&2
    exit 1
  fi
done <<EOF
$candidate_exe $expected_candidate_exe_sha
$input_root/DATASET_SHA256SUMS.txt $expected_input_dataset_manifest_sha
$input_root/PROVENANCE.txt $expected_input_provenance_sha
$input_dir/qsgw_input.off.contract $expected_off_contract_sha
$input_dir/qsgw_input.hartree.contract $expected_hartree_contract_sha
$frozen_run/current/qsgw_matrices.dat $expected_frozen_trace_sha
EOF
grep -Fqx "source_commit=$expected_candidate_commit" \
  "$candidate_build/build-provenance.txt"
grep -Fqx 'source_status=clean_linux_git_archive_from_commit' \
  "$candidate_build/build-provenance.txt"
grep -Fqx 'fish_gate0_result=60_of_60_pass' \
  "$candidate_build/build-provenance.txt"
(
  cd "$input_root"
  sha256sum --check --quiet DATASET_SHA256SUMS.txt
)
hartree_contract_sha=$(sha256sum "$input_dir/qsgw_input.hartree.contract" \
  | awk '{print $1}')

cp "$0" "$run_root/gate6c1-hartree-dump-rerun-ad9fd6ed-miniter2-v1.slurm"
cp "$candidate_build/build-provenance.txt" "$run_root/candidate-build-provenance.txt"
cp "$candidate_build/SHA256SUMS.txt" "$run_root/candidate-build-SHA256SUMS.txt"
cp "$input_root/DATASET_SHA256SUMS.txt" "$run_root/"
cp "$input_root/PROVENANCE.txt" "$run_root/input-PROVENANCE.txt"
cp "$input_dir/qsgw_input.off.contract" "$run_root/"
cp "$input_dir/qsgw_input.hartree.contract" "$run_root/"

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

frozen_librpa_in_sha=$(sha256sum "$frozen_run/current/librpa.in" | awk '{print $1}')
rerun_librpa_in_normalized=$(mktemp)
frozen_librpa_in_normalized=$(mktemp)
sed "s|input_dir = .*|input_dir = NORMALIZED|" "$current_run/librpa.in" \
  >"$rerun_librpa_in_normalized"
sed "s|input_dir = .*|input_dir = NORMALIZED|" "$frozen_run/current/librpa.in" \
  >"$frozen_librpa_in_normalized"
cmp "$rerun_librpa_in_normalized" "$frozen_librpa_in_normalized"
rm -f "$rerun_librpa_in_normalized" "$frozen_librpa_in_normalized"

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
export LIBRPA_QSGW_HARTREE_DUMP_DIR=$dump_dir
base_ld_library_path=${LD_LIBRARY_PATH:-}

cat >"$run_root/PROVENANCE.txt" <<EOF
gate=gate6c1-hartree-dump-rerun-ad9fd6ed-miniter2
final_acceptance=false_pending_trace_match
result=PENDING
candidate_commit=$expected_candidate_commit
candidate_executable=$candidate_exe
candidate_executable_sha256=$expected_candidate_exe_sha
frozen_trace_job=2350992
frozen_trace_sha256=$expected_frozen_trace_sha
frozen_librpa_in_sha256=$frozen_librpa_in_sha
librpa_in_match_after_input_dir_normalization=PASS
input=$input_dir
hartree_contract_sha256=$hartree_contract_sha
target_iteration=$target_iter
mixing=linear_beta_$mixing_beta
use_shrink_abfs=false
crystal_symmetry=off
headwing=off
band=off
hartree=delta_density
hartree_coulomb=truncated
hartree_normalization=legacy_extra_inverse_nk
dump_env=LIBRPA_QSGW_HARTREE_DUMP_DIR
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

test -d "$dump_dir/call_001"
test -d "$dump_dir/call_002"
test ! -d "$dump_dir/call_003"
for call in call_001 call_002; do
  for f in manifest.txt density_delta_k.txt hartree_k.txt hartree_r.txt; do
    test -s "$dump_dir/$call/$f"
  done
  grep -Fqx 'schema=qsgw_hartree_pipeline_dump_v1' "$dump_dir/$call/manifest.txt"
  grep -Fqx 'normalization=legacy_extra_inverse_nk' "$dump_dir/$call/manifest.txt"
  grep -Fqx 'kpoint_count=64' "$dump_dir/$call/manifest.txt"
done
grep -Fqx 'density_delta_k_count=64' "$dump_dir/call_001/manifest.txt"
grep -Fqx 'density_delta_k_count=64' "$dump_dir/call_002/manifest.txt"

printf 'dump_produced=true\ncompleted_utc=%s\n' \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >>"$run_root/PROVENANCE.txt"
(
  cd "$run_root"
  find current hartree-dump -type f -print0 | sort -z | xargs -0 sha256sum \
    >OUTPUT_SHA256SUMS.txt
  sha256sum --check --quiet OUTPUT_SHA256SUMS.txt
)
touch "$run_root/DUMP_COMPLETE"
echo GATE6C1_HARTREE_DUMP_RERUN=DUMP_COMPLETE
