#!/usr/bin/env bash
#SBATCH --job-name=librpa-g0w0-ab
#SBATCH --partition=48cp2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=12
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --time=01:00:00
#SBATCH --output=/data/home/df_iopcas_bhj/ai-runs/librpa-qsgw-rebase-gates-20260715T1910-7d69a18c/logs/gate1-%j.out
#SBATCH --error=/data/home/df_iopcas_bhj/ai-runs/librpa-qsgw-rebase-gates-20260715T1910-7d69a18c/logs/gate1-%j.err

set -euo pipefail

root=/data/home/df_iopcas_bhj/ai-runs/librpa-qsgw-rebase-gates-20260715T1910-7d69a18c
input_root=$root/inputs/si-k444-canonical-2341555
input_dir=$input_root/dataset
upstream_build=$root/build-upstream-95c4c080-v1
candidate_build=$root/build-candidate-a76fd826-v3
upstream_exe=$upstream_build/chi0_main.exe
candidate_exe=$candidate_build/chi0_main.exe
comparator=$root/jobs/compare_g0w0_sigc.py
run_root=$root/runs/gate1-g0w0-ab-a76fd826-$SLURM_JOB_ID
upstream_run=$run_root/upstream
candidate_run=$run_root/candidate

expected_upstream_commit=95c4c08009aa6752a6d386289abe1fb4358489ca
expected_candidate_commit=a76fd826eed3929185ea47cfcb992075a72f2b77
expected_upstream_exe_sha=af93da0fccd7c9346734234f0c2ef890f3a156938fd3048294de93173434e268
expected_candidate_exe_sha=8b508963f90027b5483fb2020e5feb1b5a72495dab4bef92f0e9b9be988d9b02
expected_upstream_build_manifest_sha=48c03c7a02e8b9a07d47e4fb8babeedeba5aaa1538ff1dd6ee792ba92fa50e09
expected_upstream_build_provenance_sha=0beca921dd6bc799e47f3ab20b9b091b079d17d4f9982806664dc96bfe7ab176
expected_candidate_build_manifest_sha=b7598f2b01531ffab9017db2ac39da2413a9a31ee76b52572ca1348bd4ac4d12
expected_candidate_build_provenance_sha=7ed01a1cb203cb28734d026a16c57bc1c6367a78a6b9a96e456c233df25040a8
expected_dataset_manifest_sha=ec2666cd54303b5ade8d392067cdf4c296a7ba0503f033710e4575784fea48cf
expected_freeze_output_manifest_sha=8d9bab71e5f5f04d05a83a979f9468b124543915d57e762431b80c0ed821745e
expected_freeze_provenance_sha=02da0804133bf2585f25e0c053777881451f54e35470be30b60a162ac017ae9b
expected_contract_validation_sha=3873424e11fe1da6fe2082753e2fb7f785e15c5b45885bb2cb9dfb3369859089
expected_contract_sha=25422eee1d5f8cd4db678c05d979bbd29cb516c8088157829d6c315da19ec12b
expected_comparator_sha=53a40e78775f8b51ab19d5555324eb329c180468c23dd06ef95d0866afe61dc8

test ! -e "$run_root"
test -e "$upstream_build/COMPLETE"
test -e "$candidate_build/COMPLETE"
test -e "$input_root/COMPLETE"
test -x "$upstream_exe"
test -x "$candidate_exe"
test -f "$comparator"
grep -Fqx "source_commit=$expected_upstream_commit" \
  "$upstream_build/build-provenance.txt"
grep -Fqx "source_commit=$expected_candidate_commit" \
  "$candidate_build/build-provenance.txt"
grep -Fqx 'source_status=clean_git_archive_from_commit' \
  "$upstream_build/build-provenance.txt"
grep -Fqx 'final_acceptance=true' \
  "$upstream_build/build-provenance.txt"
grep -Fqx 'source_status=clean_git_archive_from_commit' \
  "$candidate_build/build-provenance.txt"
grep -Fqx 'final_acceptance=true' \
  "$candidate_build/build-provenance.txt"
test "$(sha256sum "$upstream_exe" | awk '{print $1}')" = \
  "$expected_upstream_exe_sha"
test "$(sha256sum "$candidate_exe" | awk '{print $1}')" = \
  "$expected_candidate_exe_sha"
test "$(sha256sum "$upstream_build/SHA256SUMS.txt" | awk '{print $1}')" = \
  "$expected_upstream_build_manifest_sha"
test "$(sha256sum "$upstream_build/build-provenance.txt" | awk '{print $1}')" = \
  "$expected_upstream_build_provenance_sha"
test "$(sha256sum "$candidate_build/SHA256SUMS.txt" | awk '{print $1}')" = \
  "$expected_candidate_build_manifest_sha"
test "$(sha256sum "$candidate_build/build-provenance.txt" | awk '{print $1}')" = \
  "$expected_candidate_build_provenance_sha"
test "$(sha256sum "$input_root/DATASET_SHA256SUMS.txt" | awk '{print $1}')" = \
  "$expected_dataset_manifest_sha"
test "$(sha256sum "$input_root/OUTPUT_SHA256SUMS.txt" | awk '{print $1}')" = \
  "$expected_freeze_output_manifest_sha"
test "$(sha256sum "$input_root/PROVENANCE.txt" | awk '{print $1}')" = \
  "$expected_freeze_provenance_sha"
test "$(sha256sum "$input_root/contract-validation.json" | awk '{print $1}')" = \
  "$expected_contract_validation_sha"
test "$(sha256sum "$input_dir/qsgw_input.contract" | awk '{print $1}')" = \
  "$expected_contract_sha"
test "$(sha256sum "$comparator" | awk '{print $1}')" = \
  "$expected_comparator_sha"
grep -Fqx 'n_scf_kpoints 64' "$input_dir/qsgw_input.contract"
grep -Fqx 'n_headwing_kpoints 0' "$input_dir/qsgw_input.contract"
grep -Fqx 'headwing_grid disabled' "$input_dir/qsgw_input.contract"
grep -Fqx 'hartree_update off' "$input_dir/qsgw_input.contract"
grep -Fqx 'band_update off' "$input_dir/qsgw_input.contract"
grep -Fq '"passed": true' "$input_root/contract-validation.json"
grep -Fqx 'slurm_job_id=2341555' "$input_root/PROVENANCE.txt"
grep -Fqx 'freeze_dataset_type=dereferenced_regular_payload' \
  "$input_root/PROVENANCE.txt"
test "$(find "$input_dir" -type l | wc -l)" -eq 0

mkdir -p "$upstream_run" "$candidate_run"
cp "$0" "$run_root/gate1-g0w0-ab-a76fd826.slurm"
cp "$upstream_build/build-provenance.txt" \
  "$run_root/upstream-build-provenance.txt"
cp "$upstream_build/SHA256SUMS.txt" \
  "$run_root/upstream-build-SHA256SUMS.txt"
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
cp "$comparator" "$run_root/"

(
  cd "$input_root"
  sha256sum --check --quiet OUTPUT_SHA256SUMS.txt
  sha256sum --check --quiet DATASET_SHA256SUMS.txt
  printf 'frozen canonical Si-k444 payload and evidence manifests verified\n'
  cd "$candidate_build"
  sha256sum --check --quiet SHA256SUMS.txt
  printf 'clean candidate build manifest verified\n'
  cd "$upstream_build"
  sha256sum --check --quiet SHA256SUMS.txt
  printf 'clean upstream build manifest verified\n'
) >"$run_root/input-hash-check.log"

cat >"$run_root/librpa.in" <<EOF
task = g0w0
input_dir = $input_dir
output_dir = .
fn_vxc_scf = vxc_out.dat
fn_band_kpath_info = disabled_for_grid_only_sigc_comparison
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
output_gw_sigc_ks_mat_kf = true
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
output_energy_qp = false
EOF

cp "$run_root/librpa.in" "$upstream_run/librpa.in"
cp "$run_root/librpa.in" "$candidate_run/librpa.in"
cmp "$upstream_run/librpa.in" "$candidate_run/librpa.in"

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
  printf 'gate=canonical-si-k444-g0w0-upstream-candidate-ab\n'
  printf 'upstream_commit=%s\n' "$expected_upstream_commit"
  printf 'candidate_commit=%s\n' "$expected_candidate_commit"
  printf 'upstream_executable_sha256=%s\n' "$expected_upstream_exe_sha"
  printf 'candidate_executable_sha256=%s\n' "$expected_candidate_exe_sha"
  printf 'upstream_build_manifest_sha256=%s\n' \
    "$expected_upstream_build_manifest_sha"
  printf 'upstream_build_provenance_sha256=%s\n' \
    "$expected_upstream_build_provenance_sha"
  printf 'candidate_build_manifest_sha256=%s\n' \
    "$expected_candidate_build_manifest_sha"
  printf 'candidate_build_provenance_sha256=%s\n' \
    "$expected_candidate_build_provenance_sha"
  printf 'input_root=%s\n' "$input_root"
  printf 'input_freeze_job_id=2341555\n'
  printf 'dataset_manifest_sha256=%s\n' "$expected_dataset_manifest_sha"
  printf 'input_freeze_output_manifest_sha256=%s\n' \
    "$expected_freeze_output_manifest_sha"
  printf 'input_freeze_provenance_sha256=%s\n' \
    "$expected_freeze_provenance_sha"
  printf 'input_contract_validation_sha256=%s\n' \
    "$expected_contract_validation_sha"
  printf 'qsgw_input_contract_sha256=%s\n' "$expected_contract_sha"
  printf 'librpa_input_sha256=%s\n' "$(sha256sum "$run_root/librpa.in" | awk '{print $1}')"
  printf 'crystal_symmetry=off\n'
  printf 'time_reversal_reduction=off_full_64_kpoint_input\n'
  printf 'headwing=off\n'
  printf 'hartree=off\n'
  printf 'band=off\n'
  printf 'mpi_ranks=%s\n' "${SLURM_NTASKS:-unknown}"
  printf 'omp_threads=%s\n' "$OMP_NUM_THREADS"
  printf 'LIBRI_DETERMINISTIC_REDUCTION=%s\n' "$LIBRI_DETERMINISTIC_REDUCTION"
  printf 'python=%s\n' "$(python3 --version 2>&1)"
  printf 'numpy=%s\n' "$(python3 -c 'import numpy; print(numpy.__version__)')"
  printf 'slurm_job_id=%s\n' "${SLURM_JOB_ID:-unknown}"
  printf 'slurm_node_list=%s\n' "${SLURM_JOB_NODELIST:-unknown}"
  printf 'started_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  mpirun -np "$SLURM_NTASKS" hostname | sort | uniq -c
} >"$run_root/PROVENANCE.txt"

cd "$upstream_run"
mpirun -np "$SLURM_NTASKS" "$upstream_exe" \
  >librpa.stdout 2>librpa.stderr
test "$(find . -maxdepth 1 -name 'Sigc_fk_mn_kgrid_ispin_*_ik_*_ifreq_*.bin' | wc -l)" \
  -eq 384

cd "$candidate_run"
mpirun -np "$SLURM_NTASKS" "$candidate_exe" \
  >librpa.stdout 2>librpa.stderr
test "$(find . -maxdepth 1 -name 'Sigc_fk_mn_kgrid_ispin_*_ik_*_ifreq_*.bin' | wc -l)" \
  -eq 384

cd "$run_root"
python3 "$comparator" \
  "$upstream_run" \
  "$candidate_run" \
  "$run_root/g0w0-comparison.json" \
  --source kgrid \
  --max-abs-tolerance-ha 1e-12 \
  --relative-frobenius-tolerance 1e-12 \
  >"$run_root/comparator.stdout" \
  2>"$run_root/comparator.stderr"
grep -Fq '"passed": true' "$run_root/g0w0-comparison.json"

(
  cd "$upstream_run"
  find . -maxdepth 1 -name 'Sigc_fk_mn_kgrid_ispin_*_ik_*_ifreq_*.bin' \
    -print0 | sort -z | xargs -0 sha256sum
) >"$run_root/upstream-sigc.sha256"
(
  cd "$candidate_run"
  find . -maxdepth 1 -name 'Sigc_fk_mn_kgrid_ispin_*_ik_*_ifreq_*.bin' \
    -print0 | sort -z | xargs -0 sha256sum
) >"$run_root/candidate-sigc.sha256"
(
  cd "$upstream_run"
  find . -maxdepth 1 -type f ! -name 'Sigc_fk_mn_*.bin' \
    -print0 | sort -z | xargs -0 sha256sum
) >"$run_root/upstream-nonbinary.sha256"
(
  cd "$candidate_run"
  find . -maxdepth 1 -type f ! -name 'Sigc_fk_mn_*.bin' \
    -print0 | sort -z | xargs -0 sha256sum
) >"$run_root/candidate-nonbinary.sha256"

printf 'completed_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  >>"$run_root/PROVENANCE.txt"
sha256sum \
  "$run_root/PROVENANCE.txt" \
  "$run_root/librpa.in" \
  "$run_root/g0w0-comparison.json" \
  "$run_root/comparator.stdout" \
  "$run_root/comparator.stderr" \
  "$run_root/upstream-sigc.sha256" \
  "$run_root/candidate-sigc.sha256" \
  "$run_root/upstream-nonbinary.sha256" \
  "$run_root/candidate-nonbinary.sha256" \
  "$run_root/input-hash-check.log" \
  "$run_root/DATASET_SHA256SUMS.txt" \
  "$run_root/input-freeze-OUTPUT_SHA256SUMS.txt" \
  "$run_root/input-freeze-PROVENANCE.txt" \
  "$run_root/input-contract-validation.json" \
  "$run_root/qsgw_input.contract" \
  "$run_root/upstream-build-SHA256SUMS.txt" \
  "$run_root/upstream-build-provenance.txt" \
  "$run_root/candidate-build-SHA256SUMS.txt" \
  "$run_root/candidate-build-provenance.txt" \
  "$run_root/gate1-g0w0-ab-a76fd826.slurm" \
  "$run_root/compare_g0w0_sigc.py" \
  >"$run_root/OUTPUT_SHA256SUMS.txt"
touch "$run_root/COMPLETE"
printf 'CANONICAL_SI_K444_G0W0_AB=PASS\n'
