#!/usr/bin/env bash
# Smoke diagnostic (not a gate): does the pre-port QSGW implementation
# (old-new-a03a4c0-majorfix-v14, commit e08f4a13) start on (a) the FHI-aims
# Si k333 regression dataset and (b) the ABACUS BN shrink-v1 regression
# dataset? Answers the FHI/ABACUS no-symmetry QSGW regression feasibility.
# benchmark_acceptance=false: COMPLETE means the diagnostics executed.
#SBATCH --job-name=qsgw-old-smoke-fhi-bn
#SBATCH --partition=v100m3
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=32G
#SBATCH --gres=gpu:2
#SBATCH --time=00:30:00
#SBATCH --output=/data/home/df_iopcas_bhj/ai-runs/librpa-qsgw-rebase-gates-20260715T1910-7d69a18c/logs/old-smoke-fhi-bn-%j.out
#SBATCH --error=/data/home/df_iopcas_bhj/ai-runs/librpa-qsgw-rebase-gates-20260715T1910-7d69a18c/logs/old-smoke-fhi-bn-%j.err

set -euo pipefail

root=/data/home/df_iopcas_bhj/ai-runs/librpa-qsgw-rebase-gates-20260715T1910-7d69a18c
old_build=/data/home/df_iopcas_bhj/qsgw-builds/old-new-a03a4c0-majorfix-v14-20260713
old_exe=$old_build/old/build/chi0_main.exe
fhi_tgz=$root/inputs/regression-src/g0w0_aims_Si_libri-dataset.tar.gz
bn_tgz=$root/inputs/regression-src/g0w0_band_abacus_BN_shrink_v1_libri-dataset.tar.gz
work=$root/runs/old-smoke-fhi-bn-$SLURM_JOB_ID

expected_old_exe_sha=a1292eff5364565d5b7463596882580a9a758b6e0e3e600ac5dfe67113bef788
expected_fhi_sha=placeholder_fhi
expected_bn_sha=placeholder_bn

record_failure() { :; }
trap ':' ERR

mkdir -p "$work/fhi" "$work/bn"
test -x "$old_exe"
test "$(sha256sum "$old_exe" | awk '{print $1}')" = "$expected_old_exe_sha"
sha256sum "$fhi_tgz" "$bn_tgz" >"$work/source-archives.sha256"
cp "$0" "$work/old-smoke-fhi-bn.slurm"

set +u
source /data/app/intel/oneapi-2024.2/setvars.sh --force >"$work/oneapi.log" 2>&1
set -u
export OMP_NUM_THREADS=1
export LIBRI_DETERMINISTIC_REDUCTION=1
base_ld_library_path=${LD_LIBRARY_PATH:-}

smoke() {
  local label=$1 tgz=$2
  local dir=$work/$label
  tar -xzf "$tgz" -C "$dir" >"$dir/untar.log" 2>&1 || return 10
  find "$dir" -maxdepth 2 -type f | head -20 >"$dir/files.txt"
  cat >"$dir/librpa.in" <<EOF
task = qsgw
input_dir = $dir/dataset
output_dir = .
nfreq = 6
tfgrid_type = minimax
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
max_iter = 1
EOF
  (
    cd "$dir"
    export QSGW_ORACLE_TRACE=1
    export QSGW_ORACLE_UPDATE_HARTREE=0
    export QSGW_HROUND_SCALE=0
    export LIBRPA_QSGW_MIXING_BETA=1
    export LD_LIBRARY_PATH="$old_build/old/build/qsgw:$old_build/old/build/src:$base_ld_library_path"
    timeout 240 "$old_exe" >smoke.stdout 2>smoke.stderr
  )
  echo "$?"
}

set +e
fhi_rc=$(smoke fhi "$fhi_tgz")
bn_rc=$(smoke bn "$bn_tgz")
set -e
{
  echo "fhi_rc=$fhi_rc"
  echo "bn_rc=$bn_rc"
  echo "fhi_first_lines:"
  head -8 "$work/fhi/smoke.stdout" 2>/dev/null
  echo "fhi_error_tail:"
  tail -6 "$work/fhi/smoke.stderr" 2>/dev/null
  echo "bn_first_lines:"
  head -8 "$work/bn/smoke.stdout" 2>/dev/null
  echo "bn_error_tail:"
  tail -6 "$work/bn/smoke.stderr" 2>/dev/null
} >"$work/SMOKE_RESULT.txt"

cat >"$work/PROVENANCE.txt" <<EOF
analysis=old_qsgw_smoke_fhi_aims_si_vs_abacus_bn
benchmark_acceptance=false
diagnostic_only=true
old_commit=e08f4a130df7661e9ac355b9be45fb2bf9c3ed01
old_executable_sha256=$expected_old_exe_sha
fhi_rc=$fhi_rc
bn_rc=$bn_rc
slurm_job_id=$SLURM_JOB_ID
completed_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
EOF
cat "$work/SMOKE_RESULT.txt"
touch "$work/COMPLETE"
echo OLD_SMOKE_DONE
