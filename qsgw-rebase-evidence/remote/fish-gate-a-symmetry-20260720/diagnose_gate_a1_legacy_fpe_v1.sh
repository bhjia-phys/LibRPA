#!/usr/bin/env bash
set -euo pipefail

: "${RUNNER_SHA256:?RUNNER_SHA256 must identify this exact runner}"

run_root=/home/bhj/ai-runs/librpa-qsgw-gate-a1-legacy-fpe-diagnostic-20260720-v1
bundle=/home/bhj/ai-runs/librpa-qsgw-gate-a-symmetry-bundle-20260720-v2-2378526
dataset=$bundle/dataset
dataset_input_dir=$dataset/
legacy_build=/tmp/librpa-qsgw-gate-a0-legacy-e08f4a13-20260720-v3/build
legacy_exe=$legacy_build/chi0_main.exe
legacy_exe_sha=28ca6be12c777441abf98f6047a079cb3a07924efe816caa97b0a80e9566c267

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
test -e "$bundle/COMPLETE"
test ! -e "$bundle/FAILED"
test -x "$legacy_exe"
test "$(sha256sum "$legacy_exe" | awk '{print $1}')" = "$legacy_exe_sha"
test "${dataset_input_dir: -1}" = "/"
command -v gdb
mkdir -p "$run_root"

cat >"$run_root/librpa.in" <<EOF
task = qsgw
input_dir = $dataset_input_dir
output_dir = .
nfreq = 6
tfgrid_type = minimax
n_params_anacon = -1
parallel_routing = libri
vq_threshold = 0
sqrt_coulomb_threshold = 0
use_scalapack_gw_wc = true
use_shrink_abfs = true
use_shrink_chi = false
use_pyatb = false
replace_w_head = false
option_dielect_func = 0
use_fullcoul_exx = false
use_fullcoul_wc = false
use_abacus_exx_symmetry = true
use_abacus_gw_symmetry = true
qsgw_iterative_headwing = false
max_iter = 0
EOF

cat >"$run_root/PROVENANCE.txt" <<EOF
gate=gate_a1_legacy_fpe_diagnostic
acceptance=false_diagnostic_only
runner_sha256=$RUNNER_SHA256
legacy_executable=$legacy_exe
legacy_executable_sha256=$legacy_exe_sha
dataset=$dataset
input_dir=$dataset_input_dir
input_dir_trailing_slash=true
mpi_ranks=1
omp_threads=1
max_iter=0
started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
EOF

set +u
source /opt/intel/oneapi/setvars.sh --force \
  >"$run_root/oneapi-setvars.stdout" \
  2>"$run_root/oneapi-setvars.stderr"
set -u
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export LIBRI_DETERMINISTIC_REDUCTION=1
export LD_LIBRARY_PATH="$legacy_build/src:$legacy_build/qsgw:${LD_LIBRARY_PATH:-}"
export QSGW_ORACLE_TRACE=1
export QSGW_ORACLE_UPDATE_HARTREE=0
export QSGW_HROUND_SCALE=0
export LIBRPA_QSGW_MIXING_BETA=1

cd "$run_root"
ulimit -c unlimited
set +e
gdb --batch \
  -ex 'set pagination off' \
  -ex 'handle SIGFPE stop print nopass' \
  -ex run \
  -ex 'thread apply all bt full' \
  --args "$legacy_exe" >gdb.stdout 2>gdb.stderr
rc=$?
set -e
printf 'gdb_exit_code=%s\ncompleted_utc=%s\n' \
  "$rc" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >>PROVENANCE.txt
sha256sum librpa.in gdb.stdout gdb.stderr PROVENANCE.txt >OUTPUT_SHA256SUMS.txt
sha256sum --check --quiet OUTPUT_SHA256SUMS.txt
touch DIAGNOSTIC_COMPLETE
echo GATE_A1_LEGACY_FPE_DIAGNOSTIC=COMPLETE
exit 0
