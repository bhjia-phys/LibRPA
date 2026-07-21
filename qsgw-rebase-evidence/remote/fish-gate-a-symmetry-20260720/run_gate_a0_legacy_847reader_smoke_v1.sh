#!/usr/bin/env bash
set -euo pipefail

: "${RUNNER_SHA256:?RUNNER_SHA256 must identify this exact runner}"

run_root=/home/bhj/ai-runs/librpa-qsgw-gate-a0-legacy-847reader-smoke-20260720-v1
build_evidence=/home/bhj/ai-runs/librpa-qsgw-gate-a0-legacy-847reader-build-20260720-v4
bundle=/home/bhj/ai-runs/librpa-qsgw-gate-a-symmetry-bundle-20260720-v2-2378526
dataset=$bundle/dataset
dataset_input_dir=$dataset/
legacy_build=/tmp/librpa-qsgw-gate-a0-legacy-e08f4a13-847reader-20260720-v4/build
legacy_exe=$legacy_build/chi0_main.exe
legacy_exe_sha=531c9e2f1def81439d9518743089a9394bc63356ee2f8d119a2941a8e5c9bb76

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
test -e "$build_evidence/GREEN_CONFIRMED"
test -e "$bundle/COMPLETE"
test ! -e "$bundle/FAILED"
test -x "$legacy_exe"
test "$(sha256sum "$legacy_exe" | awk '{print $1}')" = "$legacy_exe_sha"
test "${dataset_input_dir: -1}" = "/"
mkdir -p "$run_root"
printf '%s\n' "$RUNNER_SHA256" >"$run_root/runner-sha256.txt"
cp "$build_evidence/PROVENANCE.txt" "$run_root/build-PROVENANCE.txt"
cp "$build_evidence/OUTPUT_SHA256SUMS.txt" "$run_root/build-OUTPUT_SHA256SUMS.txt"

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

set +u
source /opt/intel/oneapi/setvars.sh --force \
  >"$run_root/oneapi-setvars.stdout" \
  2>"$run_root/oneapi-setvars.stderr"
set -u
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export OMP_PROC_BIND=close
export OMP_PLACES=cores
export LIBRI_DETERMINISTIC_REDUCTION=1
export LD_LIBRARY_PATH="$legacy_build/src:$legacy_build/qsgw:${LD_LIBRARY_PATH:-}"
export QSGW_ORACLE_TRACE=1
export QSGW_ORACLE_UPDATE_HARTREE=0
export QSGW_HROUND_SCALE=0
export LIBRPA_QSGW_MIXING_BETA=1

cat >"$run_root/PROVENANCE.txt" <<EOF
gate=gate_a0_legacy_847reader_smoke
acceptance=false_reader_smoke_only
runner_sha256=$RUNNER_SHA256
legacy_executable=$legacy_exe
legacy_executable_sha256=$legacy_exe_sha
dataset=$dataset
input_dir=$dataset_input_dir
input_dir_trailing_slash=true
use_shrink_abfs=true
crystal_symmetry=on_ibz_8_to_full_bz_64
mpi_ranks=1
omp_threads=1
max_iter=0
started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
EOF

cd "$run_root"
timeout 1800 mpirun -np 1 "$legacy_exe" >librpa.stdout 2>librpa.stderr
grep -Fq 'iatom & small Nabfs:' librpa.stdout
grep -Fq '0,122' librpa.stdout
grep -Fq '1,122' librpa.stdout
grep -Fq 'coulomb_mat read.' librpa.stdout
grep -Fq 'Initialization finished' librpa.stdout
grep -Fq 'Task work begins: qsgw' librpa.stdout
grep -Fq 'libRPA finished successfully' librpa.stdout
if grep -Fq 'BAD TERMINATION' librpa.stdout; then
  echo 'unexpected MPI termination in reader smoke' >&2
  exit 1
fi
test -s qsgw_oracle_matrices.dat
grep -Fqx '# qsgw_contract_version 4' qsgw_oracle_matrices.dat

printf 'acceptance=true_reader_smoke\ncompleted_utc=%s\n' \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >>PROVENANCE.txt
find . -type f ! -name OUTPUT_SHA256SUMS.txt ! -name GREEN_CONFIRMED \
  ! -name FAILED -print0 | sort -z | xargs -0 sha256sum >OUTPUT_SHA256SUMS.txt
sha256sum --check --quiet OUTPUT_SHA256SUMS.txt
touch GREEN_CONFIRMED
echo GATE_A0_LEGACY_847READER_SMOKE=PASS
exit 0
