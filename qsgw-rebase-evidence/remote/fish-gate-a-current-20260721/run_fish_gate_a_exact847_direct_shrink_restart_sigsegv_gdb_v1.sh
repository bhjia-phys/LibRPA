#!/usr/bin/env bash
set -euo pipefail

: "${RUNNER_COMMIT:?RUNNER_COMMIT must identify the clean runner checkout}"
: "${RUNNER_SOURCE:?RUNNER_SOURCE must identify the clean runner checkout}"
: "${RUNNER_SHA256:?RUNNER_SHA256 must identify this exact runner}"
: "${RUN_ID:?RUN_ID must name a fresh immutable run directory}"

[[ "$RUNNER_COMMIT" =~ ^[0-9a-f]{40}$ ]]
[[ "$RUNNER_SHA256" =~ ^[0-9a-f]{64}$ ]]
case "$RUN_ID" in
  *[!A-Za-z0-9._-]*|'') echo "RUN_ID contains unsafe characters" >&2; exit 2 ;;
esac

base=/home/bhj/ai-runs
run_root=$base/$RUN_ID
source_run=$base/librpa-qsgw-gate-a1-exact847-fullcoul-nohead-shrinkchi-off-miniter2-20260722-v3
source_work=$source_run/legacy
source_checkpoint=$source_work/librpa.d/qsgw_checkpoints
bundle=$base/librpa-qsgw-gate-a-symmetry-band0-bundle-20260720-v2
dataset=$bundle/dataset
legacy_source=/tmp/librpa-qsgw-gate-a0-legacy-exact847-band0-scheme-a-20260720-v12
legacy_build=$legacy_source/build
legacy_exe=$legacy_build/chi0_main.exe
work=$run_root/restart-gdb

expected_legacy_exe_sha=6edd4e9847ce6536815eee17627bfd35be5b8443283d52818726ab2e5a74af50
expected_failed_sha=acea8cb23bf445e18da415193e099532413036a0ce9c28393835b38fa1cc5ddf
expected_source_provenance_sha=ba3db4dd770806337c85e4f4c446eb02ad959473f69f99d02d81e80c1dc5df0c
expected_source_input_sha=75c895f04642497b6578e9ec061cb29c7a1af8d6ae680eaddd8d0e7115b20b85
expected_source_stdout_sha=e7642746b53937792a974e791f5e6ca93c3e63b48584709126022ffd9e7184ac
expected_source_stderr_sha=0cb18c37eb3f24e1fe03482852987d5aa34220d9755b07af197c800486421d90
expected_source_history_sha=8157cd3154c5515a8bbe3ede38731cf40c0bb952d47879004f8934d5d0addeb7
expected_source_band_sha=9b92a7267695fcd29d133cd012d93f571735517cb619c26c956f60e314ccc9cc
expected_latest_sha=4355a46b19d348dc2f57c046f8ef63d4538ebb936000f3c9ee954a27460dd865
expected_checkpoint_meta_sha=5e7ff5f8a44a77e57d860f57f72e0623d5ef503077f841267ffa42c6460ac7ee
expected_checkpoint_matrix_shas=(
  0e28d870136a05e67f012b22b31ddac6af31e4ee6ffef3cddc2dc65eabc92a03
  2c35d317cdff6a3f189d1da8177616fa7cca09731d71cb2d74a0fc06f1cf3cca
  0836c9dd0e82dc2235c3d2134ed678f7483d3534c2455f83547171454539603a
  525a185516d4d9965b00137de39465d90b38ab01893bfc45c834966e8ffb3cf2
  9c6c58dc7d707f87676783f4356998c54c1a893ede6adb21defc3a4aa2822fe8
  f6ab1d381cba4a0236c4a8ec4ff7b2a8b5ac5ee833128d6f49ae5cac63629804
  acf93703141588102afa9b18b5e31bd29e726743030034e89ac120e50070911a
  70b9efe5e82eb3cb44cf0e830225e43367291c496eeea9dcf83b2f205930f37b
)

run_succeeded=0
record_exit() {
  local rc=$?
  trap - EXIT
  if [[ $run_succeeded -ne 1 && -d ${run_root:-/nonexistent} ]]; then
    printf 'failed_utc=%s\nexit_code=%s\n' \
      "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$rc" >"$run_root/FAILED"
  fi
  exit "$rc"
}
trap record_exit EXIT

test ! -e "$run_root"
test -e "$RUNNER_SOURCE/.git"
test "$(git -C "$RUNNER_SOURCE" rev-parse HEAD)" = "$RUNNER_COMMIT"
test -z "$(git -C "$RUNNER_SOURCE" status --porcelain)"
test "$(sha256sum "$0" | awk '{print $1}')" = "$RUNNER_SHA256"
command -v gdb >/dev/null

test -e "$source_run/FAILED"
test ! -e "$source_run/COMPLETE"
test "$(sha256sum "$source_run/FAILED" | awk '{print $1}')" = "$expected_failed_sha"
test "$(sha256sum "$source_run/PROVENANCE.txt" | awk '{print $1}')" = \
  "$expected_source_provenance_sha"
test "$(sha256sum "$source_work/librpa.in" | awk '{print $1}')" = \
  "$expected_source_input_sha"
test "$(sha256sum "$source_work/librpa.stdout" | awk '{print $1}')" = \
  "$expected_source_stdout_sha"
test "$(sha256sum "$source_work/librpa.stderr" | awk '{print $1}')" = \
  "$expected_source_stderr_sha"
test "$(sha256sum "$source_work/homo_lumo_vs_iterations.dat" | awk '{print $1}')" = \
  "$expected_source_history_sha"
test "$(sha256sum "$source_work/QSGW_band_spin_1_1.dat" | awk '{print $1}')" = \
  "$expected_source_band_sha"
test "$(sha256sum "$source_checkpoint/latest_iteration.txt" | awk '{print $1}')" = \
  "$expected_latest_sha"
test "$(cat "$source_checkpoint/latest_iteration.txt")" = 1
test "$(sha256sum "$source_checkpoint/iter_00001/checkpoint.meta" | awk '{print $1}')" = \
  "$expected_checkpoint_meta_sha"
for ik in $(seq 1 8); do
  printf -v name 'H0_GW_spin_01_k_%06d.bin' "$ik"
  test "$(sha256sum "$source_checkpoint/iter_00001/$name" | awk '{print $1}')" = \
    "${expected_checkpoint_matrix_shas[$((ik - 1))]}"
done
test -x "$legacy_exe"
test "$(sha256sum "$legacy_exe" | awk '{print $1}')" = "$expected_legacy_exe_sha"
grep -Fq 'exit_code=255' "$source_run/FAILED"
grep -Fq 'KILLED BY SIGNAL: 11 (Segmentation fault)' "$source_work/librpa.stdout"
grep -Fqx '1 4.08114 4.08114 4.08114' \
  "$source_work/homo_lumo_vs_iterations.dat"

mkdir -p "$work/librpa.d"
for entry in "$dataset"/*; do
  ln -s "$entry" "$work/$(basename "$entry")"
done
ln -s "$source_work/QSGW_band_spin_1_1.dat" \
  "$work/QSGW_band_spin_1_1.dat"

cat >"$work/librpa.in" <<EOF
task = qsgw_band0
nfreq = 16
tfgrid_type = minimax
n_params_anacon = 16
option_dielect_func = 0
replace_w_head = f
use_scalapack_gw_wc = t
use_scalapack_ecrpa = t
parallel_routing = libri
vq_threshold = 0
sqrt_coulomb_threshold = 0
gf_R_threshold = 1e-12
use_shrink_abfs = t
use_shrink_chi = f
use_abacus_exx_symmetry = t
use_abacus_gw_symmetry = t
use_fullcoul_exx = t
use_fullcoul_eps = t
use_fullcoul_wc = f
use_pyatb = f
output_energy_qp = t
output_gw_sigc_mat_rf = f
libri_chi0_threshold_C = 1e-4
libri_chi0_threshold_G = 1e-5
libri_exx_threshold_V = 1e-1
libri_exx_threshold_C = 1e-4
libri_exx_threshold_D = 1e-4
libri_g0w0_threshold_C = 1e-5
libri_g0w0_threshold_G = 1e-5
libri_g0w0_threshold_Wc = 1e-6
max_iter = 2
qsgw_restart = t
qsgw_restart_dir = $source_checkpoint/
qsgw_restart_iteration = 1
qsgw_checkpoint_every = 1
qsgw_export_hamiltonian_for_pyatb = f
qsgw_band0_unoccupied_keep = 44
qsgw_band0_cut_mode = 0
qsgw_band0_cut_shift_ha = 20.0
qsgw_band0_update_hartree = f
output_dir = librpa.d/
EOF

set +u
source /opt/intel/oneapi/setvars.sh --force \
  >"$run_root/oneapi-setvars.stdout" \
  2>"$run_root/oneapi-setvars.stderr"
set -u
export OMP_NUM_THREADS=32
export MKL_NUM_THREADS=32
export OPENBLAS_NUM_THREADS=32
export OMP_PROC_BIND=spread
export OMP_PLACES=cores
export I_MPI_PIN_DOMAIN=omp
export LIBRI_DETERMINISTIC_REDUCTION=1
export LD_LIBRARY_PATH="$legacy_build/src:$legacy_build/qsgw:${LD_LIBRARY_PATH:-}"
ulimit -c 0

cat >"$run_root/PROVENANCE.txt" <<EOF
gate=gate_a1_exact847_direct_shrink_restart_sigsegv_gdb_v1
acceptance=diagnostic_only_expected_failure
runner_commit=$RUNNER_COMMIT
runner_sha256=$RUNNER_SHA256
source_failed_run=$source_run
source_failed_sha256=$expected_failed_sha
source_stdout_sha256=$expected_source_stdout_sha
source_stderr_sha256=$expected_source_stderr_sha
source_history_sha256=$expected_source_history_sha
source_checkpoint_iteration=1
legacy_commit=8476213f66c68efb43404713eacbd04966820f26
legacy_executable=$legacy_exe
legacy_executable_sha256=$expected_legacy_exe_sha
use_shrink_chi=false
symmetry=on
headwing=off
hartree=off
restart_target_iteration=2
omp_threads=32
started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
EOF
sha256sum "$work/librpa.in" >>"$run_root/PROVENANCE.txt"

set +e
(
  cd "$work"
  timeout 1800 mpirun -np 1 gdb -q -batch -return-child-result \
    -ex 'set pagination off' \
    -ex 'set print thread-events off' \
    -ex 'handle SIGSEGV stop print nopass' \
    -ex run \
    -ex 'thread apply all bt full' \
    --args "$legacy_exe" 16 1e-12
) >"$run_root/gdb.stdout" 2>"$run_root/gdb.stderr"
gdb_rc=$?
set -e
printf 'gdb_exit_code=%s\n' "$gdb_rc" >"$run_root/gdb-exit.txt"
cat "$run_root/gdb.stdout" "$run_root/gdb.stderr" >"$run_root/gdb.combined"

test "$gdb_rc" -ne 0
grep -Fq 'QSGW_BAND] Restarting from checkpoint iteration 1' \
  "$run_root/gdb.combined"
grep -Fq 'Program received signal SIGSEGV' "$run_root/gdb.combined"
grep -Eq '^#[0-9]+ ' "$run_root/gdb.combined"

printf 'diagnostic_captured=true\ncompleted_utc=%s\n' \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >>"$run_root/PROVENANCE.txt"
touch "$run_root/EXPECTED_FAILURE_CAPTURED"
touch "$run_root/DIAGNOSTIC_COMPLETE"
(
  cd "$run_root"
  find . -type f ! -name OUTPUT_SHA256SUMS.txt ! -name COMPLETE \
    ! -name FAILED -print0 | LC_ALL=C sort -z | xargs -0 sha256sum \
    >OUTPUT_SHA256SUMS.txt
  sha256sum --check --quiet OUTPUT_SHA256SUMS.txt
)
touch "$run_root/COMPLETE"
run_succeeded=1

echo GATE_A1_EXACT847_DIRECT_SHRINK_RESTART_SIGSEGV_GDB_V1=DIAGNOSTIC_CAPTURED
grep -m 24 -E '^#[0-9]+ ' "$run_root/gdb.combined"
