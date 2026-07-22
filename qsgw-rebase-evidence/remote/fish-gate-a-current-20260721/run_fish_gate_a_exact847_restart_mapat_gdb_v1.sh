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
source_run=$base/librpa-qsgw-gate-a1-exact847-fullcoul-nohead-miniter2-20260722-6a85c7fc-v1
source_work=$source_run/legacy
source_checkpoint=$source_work/librpa.d/qsgw_checkpoints
bundle=$base/librpa-qsgw-gate-a-symmetry-band0-bundle-20260720-v2
dataset=$bundle/dataset
legacy_source=/tmp/librpa-qsgw-gate-a0-legacy-exact847-band0-scheme-a-20260720-v12
legacy_build=$legacy_source/build
legacy_exe=$legacy_build/chi0_main.exe
work=$run_root/restart-gdb

expected_legacy_exe_sha=6edd4e9847ce6536815eee17627bfd35be5b8443283d52818726ab2e5a74af50
expected_failed_sha=9f1826056923ac1a29c3b6d5df0904c7e2f200c1f88c5051c2308a227a9eaf33
expected_source_provenance_sha=7fe87a9d85eeeb4badd1bbbd3f419054a714d7aff61a7b02a849ad6d31ac9602
expected_source_input_sha=4fc3699bb2b49d9e59ff2cf1ed438bfd558cd35a7b7b639fabcb4cf39056c796
expected_source_stdout_sha=5f9540cdd0903b2a5d8ed0a45f7f47ada777160350ff73808c8f79f38aeb4751
expected_source_stderr_sha=4208e3a96cd66ef8121f1b33363a02734a75fcbcd55a7e3ed5eb1567f0cda147
expected_source_history_sha=8a2e96435bc6c7604d69e500f87563c199df09d8546c7230ad14dd2083fd8d89
expected_latest_sha=4355a46b19d348dc2f57c046f8ef63d4538ebb936000f3c9ee954a27460dd865
expected_checkpoint_meta_sha=ff3126cc8b6ca8a7a6ced9ec8189aa6ded94cb3b4cdd26bdd3aa7720f1773dd6
expected_checkpoint_matrix_shas=(
  1490ad99894f41beab8675f47f92d24814f2854b4703cb689a139a0318de0254
  86aa1643255e9546aab10ce8cd6751e90d74d2591b3a81c30cd5107f9ea56391
  57517a73c066557d66e3660715359c85bc6b4ec44550188ca21dcba5b2729ba1
  b4c2d9dc02ca9e3750eec5a3c9799210d311d451c17641b3aaa2ad0ce0bb737c
  94f5650a4cd04840495df5a7cd90906ac5b30ea1f8b50aa58e546630a2e3d3a2
  ccaccb418f88277b47fb5fe8ff2cc38b431927d217c49c320bccc37166acdb6c
  46a4d878f67d4800dd3e1afd5790264461434596d04c089611f565b642631e39
  0f485ddce0af6a5cde5cfde8ebedd7946d9ec0137b33be1db89eb61d7f459af1
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
grep -Fq 'std::out_of_range' "$source_work/librpa.stderr"
grep -Fq 'what():  map::at' "$source_work/librpa.stderr"
grep -Fq 'Iteration 1: HOMO = 4.81178 eV, LUMO = 4.81216 eV' \
  "$source_work/librpa.stdout"

mkdir -p "$work/librpa.d"
for entry in "$dataset"/*; do
  ln -s "$entry" "$work/$(basename "$entry")"
done
ln -s "$source_work/QSGW_band_spin_1_1.dat" \
  "$work/QSGW_band_spin_1_1.dat"

cat >"$work/librpa.in" <<EOF
task = qsgw_band0
nfreq = 16
n_params_anacon = 16
option_dielect_func = 0
replace_w_head = f
use_scalapack_gw_wc = t
use_scalapack_ecrpa = t
parallel_routing = libri
vq_threshold = 0
sqrt_coulomb_threshold = 0
use_shrink_abfs = t
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
gate=gate_a1_exact847_restart_mapat_gdb_v1
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
    -ex run \
    -ex 'thread apply all bt' \
    --args "$legacy_exe" 16 1e-12
) >"$run_root/gdb.stdout" 2>"$run_root/gdb.stderr"
gdb_rc=$?
set -e
printf 'gdb_exit_code=%s\n' "$gdb_rc" >"$run_root/gdb-exit.txt"
cat "$run_root/gdb.stdout" "$run_root/gdb.stderr" >"$run_root/gdb.combined"

test "$gdb_rc" -ne 0
grep -Fq 'QSGW_BAND] Restarting from checkpoint iteration 1' \
  "$run_root/gdb.combined"
grep -Fq 'std::out_of_range' "$run_root/gdb.combined"
grep -Fq 'map::at' "$run_root/gdb.combined"
grep -Fq 'Program received signal SIGABRT' "$run_root/gdb.combined"
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

echo GATE_A1_EXACT847_RESTART_MAPAT_GDB_V1=DIAGNOSTIC_CAPTURED
grep -m 24 -E '^#[0-9]+ ' "$run_root/gdb.combined"
