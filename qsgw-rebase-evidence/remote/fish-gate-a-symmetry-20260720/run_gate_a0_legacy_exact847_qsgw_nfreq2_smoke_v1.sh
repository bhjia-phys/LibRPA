#!/usr/bin/env bash
set -euo pipefail

: "${RUNNER_SHA256:?RUNNER_SHA256 must identify this exact runner}"

run_root=/home/bhj/ai-runs/librpa-qsgw-gate-a0-legacy-exact847-qsgw-nfreq2-smoke-20260720-v1
build_evidence=/home/bhj/ai-runs/librpa-qsgw-gate-a0-legacy-exact847-build-20260720-v8
bundle=/home/bhj/ai-runs/librpa-qsgw-gate-a-symmetry-bundle-20260720-v3
dataset=$bundle/dataset
work=$run_root/work
legacy_build=/tmp/librpa-qsgw-gate-a0-legacy-exact847-20260720-v8/build
legacy_exe=$legacy_build/chi0_main.exe

legacy_exe_sha=481ec33b3118747eb33ff3c252ab23fe23f7202c3cee7ee7147ac60b2e5cedaa
build_provenance_sha=4a9f54c6fc03445efefe96be60885ab43e1b03d34208950b41904d8171711c80
build_output_manifest_sha=bbf6e45c2f1cf799ec9c36ac3b0a872041ac955842db76af2c752e8faf73d8fb
dataset_manifest_sha=869f4fd922dc1085af2cc02644f2a65e5b462237fde6428eed5a46143e833690
bundle_output_manifest_sha=b3be3227d0aea82492e92085340d567b47a6ba3ebb0976e1f23d542e9d5db648
mpi_ranks=1
omp_threads=32

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
test "$(sha256sum "$build_evidence/PROVENANCE.txt" | awk '{print $1}')" = \
  "$build_provenance_sha"
test "$(sha256sum "$build_evidence/OUTPUT_SHA256SUMS.txt" | awk '{print $1}')" = \
  "$build_output_manifest_sha"
test "$(sha256sum "$bundle/DATASET_SHA256SUMS.txt" | awk '{print $1}')" = \
  "$dataset_manifest_sha"
test "$(sha256sum "$bundle/OUTPUT_SHA256SUMS.txt" | awk '{print $1}')" = \
  "$bundle_output_manifest_sha"
(
  cd "$bundle"
  sha256sum --check --quiet OUTPUT_SHA256SUMS.txt
)

mkdir -p "$run_root"
printf '%s\n' "$RUNNER_SHA256" >"$run_root/runner-sha256.txt"
cp "$build_evidence/PROVENANCE.txt" "$run_root/build-PROVENANCE.txt"
cp "$bundle/PROVENANCE.txt" "$run_root/bundle-PROVENANCE.txt"
cp -a "$dataset" "$work"
chmod u+w "$work"
mkdir -p "$work/librpa.d"

while read -r hash relative; do
  printf '%s  %s\n' "$hash" "${relative#dataset/}"
done <"$bundle/DATASET_SHA256SUMS.txt" >"$run_root/INPUT_SHA256SUMS.txt"
(
  cd "$work"
  sha256sum --check --quiet "$run_root/INPUT_SHA256SUMS.txt"
)

cat >"$work/librpa.in" <<'EOF'
task = qsgw
input_dir = ./
output_dir = librpa.d/
nfreq = 2
tfgrid_type = minimax
n_params_anacon = 2
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
EOF

set +u
source /opt/intel/oneapi/setvars.sh --force \
  >"$run_root/oneapi-setvars.stdout" \
  2>"$run_root/oneapi-setvars.stderr"
set -u
export OMP_NUM_THREADS=$omp_threads
export MKL_NUM_THREADS=$omp_threads
export OPENBLAS_NUM_THREADS=$omp_threads
export OMP_PROC_BIND=spread
export OMP_PLACES=cores
export I_MPI_PIN_DOMAIN=omp
export LIBRI_DETERMINISTIC_REDUCTION=1
export LD_LIBRARY_PATH="$legacy_build/src:$legacy_build/qsgw:${LD_LIBRARY_PATH:-}"

cat >"$run_root/PROVENANCE.txt" <<EOF
gate=gate_a0_legacy_exact847_qsgw_nfreq2_smoke_v1
acceptance=false_diagnostic_frequency_grid
purpose=test_exact847_task_qsgw_shrink_sinvS_path_before_nfreq6
legacy_commit=8476213f66c68efb43404713eacbd04966820f26
legacy_executable=$legacy_exe
legacy_executable_sha256=$legacy_exe_sha
source_patch=none
bundle=$bundle
dataset_manifest_sha256=$dataset_manifest_sha
input_layout=physical_copy_with_input_dir_dot_slash
task=qsgw
nfreq=2
n_params_anacon=2
symmetry=on_ibz_8_to_full_bz_64
headwing=off
hartree=off
band=off
mpi_ranks=$mpi_ranks
omp_threads=$omp_threads
started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
EOF
sha256sum "$work/librpa.in" >>"$run_root/PROVENANCE.txt"

cd "$work"
timeout 7200 mpirun -np "$mpi_ranks" "$legacy_exe" \
  >"$run_root/librpa.stdout" 2>"$run_root/librpa.stderr"
grep -Fq 'Initialization finished' "$run_root/librpa.stdout"
grep -Fq 'Task work begins: qsgw' "$run_root/librpa.stdout"
grep -Fq 'libRPA finished successfully' "$run_root/librpa.stdout"
test -s homo_lumo_vs_iterations.dat
test "$(awk 'NF && $1 !~ /^#/ {last=$1} END {print last}' \
  homo_lumo_vs_iterations.dat)" = 1
if grep -Fq 'Failed to match shrink_sinvS' \
  "$run_root/librpa.stdout" "$run_root/librpa.stderr"; then
  echo 'exact847 task=qsgw still fails shrink_sinvS matching' >&2
  exit 1
fi
(
  cd "$work"
  sha256sum --check --quiet "$run_root/INPUT_SHA256SUMS.txt"
)

printf 'run_complete=true\ncompleted_utc=%s\n' \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >>"$run_root/PROVENANCE.txt"
cd "$run_root"
find work -maxdepth 1 -type f -print0 | LC_ALL=C sort -z | \
  xargs -0 sha256sum >RUN_OUTPUT_SHA256SUMS.txt
find . -type f ! -name OUTPUT_SHA256SUMS.txt ! -name GREEN_CONFIRMED \
  ! -name FAILED -print0 | LC_ALL=C sort -z | xargs -0 sha256sum \
  >OUTPUT_SHA256SUMS.txt
sha256sum --check --quiet OUTPUT_SHA256SUMS.txt
touch GREEN_CONFIRMED
echo GATE_A0_LEGACY_EXACT847_QSGW_NFREQ2_SMOKE_V1=PASS
cat work/homo_lumo_vs_iterations.dat
