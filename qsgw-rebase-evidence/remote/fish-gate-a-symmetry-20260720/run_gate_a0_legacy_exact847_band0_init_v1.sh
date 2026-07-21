#!/usr/bin/env bash
set -euo pipefail

: "${RUNNER_SHA256:?RUNNER_SHA256 must identify this exact runner}"

run_root=/home/bhj/ai-runs/librpa-qsgw-gate-a0-legacy-exact847-band0-init-20260720-v1
bundle=/home/bhj/ai-runs/librpa-qsgw-gate-a-symmetry-band0-bundle-20260720-v2
dataset=$bundle/dataset
build_evidence=/home/bhj/ai-runs/librpa-qsgw-gate-a0-legacy-exact847-build-20260720-v8
legacy_build=/tmp/librpa-qsgw-gate-a0-legacy-exact847-20260720-v8/build
legacy_exe=$legacy_build/chi0_main.exe
work=$run_root/work

expected_bundle_dataset_sha=7fe17e43e20978833cd3c4bead17943c1ed9f231e5b03b3e46d9f342e9904956
expected_bundle_output_sha=25590340b95edd0aaa91e3c8b34e332545afd9143ba96807a391290a7e0e9a43
expected_legacy_exe_sha=481ec33b3118747eb33ff3c252ab23fe23f7202c3cee7ee7147ac60b2e5cedaa
expected_initial='Initial HOMO = 6.28506 eV, LUMO = 6.97617 eV, Fermi Energy = 6.48709 eV'
mpi_ranks=1
omp_threads=32

record_failure() {
  local rc=$?
  trap - ERR
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
test -e "$build_evidence/GREEN_CONFIRMED"
test -x "$legacy_exe"
test "$(sha256sum "$bundle/DATASET_SHA256SUMS.txt" | awk '{print $1}')" = \
  "$expected_bundle_dataset_sha"
test "$(sha256sum "$bundle/OUTPUT_SHA256SUMS.txt" | awk '{print $1}')" = \
  "$expected_bundle_output_sha"
test "$(sha256sum "$legacy_exe" | awk '{print $1}')" = \
  "$expected_legacy_exe_sha"
test -z "$(find "$bundle" -type l -print -quit)"
test -z "$(find "$bundle" -perm /222 -print -quit)"
(
  cd "$bundle"
  sha256sum --check --quiet OUTPUT_SHA256SUMS.txt
)

mkdir -p "$work/librpa.d"
for entry in "$dataset"/*; do
  ln -s "$entry" "$work/$(basename "$entry")"
done
test -L "$work/band_out"
test -L "$work/band_kpath_info"
test -L "$work/symrot_abf_k.txt"
test -L "$work/pyatb_librpa_df"
test "$(readlink -f "$work/band_out")" = "$dataset/band_out"
test "$(readlink -f "$work/band_kpath_info")" = "$dataset/band_kpath_info"

cat >"$work/librpa.in" <<'EOF'
task = qsgw_band0
nfreq = 16
n_params_anacon = 16
option_dielect_func = 3
replace_w_head = t
use_scalapack_gw_wc = t
use_scalapack_ecrpa = t
parallel_routing = libri
vq_threshold = 0
sqrt_coulomb_threshold = 0
use_shrink_abfs = t
use_abacus_exx_symmetry = t
use_abacus_gw_symmetry = t
use_fullcoul_exx = t
use_pyatb = t
output_energy_qp = t
output_gw_sigc_mat_rf = t
libri_chi0_threshold_C = 1e-4
libri_chi0_threshold_G = 1e-5
libri_exx_threshold_V = 1e-1
libri_exx_threshold_C = 1e-4
libri_exx_threshold_D = 1e-4
libri_g0w0_threshold_C = 1e-5
libri_g0w0_threshold_G = 1e-5
libri_g0w0_threshold_Wc = 1e-6
max_iter = 0
qsgw_checkpoint_every = 1
qsgw_export_hamiltonian_for_pyatb = t
qsgw_hr_export_full_mp_rgrid = t
qsgw_band0_unoccupied_keep = 10
qsgw_band0_cut_mode = 2
qsgw_band0_cut_shift_ha = 20.0
output_dir = librpa.d/
EOF

mkdir -p "$run_root/provenance"
printf '%s\n' "$RUNNER_SHA256" >"$run_root/provenance/runner-sha256.txt"
cp "$bundle/PROVENANCE.txt" "$run_root/provenance/input-bundle-PROVENANCE.txt"
cp "$bundle/DATASET_SHA256SUMS.txt" "$run_root/provenance/input-DATASET_SHA256SUMS.txt"
cp "$build_evidence/PROVENANCE.txt" "$run_root/provenance/legacy-build-PROVENANCE.txt"
sha256sum "$work/librpa.in" >"$run_root/provenance/librpa.in.sha256"

set +u
source /opt/intel/oneapi/setvars.sh --force \
  >"$run_root/provenance/oneapi-setvars.stdout" \
  2>"$run_root/provenance/oneapi-setvars.stderr"
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
gate=gate_a0_legacy_exact847_qsgw_band0_initialization_v1
acceptance=initialization_and_input_binding_only
legacy_commit=8476213f66c68efb43404713eacbd04966820f26
legacy_executable=$legacy_exe
legacy_executable_sha256=$expected_legacy_exe_sha
source_patch=none
input_bundle=$bundle
input_dataset_manifest_sha256=$expected_bundle_dataset_sha
input_layout=symlink_farm_to_immutable_physical_bundle
task=qsgw_band0
max_iter=0
nfreq=16
n_params_anacon=16
symmetry=on_ibz_8_to_full_bz_64
headwing=historical_input_enabled_preflight_only
hartree=off
band=historical_201_point_reference_loaded
h_qsgw_cut=mode2_keep10_shift20Ha_not_executed
mpi_ranks=$mpi_ranks
omp_threads=$omp_threads
started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
EOF

cd "$work"
timeout 1800 mpirun -np "$mpi_ranks" "$legacy_exe" \
  >"$run_root/librpa.stdout" 2>"$run_root/librpa.stderr"
grep -Fq 'Initialization finished' "$run_root/librpa.stdout"
grep -Fq 'Task work begins: qsgw_band0' "$run_root/librpa.stdout"
grep -Fq 'QSGW band0: max_iterations = 0' "$run_root/librpa.stdout"
grep -Fqx "$expected_initial" "$run_root/librpa.stdout"
grep -Fq 'libRPA finished successfully' "$run_root/librpa.stdout"
test "$(wc -l <"$run_root/librpa.stderr")" -eq 8
for kpoint in $(seq 1 8); do
  grep -Fqx "HF file not found: hf_exchange_spin_01_kpt_$(printf '%06d' "$kpoint").csc" \
    "$run_root/librpa.stderr"
done
test ! -e QSGW_band_spin_1_0.dat

(
  cd "$bundle"
  sha256sum --check --quiet OUTPUT_SHA256SUMS.txt
)
printf 'result=PASS\ncompleted_utc=%s\n' \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >>"$run_root/PROVENANCE.txt"
(
  cd "$run_root"
  find . -type f ! -name OUTPUT_SHA256SUMS.txt ! -name GREEN_CONFIRMED \
    ! -name FAILED -print0 | LC_ALL=C sort -z | xargs -0 sha256sum \
    >OUTPUT_SHA256SUMS.txt
  sha256sum --check --quiet OUTPUT_SHA256SUMS.txt
)
touch "$run_root/GREEN_CONFIRMED"

echo GATE_A0_LEGACY_EXACT847_BAND0_INIT_V1=PASS
grep -F "$expected_initial" "$run_root/librpa.stdout"
