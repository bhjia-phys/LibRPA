#!/usr/bin/env bash
set -euo pipefail

: "${RUNNER_COMMIT:?RUNNER_COMMIT must identify the clean runner checkout}"
: "${RUNNER_SOURCE:?RUNNER_SOURCE must identify the clean runner checkout}"
: "${RUNNER_SHA256:?RUNNER_SHA256 must identify this exact runner}"
: "${RUN_ID:?RUN_ID must name a fresh immutable observer run}"

base=/home/bhj/ai-runs
run_root=$base/$RUN_ID
legacy_source=/tmp/librpa-qsgw-gate-a0-legacy-exact847-band0-scheme-a-20260720-v12
observer_root=/tmp/librpa-qsgw-exact847-component-observer-${RUNNER_COMMIT:0:8}-v1
observer_source=$observer_root/source
observer_build=$observer_root/build
observer_exe=$observer_build/chi0_main.exe
bundle=$base/librpa-qsgw-gate-a-symmetry-band0-bundle-20260720-v2
dataset=$bundle/dataset
frozen_legacy=$base/librpa-qsgw-gate-a1-exact847-fullcoul-nohead-shrinkchi-off-miniter2-20260722-v3
frozen_checkpoint=$frozen_legacy/legacy/librpa.d/qsgw_checkpoints/iter_00001
candidate_run=$base/librpa-qsgw-gate-a1-exact847-candidate-one-update-20260722-69c33c2f-v3
candidate_trace=$candidate_run/candidate/qsgw_matrices.dat
work=$run_root/legacy
python=$base/librpa-qsgw-gate0-20260715T1731-7d69a18c/venv/bin/python
current_dir=$RUNNER_SOURCE/qsgw-rebase-evidence/remote/fish-gate-a-current-20260721
symmetry_dir=$RUNNER_SOURCE/qsgw-rebase-evidence/remote/fish-gate-a-symmetry-20260720

expected_source_task_sha=34e5c93fe12259f4838469b0e19b2c2316d4b6871ba9d6c9da5b3c057a01ab34
expected_fermi_cpp_sha=e4dcf3cb0998f312eaeab2e530c1cb306c0b9790608034784dad0ca3ea1437f3
expected_fermi_h_sha=6965348b51d698720ce8b3ef9bc27314e66c5fb82a82ef8b31d524dc0dc1f0d3
expected_bundle_dataset_sha=7fe17e43e20978833cd3c4bead17943c1ed9f231e5b03b3e46d9f342e9904956
expected_grid_contract_sha=5b90f7314d7231e0d1cc3d272957e26b9aa89d9b204c71d0378d1940098f0d46
expected_candidate_trace_sha=e8f96767491dca9ac2f379598142cbf093b132c2faf9dcf22f3e97c6f271de1e
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
mpi_ranks=1
omp_threads=32

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
test ! -e "$observer_root"
test -e "$RUNNER_SOURCE/.git"
test "$(git -C "$RUNNER_SOURCE" rev-parse HEAD)" = "$RUNNER_COMMIT"
test -z "$(git -C "$RUNNER_SOURCE" status --porcelain)"
test "$(sha256sum "$0" | awk '{print $1}')" = "$RUNNER_SHA256"
test -x "$python"
test "$(sha256sum "$legacy_source/driver/task_qsgw_band_0.cpp" | awk '{print $1}')" = \
  "$expected_source_task_sha"
test "$(sha256sum "$legacy_source/qsgw/fermi_energy_occupation.cpp" | awk '{print $1}')" = \
  "$expected_fermi_cpp_sha"
test "$(sha256sum "$legacy_source/qsgw/fermi_energy_occupation.h" | awk '{print $1}')" = \
  "$expected_fermi_h_sha"
test -e "$bundle/COMPLETE"
test ! -e "$bundle/FAILED"
test "$(sha256sum "$bundle/DATASET_SHA256SUMS.txt" | awk '{print $1}')" = \
  "$expected_bundle_dataset_sha"
test "$(sha256sum "$dataset/qsgw_input.contract" | awk '{print $1}')" = \
  "$expected_grid_contract_sha"
test "$(sha256sum "$candidate_trace" | awk '{print $1}')" = \
  "$expected_candidate_trace_sha"

mkdir -p "$run_root" "$observer_source"
cp "$0" "$run_root/"
/usr/bin/rsync -a --exclude build/ "$legacy_source/" "$observer_source/"
"$python" -B "$current_dir/instrument_exact847_component_dump_v1.py" \
  "$observer_source/driver/task_qsgw_band_0.cpp" \
  "$observer_source/driver/task_qsgw_band_0.cpp.instrumented" \
  >"$run_root/instrumentation.stdout" \
  2>"$run_root/instrumentation.stderr"
test ! -s "$run_root/instrumentation.stderr"
mv "$observer_source/driver/task_qsgw_band_0.cpp.instrumented" \
  "$observer_source/driver/task_qsgw_band_0.cpp"
grep -Fq 'LIBRPA_QSGW_LEGACY_COMPONENT_DUMP' \
  "$observer_source/driver/task_qsgw_band_0.cpp"
instrumented_task_sha=$(sha256sum \
  "$observer_source/driver/task_qsgw_band_0.cpp" | awk '{print $1}')
instrumenter_sha=$(sha256sum \
  "$current_dir/instrument_exact847_component_dump_v1.py" | awk '{print $1}')

set +u
source /opt/intel/oneapi/setvars.sh --force \
  >"$run_root/oneapi-setvars.stdout" \
  2>"$run_root/oneapi-setvars.stderr"
set -u

cmake -S "$observer_source" -B "$observer_build" \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DCMAKE_CXX_COMPILER=mpiicpx \
  -DCMAKE_Fortran_COMPILER=mpiifx \
  -DENABLE_DOCS=OFF \
  -DENABLE_DRIVER=ON \
  -DENABLE_FORTRAN_BIND=OFF \
  -DENABLE_TEST=OFF \
  -DENABLE_UNITTESTS=OFF \
  -DUSE_CMAKE_INC=OFF \
  -DUSE_EXTERNAL_GREENX=OFF \
  -DUSE_GREENX_API=ON \
  -DUSE_LIBRI=ON \
  >"$run_root/configure.stdout" \
  2>"$run_root/configure.stderr"
cmake --build "$observer_build" -j 32 \
  >"$run_root/build.stdout" \
  2>"$run_root/build.stderr"
test -x "$observer_exe"
observer_exe_sha=$(sha256sum "$observer_exe" | awk '{print $1}')
ldd "$observer_exe" >"$run_root/observer-ldd.txt"

mkdir -p "$work/librpa.d"
for entry in "$dataset"/*; do
  ln -s "$entry" "$work/$(basename "$entry")"
done
cat >"$work/librpa.in" <<'EOF'
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
output_gw_sigc_mat_rf = t
libri_chi0_threshold_C = 1e-4
libri_chi0_threshold_G = 1e-5
libri_exx_threshold_V = 1e-1
libri_exx_threshold_C = 1e-4
libri_exx_threshold_D = 1e-4
libri_g0w0_threshold_C = 1e-5
libri_g0w0_threshold_G = 1e-5
libri_g0w0_threshold_Wc = 1e-6
max_iter = 1
qsgw_checkpoint_every = 1
qsgw_export_hamiltonian_for_pyatb = f
qsgw_band0_unoccupied_keep = 44
qsgw_band0_cut_mode = 0
qsgw_band0_cut_shift_ha = 20.0
qsgw_band0_update_hartree = f
output_dir = librpa.d/
EOF

export OMP_NUM_THREADS=$omp_threads
export MKL_NUM_THREADS=$omp_threads
export OPENBLAS_NUM_THREADS=$omp_threads
export OMP_PROC_BIND=spread
export OMP_PLACES=cores
export I_MPI_PIN_DOMAIN=omp
export LIBRI_DETERMINISTIC_REDUCTION=1
export LIBRPA_WCFQ_DUMP=1
export LIBRPA_QSGW_LEGACY_COMPONENT_DUMP=1
export LD_LIBRARY_PATH="$observer_build/src:$observer_build/qsgw:${LD_LIBRARY_PATH:-}"

cat >"$run_root/PROVENANCE.txt" <<EOF
gate=gate_a1_exact847_component_observer_v1
acceptance=pending_observer_runtime
runner_commit=$RUNNER_COMMIT
runner_sha256=$RUNNER_SHA256
legacy_commit=8476213f66c68efb43404713eacbd04966820f26
legacy_role=scheme_a_occupation_only_plus_read_only_component_observer
legacy_source=$legacy_source
source_task_sha256=$expected_source_task_sha
instrumenter_sha256=$instrumenter_sha
instrumented_task_sha256=$instrumented_task_sha
observer_source=$observer_source
observer_build=$observer_build
observer_executable=$observer_exe
observer_executable_sha256=$observer_exe_sha
input_bundle=$bundle
input_dataset_manifest_sha256=$expected_bundle_dataset_sha
grid_contract_sha256=$expected_grid_contract_sha
candidate_trace=$candidate_trace
candidate_trace_sha256=$expected_candidate_trace_sha
target_iteration=1
crystal_symmetry=on_ibz_8_to_full_bz_64
mixing=direct_update_none
headwing=off
hartree=off
use_shrink_chi=false
h_qsgw_cut=off_mode0
nfreq=16
use_fullcoul_exx=true
mpi_ranks=$mpi_ranks
omp_threads=$omp_threads
started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
EOF

(
  cd "$work"
  printf 'started_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >runtime.txt
  timeout 14400 mpirun -np "$mpi_ranks" "$observer_exe" 16 1e-12 \
    >librpa.stdout 2>librpa.stderr
  printf 'completed_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >>runtime.txt
)

grep -Fq 'Task work begins: qsgw_band0' "$work/librpa.stdout"
grep -Fq 'QSGW band0: max_iterations = 1' "$work/librpa.stdout"
grep -Fq 'Iteration 1: HOMO =' "$work/librpa.stdout"
grep -Fq 'libRPA finished successfully' "$work/librpa.stdout"
test "$(sha256sum "$work/librpa.stderr" | awk '{print $1}')" = \
  0cb18c37eb3f24e1fe03482852987d5aa34220d9755b07af197c800486421d90
test "$(grep -c '^HF file not found: hf_exchange_spin_01_kpt_' \
  "$work/librpa.stderr")" -eq 8
test "$(awk 'NF {last=$1} END {print last}' \
  "$work/homo_lumo_vs_iterations.dat")" = 1

component_dir=$work/librpa.d/qsgw_legacy_components/iter_00001
test -f "$component_dir/metadata.txt"
test "$(find "$component_dir" -maxdepth 1 -type f -name '*.bin' | wc -l)" -eq 168
test "$(find "$component_dir" -maxdepth 1 -type f \
  -name 'sigma_c_iw_*.bin' | wc -l)" -eq 128
test "$(find "$component_dir" -maxdepth 1 -type f \
  ! -name 'sigma_c_iw_*.bin' -name '*.bin' | wc -l)" -eq 40

observer_checkpoint=$work/librpa.d/qsgw_checkpoints/iter_00001
for ik in $(seq 1 8); do
  printf -v name 'H0_GW_spin_01_k_%06d.bin' "$ik"
  test "$(sha256sum "$observer_checkpoint/$name" | awk '{print $1}')" = \
    "${expected_checkpoint_matrix_shas[$((ik - 1))]}"
  test "$(sha256sum "$frozen_checkpoint/$name" | awk '{print $1}')" = \
    "${expected_checkpoint_matrix_shas[$((ik - 1))]}"
done

tools_dir=$run_root/tools
mkdir -p "$tools_dir"
cp "$current_dir/instrument_exact847_component_dump_v1.py" "$tools_dir/"
cp "$current_dir/compare_exact847_component_dump_v1.py" "$tools_dir/"
cp "$current_dir/diagnose_exact847_component_parity_v1.py" "$tools_dir/"
cp "$current_dir/compare_legacy_h0_candidate_trace_v2.py" "$tools_dir/"
cp "$symmetry_dir/compare_legacy_h0_candidate_trace_v1.py" "$tools_dir/"
cp "$symmetry_dir/compare_legacy_band0_native_outputs_v1.py" "$tools_dir/"
PYTHONPATH="$tools_dir" "$python" -B \
  "$tools_dir/compare_exact847_component_dump_v1.py" \
  "$component_dir" "$candidate_trace" \
  "$run_root/component-comparison.json" \
  --iteration 1 --n-frequencies 16 --n-spins 1 --n-kpoints 8 --n-bands 44 \
  >"$run_root/comparison.stdout" \
  2>"$run_root/comparison.stderr"
test ! -s "$run_root/comparison.stderr"
grep -Fq '"diagnostic_complete": true' "$run_root/component-comparison.json"

printf 'acceptance=false_diagnostic_only\ncheckpoint_byte_parity=true\ncompleted_utc=%s\n' \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >>"$run_root/PROVENANCE.txt"
(
  cd "$run_root"
  find . -type f ! -name OUTPUT_SHA256SUMS.txt \
    ! -name DIAGNOSTIC_COMPLETE ! -name FAILED -print0 \
    | LC_ALL=C sort -z | xargs -0 sha256sum >OUTPUT_SHA256SUMS.txt
  sha256sum --check --quiet OUTPUT_SHA256SUMS.txt
)
touch "$run_root/DIAGNOSTIC_COMPLETE"
cat "$run_root/PROVENANCE.txt"
cat "$run_root/component-comparison.json"
run_succeeded=1
trap - EXIT
