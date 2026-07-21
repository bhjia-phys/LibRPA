#!/usr/bin/env bash
set -euo pipefail

: "${RUNNER_SHA256:?RUNNER_SHA256 must identify this exact runner}"

base=/home/bhj/ai-runs
run_root=$base/librpa-qsgw-gate-a1-exact847-candidate-symmetry-nomix-miniter2-20260720-v1
baseline=$base/librpa-qsgw-gate-a0-legacy-exact847-band0-comparison-20260720-v1
bundle=$base/librpa-qsgw-gate-a-symmetry-band0-bundle-20260720-v2
dataset=$bundle/dataset
legacy_build=/tmp/librpa-qsgw-gate-a0-legacy-exact847-20260720-v8/build
candidate_build=/tmp/librpa-qsgw-reader-binding-green-v2-b7273e13/build
legacy_exe=$legacy_build/chi0_main.exe
candidate_exe=$candidate_build/chi0_main.exe
staging=/tmp/librpa-qsgw-gate-a-band0-import-20260720-v1
legacy=$run_root/legacy
candidate=$run_root/candidate
tools_dir=$run_root/tools

expected_bundle_dataset_sha=7fe17e43e20978833cd3c4bead17943c1ed9f231e5b03b3e46d9f342e9904956
expected_bundle_output_sha=25590340b95edd0aaa91e3c8b34e332545afd9143ba96807a391290a7e0e9a43
expected_grid_contract_sha=5b90f7314d7231e0d1cc3d272957e26b9aa89d9b204c71d0378d1940098f0d46
expected_legacy_exe_sha=481ec33b3118747eb33ff3c252ab23fe23f7202c3cee7ee7147ac60b2e5cedaa
expected_candidate_exe_sha=e45ca971c77d32309236a78e90ddd95aa0f37f3414befdb050ae3089eb9dc4c9
native_name=compare_legacy_band0_native_outputs_v1.py
compare_name=compare_legacy_h0_candidate_trace_v1.py
test_name=test_compare_legacy_h0_candidate_trace_v1.py
expected_native_sha=350e6589b74a27a588d44ba7f1b53f42bcd46cfebd4c9a5c7658debdb520e88d
expected_compare_sha=95b6557eb93b62f020ef135ef0ea56f19145eaebb618de3611ce1665dbf2e912
expected_test_sha=c09f9841c8bc531ac1d22a17b6a8d6c28283ab587da5848c8579ea018dfcaec3
target_iter=2
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
test -e "$baseline/COMPLETE"
test -e "$baseline/LEGACY_PARITY_GREEN"
test -e "$baseline/LEGACY_ABSOLUTE_INVARIANT_GAP"
test ! -e "$baseline/FAILED"
test -e "$bundle/COMPLETE"
test ! -e "$bundle/FAILED"
test -x "$legacy_exe"
test -x "$candidate_exe"
test "$(sha256sum "$legacy_exe" | awk '{print $1}')" = \
  "$expected_legacy_exe_sha"
test "$(sha256sum "$candidate_exe" | awk '{print $1}')" = \
  "$expected_candidate_exe_sha"
test "$(sha256sum "$bundle/DATASET_SHA256SUMS.txt" | awk '{print $1}')" = \
  "$expected_bundle_dataset_sha"
test "$(sha256sum "$bundle/OUTPUT_SHA256SUMS.txt" | awk '{print $1}')" = \
  "$expected_bundle_output_sha"
test "$(sha256sum "$dataset/qsgw_input.contract" | awk '{print $1}')" = \
  "$expected_grid_contract_sha"
test "$(sha256sum "$staging/$native_name" | awk '{print $1}')" = \
  "$expected_native_sha"
test "$(sha256sum "$staging/$compare_name" | awk '{print $1}')" = \
  "$expected_compare_sha"
test "$(sha256sum "$staging/$test_name" | awk '{print $1}')" = \
  "$expected_test_sha"
test -z "$(find "$bundle" -type l -print -quit)"
test -z "$(find "$bundle" -perm /222 -print -quit)"
(
  cd "$baseline"
  sha256sum --check --quiet OUTPUT_SHA256SUMS.txt
)
(
  cd "$bundle"
  sha256sum --check --quiet OUTPUT_SHA256SUMS.txt
)

mkdir -p "$legacy/librpa.d" "$candidate" "$tools_dir"
for entry in "$dataset"/*; do
  ln -s "$entry" "$legacy/$(basename "$entry")"
done
install -m 0444 "$staging/$native_name" "$tools_dir/$native_name"
install -m 0444 "$staging/$compare_name" "$tools_dir/$compare_name"
install -m 0444 "$staging/$test_name" "$tools_dir/$test_name"
printf '%s\n' "$RUNNER_SHA256" >"$run_root/runner-sha256.txt"

cat >"$legacy/librpa.in" <<'EOF'
task = qsgw_band0
nfreq = 6
n_params_anacon = 6
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
use_fullcoul_exx = f
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
max_iter = 2
qsgw_checkpoint_every = 1
qsgw_export_hamiltonian_for_pyatb = f
qsgw_band0_unoccupied_keep = 44
qsgw_band0_cut_mode = 0
qsgw_band0_cut_shift_ha = 20.0
qsgw_band0_update_hartree = f
output_dir = librpa.d/
EOF

cat >"$candidate/librpa.in" <<EOF
task = qsgw
input_dir = $dataset/
output_dir = ./
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
use_shrink_abfs = true
use_shrink_chi = false
use_pyatb = false
replace_w_head = false
option_dielect_func = 0
use_fullcoul_exx = false
use_fullcoul_eps = true
use_fullcoul_wc = false
use_symmetry_exx = true
use_symmetry_gw = true
use_symmetry_rpa = true
use_kpara_scf_eigvec = false
qsgw_input_contract = qsgw_input.contract
qsgw_mixer = none
qsgw_mixing_beta = 0.2
qsgw_min_iter = 2
qsgw_max_iter = 2
qsgw_write_iteration_matrices = true
qsgw_update_hartree = false
qsgw_iterative_headwing = false
EOF

cat >"$run_root/PARAMETER_MAPPING.txt" <<EOF
comparison=exact847_qsgw_band0_grid_to_candidate_qsgw_grid
iterations=0:$target_iter
input_bundle=$bundle
same_physical_bundle=true
crystal_symmetry=on_ibz_8_to_full_bz_64
legacy_task=qsgw_band0
candidate_task=qsgw
legacy_extra_band_evaluation=not_in_grid_parity_comparison
legacy_mixing=commented_out_direct_update
candidate_mixing=none_direct_update
headwing=off
hartree=off
h_qsgw_cut=off_mode0
nfreq=6
n_params_anacon=all_6_points
use_shrink_abfs=true
use_fullcoul_exx=false
use_fullcoul_eps=true
use_fullcoul_wc=false
mpi_ranks=$mpi_ranks
omp_threads=$omp_threads
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
base_ld_library_path=${LD_LIBRARY_PATH:-}

cat >"$run_root/PROVENANCE.txt" <<EOF
gate=gate_a1_exact847_candidate_symmetry_nomix_miniter2_v1
acceptance=pending_native_checkpoint_vs_candidate_trace_comparison
legacy_commit=8476213f66c68efb43404713eacbd04966820f26
legacy_executable=$legacy_exe
legacy_executable_sha256=$expected_legacy_exe_sha
candidate_commit=b7273e13c77d5ea781f192cea3c4201710b6f9fa
candidate_executable=$candidate_exe
candidate_executable_sha256=$expected_candidate_exe_sha
input_bundle=$bundle
input_dataset_manifest_sha256=$expected_bundle_dataset_sha
grid_contract_sha256=$expected_grid_contract_sha
target_iteration=$target_iter
crystal_symmetry=on_ibz_8_to_full_bz_64
mixing=legacy_direct,candidate_none_direct
headwing=off
hartree=off
h_qsgw_cut=off_mode0
band=legacy_computes_but_comparison_channel_is_grid_only
nfreq=6
mpi_ranks=$mpi_ranks
omp_threads=$omp_threads
comparator_sha256=$expected_compare_sha
started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
EOF
sha256sum "$legacy/librpa.in" "$candidate/librpa.in" \
  "$dataset/qsgw_input.contract" >>"$run_root/PROVENANCE.txt"

(
  cd "$tools_dir"
  python3 -m unittest "$test_name"
) >"$run_root/comparator-tests.stdout" \
  2>"$run_root/comparator-tests.stderr"

(
  cd "$legacy"
  export LIBRPA_WCFQ_DUMP=1
  export LD_LIBRARY_PATH="$legacy_build/src:$legacy_build/qsgw:$base_ld_library_path"
  printf 'started_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    >runtime.txt
  timeout 14400 mpirun -np "$mpi_ranks" "$legacy_exe" 6 1e-12 \
    >librpa.stdout 2>librpa.stderr
  printf 'completed_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    >>runtime.txt
)
grep -Fq 'Task work begins: qsgw_band0' "$legacy/librpa.stdout"
grep -Fq 'QSGW band0: max_iterations = 2' "$legacy/librpa.stdout"
grep -Fq 'QSGW band0: H0 cut mode 0' "$legacy/librpa.stdout"
grep -Fq 'Iteration 1: HOMO =' "$legacy/librpa.stdout"
grep -Fq 'Iteration 2: HOMO =' "$legacy/librpa.stdout"
grep -Fq 'libRPA finished successfully' "$legacy/librpa.stdout"
for iteration in 1 2; do
  checkpoint="$legacy/librpa.d/qsgw_checkpoints/iter_$(printf '%05d' "$iteration")"
  test "$(find "$checkpoint" -maxdepth 1 \
    -name 'H0_GW_spin_01_k_*.bin' -type f | wc -l)" -eq 8
done

(
  cd "$candidate"
  unset LIBRPA_WCFQ_DUMP
  export LD_LIBRARY_PATH="$candidate_build/src:$base_ld_library_path"
  printf 'started_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    >runtime.txt
  timeout 14400 mpirun -np "$mpi_ranks" "$candidate_exe" \
    >librpa.stdout 2>librpa.stderr
  printf 'completed_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    >>runtime.txt
)
grep -Fq 'QSGW fixed-basis self-consistent calculation' \
  "$candidate/librpa.stdout"
grep -Fq 'QSGW iteration 1:' "$candidate/librpa.stdout"
grep -Fq 'QSGW iteration 2:' "$candidate/librpa.stdout"
grep -Fq 'QSGW completed iterations: 2' "$candidate/librpa.stdout"
grep -Fq 'libRPA finished successfully' "$candidate/librpa.stdout"
test -s "$candidate/qsgw_matrices.dat"
test -s "$candidate/qsgw_eigenvalues.dat"
test -s "$candidate/qsgw_iterations.dat"
for trace in "$candidate/qsgw_matrices.dat" \
  "$candidate/qsgw_eigenvalues.dat" "$candidate/qsgw_iterations.dat"; do
  grep -Fqx '# qsgw_contract_version 5' "$trace"
  grep -Fqx '# fixed_basis immutable_mf0' "$trace"
  grep -Fqx '# qsgw_mixer none' "$trace"
done

python3 "$tools_dir/$compare_name" \
  "$legacy/librpa.d/qsgw_checkpoints" \
  "$candidate/qsgw_matrices.dat" \
  "$candidate/qsgw_eigenvalues.dat" \
  "$run_root/legacy-candidate-comparison.json" \
  --iterations 1:2 --n-spins 1 --n-kpoints 8 --n-bands 44 \
  --occupied-bands 4 \
  >"$run_root/comparator.stdout" 2>"$run_root/comparator.stderr"
grep -Fq '"parity_passed": true' \
  "$run_root/legacy-candidate-comparison.json"
grep -Fq '"candidate_invariants_passed": true' \
  "$run_root/legacy-candidate-comparison.json"
grep -Fq '"passed": true' \
  "$run_root/legacy-candidate-comparison.json"

python3 -c \
  'import json,sys; r=json.load(open(sys.argv[1])); print("passed="+str(r["passed"]).lower()); print("parity_passed="+str(r["parity_passed"]).lower()); print("candidate_invariants_passed="+str(r["candidate_invariants_passed"]).lower()); [print("iter_"+i+"="+json.dumps(v,sort_keys=True)) for i,v in sorted(r["iteration_reports"].items(),key=lambda x:int(x[0]))]' \
  "$run_root/legacy-candidate-comparison.json" \
  >"$run_root/acceptance-summary.txt"

printf 'acceptance=true\ncompleted_utc=%s\n' \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >>"$run_root/PROVENANCE.txt"
touch "$run_root/PARITY_GREEN"
touch "$run_root/CANDIDATE_INVARIANTS_GREEN"
touch "$run_root/RUN_GREEN"
(
  cd "$run_root"
  find . -type f ! -name OUTPUT_SHA256SUMS.txt ! -name COMPLETE \
    ! -name FAILED -print0 | LC_ALL=C sort -z | xargs -0 sha256sum \
    >OUTPUT_SHA256SUMS.txt
  sha256sum --check --quiet OUTPUT_SHA256SUMS.txt
)
touch "$run_root/COMPLETE"
find "$run_root" -type d -exec chmod 0555 {} +
find "$run_root" -type f -exec chmod 0444 {} +

echo GATE_A1_EXACT847_CANDIDATE_SYMMETRY_NOMIX_MINITER2_V1=PASS
cat "$run_root/acceptance-summary.txt"
