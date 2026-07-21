#!/usr/bin/env bash
set -euo pipefail

: "${RUNNER_SHA256:?RUNNER_SHA256 must identify this exact runner}"

run_root=/home/bhj/ai-runs/librpa-qsgw-gate-a1-symmetry-nomix-miniter2-20260720-v6
gate_a0=/home/bhj/ai-runs/librpa-qsgw-gate-a0-legacy-symmetry-oracle-build-20260720-v10
bundle=/home/bhj/ai-runs/librpa-qsgw-gate-a-symmetry-bundle-20260720-v3
dataset=$bundle/dataset
dataset_input_dir=$dataset/
legacy_build=/tmp/librpa-qsgw-gate-a0-legacy-e08f4a13-symmetry-oracle-20260720-v10/build
candidate_build=/tmp/librpa-qsgw-reader-binding-green-v2-b7273e13/build
legacy_exe=$legacy_build/chi0_main.exe
candidate_exe=$candidate_build/chi0_main.exe
legacy_run=$run_root/legacy
candidate_run=$run_root/candidate
tool_bundle=/tmp/librpa-qsgw-gate-a1-observers-20260720-v1
tool_dir=$run_root/tools

comparator_source=$tool_bundle/compare_qsgw_component_traces-v4-c3daf072.py
comparator_test_source=$tool_bundle/test_compare_qsgw_legacy_v4_current_v5-v4-501ef472.py
closure_source=$tool_bundle/validate_qsgw_trace_closure-v3-4a5de94e.py
closure_test_source=$tool_bundle/test_validate_qsgw_trace_closure-v3-38de02fa.py
fixed_source=$tool_bundle/validate_qsgw_fixed_basis.py
initial_source=$tool_bundle/validate_qsgw_initial_state-v1.py
initial_test_source=$tool_bundle/test_validate_qsgw_initial_state-v1.py

legacy_commit=e08f4a130df7661e9ac355b9be45fb2bf9c3ed01
candidate_commit=b7273e13c77d5ea781f192cea3c4201710b6f9fa
legacy_exe_sha=ee198669c8e57d5e2d923f2284062572dbaa06d3d6652f29830afa898a7dd225
candidate_exe_sha=e45ca971c77d32309236a78e90ddd95aa0f37f3414befdb050ae3089eb9dc4c9
gate_a0_provenance_sha=35efa46f326697220f2b67697ab05e8433e7477c0fbb6989817ac66bd45476fc
gate_a0_output_manifest_sha=4d913cd39b76bc81e117814e1ac5fb7251800d75bdca5fa1442ed821a38a6a48
gate_a0_combined_patch_sha=bc56a7a25bf85eebd7418711164e99b332ee3f5d4d4a152bfcc1015bdf6dccf3
dataset_manifest_sha=869f4fd922dc1085af2cc02644f2a65e5b462237fde6428eed5a46143e833690
bundle_output_manifest_sha=b3be3227d0aea82492e92085340d567b47a6ba3ebb0976e1f23d542e9d5db648
contract_sha=5b90f7314d7231e0d1cc3d272957e26b9aa89d9b204c71d0378d1940098f0d46
vxc_manifest_sha=714af7a617cdf971651a21e2b599819b7a53f9eac6b76cf8e3ead8c9a4890179
comparator_sha=c3daf072f222083a7ebdb9cf45f154d4bef64474f76db05992479a66fe30ebbc
comparator_test_sha=501ef472ec997a13313081f1f5945ef47e03f83ca16a7603770b32c66dc54ced
closure_sha=4a5de94e6dbf590dded4a6ecd140aa4227fa61ffa0e73af17ec0f388610cffaf
closure_test_sha=38de02fabc0dde41911e5152b0b08a9987b312b0cea29c69396bb9e392e9c745
fixed_sha=569ecb1366bc6dd4584216bd42ac55905a4848b1a0b1bdf96c6236c782de7cb2
initial_sha=6bbade9eaeb207b6cea9fa2f80d8cbcd0baeb8cbd760a2ffab5a0fe6f6d4868a
initial_test_sha=82634e292a8fc1eb5ed454360ea2367e06687f0547da6503291c850a00e5b339
target_iter=2
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
test -e "$gate_a0/GREEN_CONFIRMED"
test -e "$bundle/COMPLETE"
test ! -e "$bundle/FAILED"
test -d "$tool_bundle"
test -x "$legacy_exe"
test -x "$candidate_exe"
test "$(sha256sum "$legacy_exe" | awk '{print $1}')" = "$legacy_exe_sha"
test "$(sha256sum "$candidate_exe" | awk '{print $1}')" = "$candidate_exe_sha"
test "$(sha256sum "$gate_a0/PROVENANCE.txt" | awk '{print $1}')" = \
  "$gate_a0_provenance_sha"
test "$(sha256sum "$gate_a0/OUTPUT_SHA256SUMS.txt" | awk '{print $1}')" = \
  "$gate_a0_output_manifest_sha"
grep -Fqx 'acceptance=true_oracle_harness_build' "$gate_a0/PROVENANCE.txt"
grep -Fqx \
  "combined_oracle_patch_sha256=$gate_a0_combined_patch_sha" \
  "$gate_a0/PROVENANCE.txt"
grep -Fqx \
  'qlist_contract=historical_qsgw_band0_and_upstream_g0w0_klist' \
  "$gate_a0/PROVENANCE.txt"
grep -Fqx \
  'shrink_transform_contract=reload_shrink_sinvs_after_wc_before_g0w0_spacetime' \
  "$gate_a0/PROVENANCE.txt"
test "$(sha256sum "$bundle/DATASET_SHA256SUMS.txt" | awk '{print $1}')" = \
  "$dataset_manifest_sha"
test "$(sha256sum "$bundle/OUTPUT_SHA256SUMS.txt" | awk '{print $1}')" = \
  "$bundle_output_manifest_sha"
test "$(sha256sum "$dataset/qsgw_input.contract" | awk '{print $1}')" = \
  "$contract_sha"
test "$(sha256sum "$dataset/qsgw_vxc_scf.manifest" | awk '{print $1}')" = \
  "$vxc_manifest_sha"
test "${dataset_input_dir: -1}" = "/"
while read -r path expected; do
  test "$(sha256sum "$path" | awk '{print $1}')" = "$expected"
done <<EOF
$comparator_source $comparator_sha
$comparator_test_source $comparator_test_sha
$closure_source $closure_sha
$closure_test_source $closure_test_sha
$fixed_source $fixed_sha
$initial_source $initial_sha
$initial_test_source $initial_test_sha
EOF
(
  cd "$bundle"
  sha256sum --check --quiet OUTPUT_SHA256SUMS.txt
)
mkdir -p "$legacy_run" "$candidate_run" "$tool_dir"
printf '%s\n' "$RUNNER_SHA256" >"$run_root/runner-sha256.txt"
cp "$gate_a0/PROVENANCE.txt" "$run_root/gate-a0-PROVENANCE.txt"
cp "$bundle/PROVENANCE.txt" "$run_root/bundle-PROVENANCE.txt"
cp "$comparator_source" "$tool_dir/compare_qsgw_component_traces.py"
cp "$comparator_source" "$tool_dir/compare_qsgw_component_traces_v3.py"
cp "$comparator_test_source" \
  "$tool_dir/test_compare_qsgw_legacy_v4_current_v5_v3.py"
cp "$closure_source" "$tool_dir/validate_qsgw_trace_closure.py"
cp "$closure_source" "$tool_dir/validate_qsgw_trace_closure_v3.py"
cp "$closure_test_source" \
  "$tool_dir/test_validate_qsgw_trace_closure_v3.py"
cp "$fixed_source" "$tool_dir/validate_qsgw_fixed_basis.py"
cp "$initial_source" "$tool_dir/validate_qsgw_initial_state.py"
cp "$initial_test_source" "$tool_dir/test_validate_qsgw_initial_state.py"

cat >"$legacy_run/librpa.in" <<EOF
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
max_iter = $target_iter
EOF

cat >"$candidate_run/librpa.in" <<EOF
task = qsgw
input_dir = $dataset_input_dir
output_dir = .
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
qsgw_min_iter = $target_iter
qsgw_max_iter = $target_iter
qsgw_write_iteration_matrices = true
qsgw_update_hartree = false
qsgw_iterative_headwing = false
EOF

cat >"$run_root/PARAMETER_MAPPING.txt" <<EOF
comparison=legacy_v4_oracle_harness_to_candidate_v5_symmetry_on
iterations=0:$target_iter
legacy_mixing=linear_beta_1_direct
candidate_mixing=none_direct
legacy_oracle_reader=historical_847_shrinked_basis_contract
legacy_oracle_ibz_occupation=stored_weights_include_geometric_kpoint_weight
legacy_oracle_qlist=historical_qsgw_band0_and_upstream_g0w0_klist
parallel_topology=historical_oracle_mpi1_omp32
legacy_symmetry=abacus_exx_on_abacus_gw_on
candidate_symmetry=exx_on_gw_on_rpa_on
use_shrink_abfs=true
headwing=off
hartree=off
band=off
nfreq=6
mpi_ranks=$mpi_ranks
omp_threads=$omp_threads
dataset=$dataset
input_dir=$dataset_input_dir
input_dir_trailing_slash=true
dataset_manifest_sha256=$dataset_manifest_sha
contract_sha256=$contract_sha
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
gate=gate_a1_symmetry_nomix_miniter2_run_v6
acceptance=false_pending_numerical_observers
legacy_commit=$legacy_commit
legacy_executable=$legacy_exe
legacy_executable_sha256=$legacy_exe_sha
candidate_commit=$candidate_commit
candidate_executable=$candidate_exe
candidate_executable_sha256=$candidate_exe_sha
dataset=$dataset
input_dir=$dataset_input_dir
input_dir_trailing_slash=true
dataset_manifest_sha256=$dataset_manifest_sha
contract_sha256=$contract_sha
vxc_manifest_sha256=$vxc_manifest_sha
target_iteration=$target_iter
crystal_symmetry=on_ibz_8_to_full_bz_64
mixing=legacy_linear_beta_1_direct,candidate_none_direct
legacy_oracle_patch_scope=reader_ibz_occupation_historical_qlist_and_shrink_transform_harness_only
legacy_oracle_combined_patch_sha256=$gate_a0_combined_patch_sha
parallel_topology=historical_oracle_mpi1_omp32
use_shrink_abfs=true
headwing=off
hartree=off
band=off
mpi_ranks=$mpi_ranks
omp_threads=$omp_threads
LIBRI_DETERMINISTIC_REDUCTION=$LIBRI_DETERMINISTIC_REDUCTION
started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
EOF
sha256sum "$legacy_run/librpa.in" "$candidate_run/librpa.in" \
  "$dataset/band_out" "$dataset/bz_sampling_out" \
  "$dataset/qsgw_input.contract" "$dataset/qsgw_vxc_scf.manifest" \
  >>"$run_root/PROVENANCE.txt"

python=$(command -v python3)
PYTHONPATH="$tool_dir" "$python" -B \
  "$tool_dir/test_compare_qsgw_legacy_v4_current_v5_v3.py" \
  >"$run_root/comparator-unit-test.stdout" \
  2>"$run_root/comparator-unit-test.stderr"
PYTHONPATH="$tool_dir" "$python" -B \
  "$tool_dir/test_validate_qsgw_trace_closure_v3.py" \
  >"$run_root/closure-unit-test.stdout" \
  2>"$run_root/closure-unit-test.stderr"
PYTHONPATH="$tool_dir" "$python" -B \
  "$tool_dir/test_validate_qsgw_initial_state.py" \
  >"$run_root/initial-state-unit-test.stdout" \
  2>"$run_root/initial-state-unit-test.stderr"

(
  cd "$legacy_run"
  export QSGW_ORACLE_TRACE=1
  export QSGW_ORACLE_UPDATE_HARTREE=0
  export QSGW_HROUND_SCALE=0
  export LIBRPA_QSGW_MIXING_BETA=1
  export LD_LIBRARY_PATH="$legacy_build/src:$legacy_build/qsgw:$base_ld_library_path"
  printf 'started_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    >runtime.txt
  timeout 14400 mpirun -np "$mpi_ranks" "$legacy_exe" \
    >librpa.stdout 2>librpa.stderr
  printf 'completed_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    >>runtime.txt
)
test -s "$legacy_run/qsgw_oracle_matrices.dat"
test -s "$legacy_run/homo_lumo_vs_iterations.dat"
grep -Fqx '# qsgw_contract_version 4' "$legacy_run/qsgw_oracle_matrices.dat"
grep -Fqx '# qsgw_mixer linear' "$legacy_run/qsgw_oracle_matrices.dat"
grep -Fqx '# qsgw_mixing_beta 1' "$legacy_run/qsgw_oracle_matrices.dat"
grep -Fqx '# use_symmetry_gw 1' "$legacy_run/qsgw_oracle_matrices.dat"
grep -Fqx '# use_symmetry_exx 1' "$legacy_run/qsgw_oracle_matrices.dat"
grep -Fqx '# qsgw_update_hartree 0' "$legacy_run/qsgw_oracle_matrices.dat"
test "$(awk 'NF && $1 !~ /^#/ {last=$1} END {print last}' \
  "$legacy_run/homo_lumo_vs_iterations.dat")" = "$target_iter"
grep -Fq 'libRPA finished successfully' "$legacy_run/librpa.stdout"
if grep -Fq 'Cannot find overlap matrix file' \
  "$legacy_run/librpa.stdout" "$legacy_run/librpa.stderr"; then
  echo 'legacy run did not consume all overlap matrices' >&2
  exit 1
fi
if grep -Fq 'occupation exceeds storage capacity' \
  "$legacy_run/librpa.stdout" "$legacy_run/librpa.stderr"; then
  echo 'legacy run rejected geometrically weighted IBZ occupations' >&2
  exit 1
fi

(
  cd "$candidate_run"
  unset QSGW_ORACLE_TRACE QSGW_ORACLE_UPDATE_HARTREE QSGW_HROUND_SCALE
  unset LIBRPA_QSGW_MIXING_BETA
  export LD_LIBRARY_PATH="$base_ld_library_path"
  printf 'started_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    >runtime.txt
  timeout 14400 mpirun -np "$mpi_ranks" "$candidate_exe" \
    >librpa.stdout 2>librpa.stderr
  printf 'completed_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    >>runtime.txt
)
test -s "$candidate_run/qsgw_matrices.dat"
test -s "$candidate_run/qsgw_eigenvalues.dat"
test -s "$candidate_run/qsgw_iterations.dat"
test -s "$candidate_run/homo_lumo_vs_iterations.dat"
for trace in "$candidate_run/qsgw_matrices.dat" \
  "$candidate_run/qsgw_eigenvalues.dat" "$candidate_run/qsgw_iterations.dat"; do
  grep -Fqx '# qsgw_contract_version 5' "$trace"
  grep -Fqx '# fixed_basis immutable_mf0' "$trace"
  grep -Fqx '# live_update eigenvalues_wfc' "$trace"
  grep -Fqx '# symmetry exx_on_gw_on_rpa_on' "$trace"
  grep -Fqx '# headwing disabled_stage1' "$trace"
  grep -Fqx '# hartree disabled_stage1' "$trace"
  grep -Fqx '# band disabled_stage1' "$trace"
  grep -Fqx '# qsgw_mixer none' "$trace"
  grep -Fqx "# qsgw_input_contract_sha256 $contract_sha" "$trace"
done
test "$(awk 'NF && $1 !~ /^#/ {last=$1} END {print last}' \
  "$candidate_run/homo_lumo_vs_iterations.dat")" = "$target_iter"
grep -Fq 'libRPA finished successfully' "$candidate_run/librpa.stdout"

PYTHONPATH="$tool_dir" "$python" -B \
  "$tool_dir/compare_qsgw_component_traces.py" \
  "$legacy_run/qsgw_oracle_matrices.dat" \
  "$candidate_run/qsgw_matrices.dat" \
  "$run_root/old-new-comparison.json" \
  --iterations 0:$target_iter \
  --channel 0 \
  --contract-mode legacy_v4_to_current_v5 \
  --current-eigenvalue-trace "$candidate_run/qsgw_eigenvalues.dat" \
  --current-iteration-trace "$candidate_run/qsgw_iterations.dat" \
  --expected-legacy-use-fullcoul-exx 0 \
  --frequency-tolerance 1e-10 \
  --matrix-max-abs-tolerance-ha 1e-8 \
  --matrix-relative-tolerance 1e-8 \
  --eigenvalue-tolerance 1e-6 \
  --gap-tolerance-ev 1e-5 \
  --degeneracy-tolerance 1e-8 \
  --state-tolerance 1e-10 \
  >"$run_root/old-new-comparison.stdout" \
  2>"$run_root/old-new-comparison.stderr"
grep -Fq '"passed": true' "$run_root/old-new-comparison.json"

PYTHONPATH="$tool_dir" "$python" -B \
  "$tool_dir/validate_qsgw_initial_state.py" \
  "$candidate_run/qsgw_matrices.dat" \
  "$candidate_run/qsgw_iterations.dat" \
  "$dataset/band_out" \
  "$run_root/candidate-initial-state-validation.json" \
  --efermi-tolerance-ha 1e-12 \
  --occupation-tolerance 1e-12 \
  >"$run_root/candidate-initial-state-validator.stdout" \
  2>"$run_root/candidate-initial-state-validator.stderr"
grep -Fq '"passed": true' \
  "$run_root/candidate-initial-state-validation.json"

PYTHONPATH="$tool_dir" "$python" -B \
  "$tool_dir/validate_qsgw_trace_closure.py" \
  "$legacy_run/qsgw_oracle_matrices.dat" \
  "$run_root/legacy-closure-validation.json" \
  --iterations 0:$target_iter \
  --channel 0 \
  --legacy-contract \
  --closure-tolerance-ha 1e-10 \
  --hermiticity-tolerance-ha 1e-10 \
  >"$run_root/legacy-closure-validator.stdout" \
  2>"$run_root/legacy-closure-validator.stderr"
grep -Fq '"passed": true' "$run_root/legacy-closure-validation.json"

PYTHONPATH="$tool_dir" "$python" -B \
  "$tool_dir/validate_qsgw_trace_closure.py" \
  "$candidate_run/qsgw_matrices.dat" \
  "$run_root/candidate-closure-validation.json" \
  --iterations 0:$target_iter \
  --channel 0 \
  --closure-tolerance-ha 1e-10 \
  --hermiticity-tolerance-ha 1e-10 \
  >"$run_root/candidate-closure-validator.stdout" \
  2>"$run_root/candidate-closure-validator.stderr"
grep -Fq '"passed": true' "$run_root/candidate-closure-validation.json"

PYTHONPATH="$tool_dir" "$python" -B \
  "$tool_dir/validate_qsgw_fixed_basis.py" \
  "$candidate_run/qsgw_matrices.dat" \
  "$candidate_run/qsgw_eigenvalues.dat" \
  "$dataset/band_out" \
  "$run_root/candidate-fixed-basis-validation.json" \
  --iterations 0:$target_iter \
  --channel 0 \
  --eigenvalue-tolerance-ha 1e-10 \
  --invariant-tolerance 1e-10 \
  >"$run_root/candidate-fixed-basis-validator.stdout" \
  2>"$run_root/candidate-fixed-basis-validator.stderr"
grep -Fq '"passed": true' \
  "$run_root/candidate-fixed-basis-validation.json"

awk -v tolerance=1e-10 '
  NF && $1 !~ /^#/ {
    value = $7 + 0.0
    if (!seen) {
      reference = value
      max_abs = 0.0
      seen = 1
    }
    delta = value - reference
    if (delta < 0.0) delta = -delta
    if (delta > max_abs) max_abs = delta
    if (delta > tolerance) failed = 1
    count += 1
  }
  END {
    printf "reference_electron_count=%.17g\n", reference
    printf "max_abs_delta=%.17g\n", max_abs
    printf "tolerance=%.17g\n", tolerance
    printf "iteration_count=%d\n", count
    if (!seen || failed) exit 1
  }
' "$candidate_run/qsgw_iterations.dat" \
  >"$run_root/candidate-electron-count-conservation.txt"

cat >>"$run_root/PROVENANCE.txt" <<EOF
acceptance=true_all_numerical_observers_passed
comparison_contract=legacy_v4_to_current_v5
matrix_max_abs_tolerance_ha=1e-8
matrix_relative_tolerance=1e-8
eigenvalue_tolerance_ha=1e-6
gap_tolerance_ev=1e-5
closure_tolerance_ha=1e-10
hermiticity_tolerance_ha=1e-10
fixed_basis_invariant_tolerance=1e-10
electron_count_tolerance=1e-10
python=$($python --version 2>&1)
numpy=$($python -c 'import numpy; print(numpy.__version__)')
run_complete=true
completed_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
EOF
(
  cd "$run_root"
  find legacy candidate -maxdepth 1 -type f -print0 | sort -z | \
    xargs -0 sha256sum >RUN_OUTPUT_SHA256SUMS.txt
  sha256sum --check --quiet RUN_OUTPUT_SHA256SUMS.txt
  find . -type f ! -name OUTPUT_SHA256SUMS.txt ! -name RUN_COMPLETE \
    ! -name FAILED -print0 | sort -z | xargs -0 sha256sum \
    >OUTPUT_SHA256SUMS.txt
  sha256sum --check --quiet OUTPUT_SHA256SUMS.txt
)
touch "$run_root/RUN_COMPLETE"
echo GATE_A1_SYMMETRY_NOMIX_MINITER2_RUN_V6=PASS
cat "$legacy_run/homo_lumo_vs_iterations.dat"
cat "$candidate_run/homo_lumo_vs_iterations.dat"
exit 0
