#!/usr/bin/env bash
set -euo pipefail

: "${CANDIDATE_COMMIT:?CANDIDATE_COMMIT must identify the clean candidate}"
: "${CANDIDATE_GATE0_ROOT:?CANDIDATE_GATE0_ROOT must identify an accepted Gate 0}"
: "${CANDIDATE_GATE0_PROVENANCE_SHA256:?Gate 0 provenance hash is required}"
: "${CANDIDATE_EXE_SHA256:?Candidate executable hash is required}"
: "${RUNNER_SHA256:?RUNNER_SHA256 must identify this exact runner}"
: "${RUN_TAG:?RUN_TAG must make the run directory immutable}"

[[ "$CANDIDATE_COMMIT" =~ ^[0-9a-f]{40}$ ]]
[[ "$CANDIDATE_GATE0_PROVENANCE_SHA256" =~ ^[0-9a-f]{64}$ ]]
[[ "$CANDIDATE_EXE_SHA256" =~ ^[0-9a-f]{64}$ ]]
[[ "$RUNNER_SHA256" =~ ^[0-9a-f]{64}$ ]]
case "$RUN_TAG" in
  *[!A-Za-z0-9._-]*|'') echo "RUN_TAG contains unsafe characters" >&2; exit 2 ;;
esac

run_root=/home/bhj/ai-runs/librpa-qsgw-gate-a-${RUN_TAG}
legacy_gate=/home/bhj/ai-runs/librpa-qsgw-gate-a0-legacy-symmetry-oracle-build-20260720-v10
legacy_build=/tmp/librpa-qsgw-gate-a0-legacy-e08f4a13-symmetry-oracle-20260720-v10/build
legacy_exe=$legacy_build/chi0_main.exe
legacy_commit=e08f4a130df7661e9ac355b9be45fb2bf9c3ed01
legacy_exe_sha=ee198669c8e57d5e2d923f2284062572dbaa06d3d6652f29830afa898a7dd225
legacy_gate_provenance_sha=35efa46f326697220f2b67697ab05e8433e7477c0fbb6989817ac66bd45476fc
legacy_gate_manifest_sha=4d913cd39b76bc81e117814e1ac5fb7251800d75bdca5fa1442ed821a38a6a48
legacy_harness_patch_sha=bc56a7a25bf85eebd7418711164e99b332ee3f5d4d4a152bfcc1015bdf6dccf3

bundle=/home/bhj/ai-runs/librpa-qsgw-gate-a-symmetry-bundle-20260720-v3
dataset=$bundle/dataset
dataset_input_dir=$dataset/
dataset_manifest_sha=869f4fd922dc1085af2cc02644f2a65e5b462237fde6428eed5a46143e833690
bundle_manifest_sha=b3be3227d0aea82492e92085340d567b47a6ba3ebb0976e1f23d542e9d5db648
contract_sha=5b90f7314d7231e0d1cc3d272957e26b9aa89d9b204c71d0378d1940098f0d46
vxc_manifest_sha=714af7a617cdf971651a21e2b599819b7a53f9eac6b76cf8e3ead8c9a4890179

tool_bundle=/tmp/librpa-qsgw-gate-a1-observers-20260720-v1
base_comparator_source=$tool_bundle/compare_qsgw_component_traces-v4-c3daf072.py
closure_source=$tool_bundle/validate_qsgw_trace_closure-v3-4a5de94e.py
closure_test_source=$tool_bundle/test_validate_qsgw_trace_closure-v3-38de02fa.py
fixed_source=$tool_bundle/validate_qsgw_fixed_basis.py
initial_source=$tool_bundle/validate_qsgw_initial_state-v1.py
initial_test_source=$tool_bundle/test_validate_qsgw_initial_state-v1.py
base_comparator_sha=c3daf072f222083a7ebdb9cf45f154d4bef64474f76db05992479a66fe30ebbc
closure_sha=4a5de94e6dbf590dded4a6ecd140aa4227fa61ffa0e73af17ec0f388610cffaf
closure_test_sha=38de02fabc0dde41911e5152b0b08a9987b312b0cea29c69396bb9e392e9c745
fixed_sha=569ecb1366bc6dd4584216bd42ac55905a4848b1a0b1bdf96c6236c782de7cb2
initial_sha=6bbade9eaeb207b6cea9fa2f80d8cbcd0baeb8cbd760a2ffab5a0fe6f6d4868a
initial_test_sha=82634e292a8fc1eb5ed454360ea2367e06687f0547da6503291c850a00e5b339

mpi_ranks=1
omp_threads=32
pytest_env=${PYTEST_ENV:-/home/bhj/ai-runs/librpa-qsgw-gate0-20260715T1731-7d69a18c/venv}
python=$pytest_env/bin/python

record_failure() {
  local rc=$?
  if [[ -d ${run_root:-/nonexistent} ]]; then
    printf 'failed_utc=%s\nexit_code=%s\n' \
      "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$rc" >"$run_root/FAILED"
  fi
  exit "$rc"
}
trap record_failure ERR

provenance_value() {
  local key=$1
  local file=$2
  awk -F= -v key="$key" '
    $1 == key {
      count += 1
      value = substr($0, length(key) + 2)
    }
    END {
      if (count != 1) exit 2
      print value
    }
  ' "$file"
}

test ! -e "$run_root"
test -e "$CANDIDATE_GATE0_ROOT/GREEN_CONFIRMED"
test ! -e "$CANDIDATE_GATE0_ROOT/FAILED"
test "$(sha256sum "$CANDIDATE_GATE0_ROOT/PROVENANCE.txt" | awk '{print $1}')" = \
  "$CANDIDATE_GATE0_PROVENANCE_SHA256"
(
  cd "$CANDIDATE_GATE0_ROOT"
  sha256sum --check --quiet OUTPUT_SHA256SUMS.txt
)
test "$(provenance_value candidate_commit "$CANDIDATE_GATE0_ROOT/PROVENANCE.txt")" = \
  "$CANDIDATE_COMMIT"
candidate_source=$(provenance_value candidate_source "$CANDIDATE_GATE0_ROOT/PROVENANCE.txt")
candidate_build=$(provenance_value candidate_build "$CANDIDATE_GATE0_ROOT/PROVENANCE.txt")
candidate_exe=$(provenance_value candidate_executable "$CANDIDATE_GATE0_ROOT/PROVENANCE.txt")
test "$(provenance_value candidate_executable_sha256 "$CANDIDATE_GATE0_ROOT/PROVENANCE.txt")" = \
  "$CANDIDATE_EXE_SHA256"
test -x "$candidate_exe"
test "$(sha256sum "$candidate_exe" | awk '{print $1}')" = "$CANDIDATE_EXE_SHA256"
test "$(git -C "$candidate_source" rev-parse HEAD)" = "$CANDIDATE_COMMIT"
test -z "$(git -C "$candidate_source" status --porcelain)"

test -e "$legacy_gate/GREEN_CONFIRMED"
test -x "$legacy_exe"
test "$(sha256sum "$legacy_exe" | awk '{print $1}')" = "$legacy_exe_sha"
test "$(sha256sum "$legacy_gate/PROVENANCE.txt" | awk '{print $1}')" = \
  "$legacy_gate_provenance_sha"
test "$(sha256sum "$legacy_gate/OUTPUT_SHA256SUMS.txt" | awk '{print $1}')" = \
  "$legacy_gate_manifest_sha"
grep -Fqx 'acceptance=true_oracle_harness_build' "$legacy_gate/PROVENANCE.txt"
grep -Fqx "combined_oracle_patch_sha256=$legacy_harness_patch_sha" \
  "$legacy_gate/PROVENANCE.txt"

test -e "$bundle/COMPLETE"
test ! -e "$bundle/FAILED"
test "$(sha256sum "$bundle/DATASET_SHA256SUMS.txt" | awk '{print $1}')" = \
  "$dataset_manifest_sha"
test "$(sha256sum "$bundle/OUTPUT_SHA256SUMS.txt" | awk '{print $1}')" = \
  "$bundle_manifest_sha"
test "$(sha256sum "$dataset/qsgw_input.contract" | awk '{print $1}')" = \
  "$contract_sha"
test "$(sha256sum "$dataset/qsgw_vxc_scf.manifest" | awk '{print $1}')" = \
  "$vxc_manifest_sha"
test "${dataset_input_dir: -1}" = "/"
(
  cd "$bundle"
  sha256sum --check --quiet OUTPUT_SHA256SUMS.txt
)

while read -r path expected; do
  test -f "$path"
  test "$(sha256sum "$path" | awk '{print $1}')" = "$expected"
done <<EOF
$base_comparator_source $base_comparator_sha
$closure_source $closure_sha
$closure_test_source $closure_test_sha
$fixed_source $fixed_sha
$initial_source $initial_sha
$initial_test_source $initial_test_sha
EOF

adapter_source=$candidate_source/qsgw-rebase-evidence/remote/fish-gate-a-current-20260721/compare_qsgw_legacy_v4_current_v6.py
adapter_test_source=$candidate_source/qsgw-rebase-evidence/remote/fish-gate-a-current-20260721/test_compare_qsgw_legacy_v4_current_v6.py
current_contract_parser_source=$candidate_source/regression_tests/backend/comparisons/cmp_qsgw.py
test -f "$adapter_source"
test -f "$adapter_test_source"
test -f "$current_contract_parser_source"
test -x "$python"
"$python" -c 'import numpy; print(numpy.__version__)' >/dev/null

mkdir -p "$run_root/tools"
printf '%s\n' "$RUNNER_SHA256" >"$run_root/runner-sha256.txt"
cp "$CANDIDATE_GATE0_ROOT/PROVENANCE.txt" "$run_root/gate0-PROVENANCE.txt"
cp "$CANDIDATE_GATE0_ROOT/OUTPUT_SHA256SUMS.txt" "$run_root/gate0-OUTPUT_SHA256SUMS.txt"
cp "$legacy_gate/PROVENANCE.txt" "$run_root/legacy-gate-PROVENANCE.txt"
cp "$bundle/PROVENANCE.txt" "$run_root/bundle-PROVENANCE.txt"
cp "$base_comparator_source" "$run_root/tools/base_comparator.py"
cp "$closure_source" "$run_root/tools/validate_qsgw_trace_closure.py"
cp "$fixed_source" "$run_root/tools/validate_qsgw_fixed_basis.py"
cp "$initial_source" "$run_root/tools/validate_qsgw_initial_state.py"
cp "$adapter_source" "$run_root/tools/compare_qsgw_legacy_v4_current_v6.py"
cp "$current_contract_parser_source" "$run_root/tools/cmp_qsgw_v6.py"

"$python" -B "$adapter_test_source" \
  >"$run_root/adapter-unit-test.stdout" \
  2>"$run_root/adapter-unit-test.stderr"
PYTHONPATH="$run_root/tools" "$python" -B "$closure_test_source" \
  >"$run_root/closure-unit-test.stdout" \
  2>"$run_root/closure-unit-test.stderr"
PYTHONPATH="$run_root/tools" "$python" -B "$initial_test_source" \
  >"$run_root/initial-unit-test.stdout" \
  2>"$run_root/initial-unit-test.stderr"

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

cat >"$run_root/CONTROLLED_PAIR.txt" <<EOF
pair=candidate_none_miniter2_vs_candidate_linear_beta_0.2_miniter5
changed_factor=parameters.qsgw_mixer
side_a_qsgw_mixer=none
side_b_qsgw_mixer=linear
candidate_configured_beta_both_sides=0.2
legacy_semantic_mapping=direct_beta_1_vs_linear_beta_0.2
dataset_sha256=$dataset_manifest_sha
candidate_commit=$CANDIDATE_COMMIT
candidate_executable_sha256=$CANDIDATE_EXE_SHA256
mpi_ranks=$mpi_ranks
omp_threads=$omp_threads
symmetry=on
headwing=off
hartree=off
band=off
EOF

run_mode() {
  local mode_name=$1
  local target_iter=$2
  local legacy_beta=$3
  local candidate_mode=$4
  local candidate_beta=$5
  local mode_root=$run_root/$mode_name
  local legacy_run=$mode_root/legacy
  local candidate_run=$mode_root/candidate

  mkdir -p "$legacy_run" "$candidate_run"
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
qsgw_mixer = $candidate_mode
qsgw_mixing_beta = $candidate_beta
qsgw_min_iter = $target_iter
qsgw_max_iter = $target_iter
qsgw_write_iteration_matrices = true
qsgw_update_hartree = false
EOF

  cat >"$mode_root/PARAMETERS.txt" <<EOF
mode=$mode_name
iterations=0:$target_iter
legacy_role=compatibility_harness_not_raw_source
legacy_effective_beta=$legacy_beta
candidate_mixer=$candidate_mode
candidate_configured_beta=$candidate_beta
symmetry=exx_on_gw_on_rpa_on
headwing=off
hartree=off
band=off
dataset=$dataset
dataset_manifest_sha256=$dataset_manifest_sha
EOF

  (
    cd "$legacy_run"
    export QSGW_ORACLE_TRACE=1
    export QSGW_ORACLE_UPDATE_HARTREE=0
    export QSGW_HROUND_SCALE=0
    export LIBRPA_QSGW_MIXING_BETA=$legacy_beta
    export LD_LIBRARY_PATH="$legacy_build/src:$legacy_build/qsgw:$base_ld_library_path"
    printf 'started_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >runtime.txt
    timeout 21600 mpirun -np "$mpi_ranks" "$legacy_exe" \
      >librpa.stdout 2>librpa.stderr
    printf 'completed_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >>runtime.txt
  )
  test -s "$legacy_run/qsgw_oracle_matrices.dat"
  test -s "$legacy_run/homo_lumo_vs_iterations.dat"
  grep -Fqx '# qsgw_contract_version 4' "$legacy_run/qsgw_oracle_matrices.dat"
  grep -Fqx '# use_symmetry_gw 1' "$legacy_run/qsgw_oracle_matrices.dat"
  grep -Fqx '# use_symmetry_exx 1' "$legacy_run/qsgw_oracle_matrices.dat"
  grep -Fqx '# use_shrink_abfs 1' "$legacy_run/qsgw_oracle_matrices.dat"
  grep -Fq 'libRPA finished successfully' "$legacy_run/librpa.stdout"
  test "$(awk 'NF && $1 !~ /^#/ {last=$1} END {print last}' \
    "$legacy_run/homo_lumo_vs_iterations.dat")" = "$target_iter"

  (
    cd "$candidate_run"
    unset QSGW_ORACLE_TRACE QSGW_ORACLE_UPDATE_HARTREE QSGW_HROUND_SCALE
    unset LIBRPA_QSGW_MIXING_BETA
    export LD_LIBRARY_PATH="$base_ld_library_path"
    printf 'started_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >runtime.txt
    timeout 21600 mpirun -np "$mpi_ranks" "$candidate_exe" \
      >librpa.stdout 2>librpa.stderr
    printf 'completed_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >>runtime.txt
  )
  for trace in qsgw_matrices.dat qsgw_eigenvalues.dat qsgw_iterations.dat; do
    test -s "$candidate_run/$trace"
    grep -Fqx '# qsgw_contract_version 6' "$candidate_run/$trace"
    grep -Fqx '# fixed_basis immutable_mf0' "$candidate_run/$trace"
    grep -Fqx '# live_update eigenvalues_wfc' "$candidate_run/$trace"
    grep -Fqx '# symmetry exx_on_gw_on_rpa_on' "$candidate_run/$trace"
    grep -Fqx '# headwing disabled_stage1' "$candidate_run/$trace"
    grep -Fqx '# hartree disabled_stage1' "$candidate_run/$trace"
    grep -Fqx '# band disabled_stage1' "$candidate_run/$trace"
    grep -Fqx '# h_qsgw_cut disabled_non_band' "$candidate_run/$trace"
    grep -Fqx "# qsgw_mixer $candidate_mode" "$candidate_run/$trace"
    grep -Fqx "# qsgw_input_contract_sha256 $contract_sha" "$candidate_run/$trace"
  done
  test -s "$candidate_run/homo_lumo_vs_iterations.dat"
  grep -Fq 'libRPA finished successfully' "$candidate_run/librpa.stdout"
  test "$(awk 'NF && $1 !~ /^#/ {last=$1} END {print last}' \
    "$candidate_run/homo_lumo_vs_iterations.dat")" = "$target_iter"

  "$python" -B "$run_root/tools/compare_qsgw_legacy_v4_current_v6.py" \
    "$legacy_run/qsgw_oracle_matrices.dat" \
    "$candidate_run/qsgw_matrices.dat" \
    "$candidate_run/qsgw_eigenvalues.dat" \
    "$candidate_run/qsgw_iterations.dat" \
    "$mode_root/legacy-current-comparison.json" \
    --base-comparator "$run_root/tools/base_comparator.py" \
    --current-contract-parser "$run_root/tools/cmp_qsgw_v6.py" \
    --iterations "0:$target_iter" \
    --expected-mode "$candidate_mode" \
    --expected-legacy-beta "$legacy_beta" \
    --expected-current-beta "$candidate_beta" \
    --frequency-tolerance 1e-10 \
    --matrix-max-abs-tolerance-ha 1e-8 \
    --matrix-relative-tolerance 1e-8 \
    --eigenvalue-tolerance-ha 1e-6 \
    --gap-tolerance-ev 1e-5 \
    --degeneracy-tolerance-ha 1e-8 \
    --state-tolerance 1e-10 \
    --normalized-current-matrix "$mode_root/current-v5-self-matrices.dat" \
    --normalized-current-eigenvalues "$mode_root/current-v5-self-eigenvalues.dat" \
    --normalized-current-iterations "$mode_root/current-v5-self-iterations.dat" \
    >"$mode_root/legacy-current-comparison.stdout" \
    2>"$mode_root/legacy-current-comparison.stderr"
  grep -Fq '"passed": true' "$mode_root/legacy-current-comparison.json"

  PYTHONPATH="$run_root/tools" "$python" -B \
    "$run_root/tools/validate_qsgw_trace_closure.py" \
    "$legacy_run/qsgw_oracle_matrices.dat" \
    "$mode_root/legacy-closure.json" \
    --iterations "0:$target_iter" --channel 0 --legacy-contract \
    --closure-tolerance-ha 1e-10 --hermiticity-tolerance-ha 1e-10 \
    >"$mode_root/legacy-closure.stdout" \
    2>"$mode_root/legacy-closure.stderr"
  grep -Fq '"passed": true' "$mode_root/legacy-closure.json"

  PYTHONPATH="$run_root/tools" "$python" -B \
    "$run_root/tools/validate_qsgw_trace_closure.py" \
    "$mode_root/current-v5-self-matrices.dat" \
    "$mode_root/current-closure.json" \
    --iterations "0:$target_iter" --channel 0 \
    --closure-tolerance-ha 1e-10 --hermiticity-tolerance-ha 1e-10 \
    >"$mode_root/current-closure.stdout" \
    2>"$mode_root/current-closure.stderr"
  grep -Fq '"passed": true' "$mode_root/current-closure.json"

  PYTHONPATH="$run_root/tools" "$python" -B \
    "$run_root/tools/validate_qsgw_fixed_basis.py" \
    "$mode_root/current-v5-self-matrices.dat" \
    "$mode_root/current-v5-self-eigenvalues.dat" \
    "$dataset/band_out" "$mode_root/current-fixed-basis.json" \
    --iterations "0:$target_iter" --channel 0 \
    --eigenvalue-tolerance-ha 1e-10 --invariant-tolerance 1e-10 \
    >"$mode_root/current-fixed-basis.stdout" \
    2>"$mode_root/current-fixed-basis.stderr"
  grep -Fq '"passed": true' "$mode_root/current-fixed-basis.json"

  PYTHONPATH="$run_root/tools" "$python" -B \
    "$run_root/tools/validate_qsgw_initial_state.py" \
    "$candidate_run/qsgw_matrices.dat" \
    "$candidate_run/qsgw_iterations.dat" \
    "$dataset/band_out" "$mode_root/current-initial-state.json" \
    --efermi-tolerance-ha 1e-12 --occupation-tolerance 1e-12 \
    >"$mode_root/current-initial-state.stdout" \
    2>"$mode_root/current-initial-state.stderr"
  grep -Fq '"passed": true' "$mode_root/current-initial-state.json"

  awk -v tolerance=1e-10 '
    NF && $1 !~ /^#/ {
      value = $7 + 0.0
      if (!seen) { reference = value; max_abs = 0.0; seen = 1 }
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
    >"$mode_root/current-electron-count.txt"

  cat >"$mode_root/ACCEPTANCE.txt" <<EOF
accepted=true
mode=$mode_name
iterations=0:$target_iter
legacy_current_matrix_relative_tolerance=1e-8
legacy_current_eigenvalue_tolerance_ha=1e-6
legacy_current_gap_tolerance_ev=1e-5
closure_tolerance_ha=1e-10
hermiticity_tolerance_ha=1e-10
fixed_basis_tolerance=1e-10
electron_count_tolerance=1e-10
EOF
}

run_mode no-mix-miniter2 2 1 none 0.2
run_mode linear-beta-0.2-miniter5 5 0.2 linear 0.2

cat >"$run_root/PROVENANCE.txt" <<EOF
gate=fish_gate_a_current_v1
acceptance=true
runner_sha256=$RUNNER_SHA256
run_tag=$RUN_TAG
legacy_role=compatibility_harness_not_raw_source
legacy_commit=$legacy_commit
legacy_executable=$legacy_exe
legacy_executable_sha256=$legacy_exe_sha
legacy_harness_patch_sha256=$legacy_harness_patch_sha
candidate_commit=$CANDIDATE_COMMIT
candidate_gate0_root=$CANDIDATE_GATE0_ROOT
candidate_gate0_provenance_sha256=$CANDIDATE_GATE0_PROVENANCE_SHA256
candidate_source=$candidate_source
candidate_build=$candidate_build
candidate_executable=$candidate_exe
candidate_executable_sha256=$CANDIDATE_EXE_SHA256
dataset=$dataset
dataset_manifest_sha256=$dataset_manifest_sha
contract_sha256=$contract_sha
vxc_manifest_sha256=$vxc_manifest_sha
symmetry=on
headwing=off
hartree=off
band=off
mode_a=none_miniter2
mode_b=linear_beta_0.2_miniter5
mpi_ranks=$mpi_ranks
omp_threads=$omp_threads
LIBRI_DETERMINISTIC_REDUCTION=$LIBRI_DETERMINISTIC_REDUCTION
python=$($python --version 2>&1)
numpy=$($python -c 'import numpy; print(numpy.__version__)')
completed_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
EOF

(
  cd "$run_root"
  find . -type f ! -name OUTPUT_SHA256SUMS.txt \
    ! -name RUN_COMPLETE ! -name FAILED -print0 | \
    sort -z | xargs -0 sha256sum >OUTPUT_SHA256SUMS.txt
  sha256sum --check --quiet OUTPUT_SHA256SUMS.txt
)
touch "$run_root/RUN_COMPLETE"
printf 'FISH_GATE_A_CURRENT_V1=PASS\n'
cat "$run_root/no-mix-miniter2/candidate/homo_lumo_vs_iterations.dat"
cat "$run_root/linear-beta-0.2-miniter5/candidate/homo_lumo_vs_iterations.dat"
