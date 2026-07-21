#!/usr/bin/env bash
set -euo pipefail

: "${CANDIDATE_COMMIT:?CANDIDATE_COMMIT must identify the clean candidate}"
: "${CANDIDATE_GATE0_ROOT:?CANDIDATE_GATE0_ROOT must identify accepted Gate 0}"
: "${CANDIDATE_GATE0_PROVENANCE_SHA256:?Gate 0 provenance hash is required}"
: "${CANDIDATE_EXE_SHA256:?Candidate executable hash is required}"
: "${LEGACY_BUILD_ROOT:?LEGACY_BUILD_ROOT must identify corrected legacy build}"
: "${LEGACY_BUILD_PROVENANCE_SHA256:?Corrected legacy provenance hash is required}"
: "${LEGACY_BUILD_OUTPUT_SHA256:?Corrected legacy output manifest hash is required}"
: "${LEGACY_EXE_SHA256:?Corrected legacy executable hash is required}"
: "${FULLBZ_BUNDLE_ROOT:?FULLBZ_BUNDLE_ROOT must identify the pinned full-BZ bundle}"
: "${FULLBZ_BUNDLE_PROVENANCE_SHA256:?Full-BZ bundle provenance hash is required}"
: "${FULLBZ_BUNDLE_OUTPUT_SHA256:?Full-BZ bundle output manifest hash is required}"
: "${FULLBZ_DATASET_MANIFEST_SHA256:?Full-BZ dataset manifest hash is required}"
: "${GRID_BASE_CONTRACT_SHA256:?Hartree-off contract hash is required}"
: "${GRID_HARTREE_TRUNCATED_CONTRACT_SHA256:?Hartree contract hash is required}"
: "${RUNNER_SHA256:?RUNNER_SHA256 must identify this exact runner}"
: "${RUN_TAG:?RUN_TAG must make the run directory immutable}"

require_sha() {
  local value=$1
  local length=$2
  local label=$3
  [[ "$value" =~ ^[0-9a-f]+$ ]]
  test "${#value}" -eq "$length" || {
    echo "$label has the wrong length" >&2
    exit 2
  }
}

require_sha "$CANDIDATE_COMMIT" 40 CANDIDATE_COMMIT
for specification in \
  "CANDIDATE_GATE0_PROVENANCE_SHA256:$CANDIDATE_GATE0_PROVENANCE_SHA256" \
  "CANDIDATE_EXE_SHA256:$CANDIDATE_EXE_SHA256" \
  "LEGACY_BUILD_PROVENANCE_SHA256:$LEGACY_BUILD_PROVENANCE_SHA256" \
  "LEGACY_BUILD_OUTPUT_SHA256:$LEGACY_BUILD_OUTPUT_SHA256" \
  "LEGACY_EXE_SHA256:$LEGACY_EXE_SHA256" \
  "FULLBZ_BUNDLE_PROVENANCE_SHA256:$FULLBZ_BUNDLE_PROVENANCE_SHA256" \
  "FULLBZ_BUNDLE_OUTPUT_SHA256:$FULLBZ_BUNDLE_OUTPUT_SHA256" \
  "FULLBZ_DATASET_MANIFEST_SHA256:$FULLBZ_DATASET_MANIFEST_SHA256" \
  "GRID_BASE_CONTRACT_SHA256:$GRID_BASE_CONTRACT_SHA256" \
  "GRID_HARTREE_TRUNCATED_CONTRACT_SHA256:$GRID_HARTREE_TRUNCATED_CONTRACT_SHA256" \
  "RUNNER_SHA256:$RUNNER_SHA256"; do
  require_sha "${specification#*:}" 64 "${specification%%:*}"
done
case "$RUN_TAG" in
  *[!A-Za-z0-9._-]*|'') echo "RUN_TAG contains unsafe characters" >&2; exit 2 ;;
esac

run_root=/home/bhj/ai-runs/librpa-qsgw-gate-c-legacy-corrected-${RUN_TAG}
dataset=$FULLBZ_BUNDLE_ROOT/dataset
dataset_input_dir=$dataset/
legacy_exe=$LEGACY_BUILD_ROOT/build/chi0_main.exe
mpi_ranks=1
omp_threads=32
python_env=${PYTEST_ENV:-/home/bhj/ai-runs/librpa-qsgw-gate0-20260715T1731-7d69a18c/venv}
python=$python_env/bin/python

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
    $1 == key { count += 1; value = substr($0, length(key) + 2) }
    END { if (count != 1) exit 2; print value }
  ' "$file"
}

test ! -e "$run_root"
test -x "$python"
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
candidate_exe=$(provenance_value candidate_executable "$CANDIDATE_GATE0_ROOT/PROVENANCE.txt")
test "$(provenance_value candidate_executable_sha256 "$CANDIDATE_GATE0_ROOT/PROVENANCE.txt")" = \
  "$CANDIDATE_EXE_SHA256"
test -x "$candidate_exe"
test "$(sha256sum "$candidate_exe" | awk '{print $1}')" = "$CANDIDATE_EXE_SHA256"
test "$(git -C "$candidate_source" rev-parse HEAD)" = "$CANDIDATE_COMMIT"
test -z "$(git -C "$candidate_source" status --porcelain)"

test -e "$LEGACY_BUILD_ROOT/GREEN_CONFIRMED"
test ! -e "$LEGACY_BUILD_ROOT/FAILED"
test -x "$legacy_exe"
test "$(sha256sum "$legacy_exe" | awk '{print $1}')" = "$LEGACY_EXE_SHA256"
test "$(sha256sum "$LEGACY_BUILD_ROOT/PROVENANCE.txt" | awk '{print $1}')" = \
  "$LEGACY_BUILD_PROVENANCE_SHA256"
test "$(sha256sum "$LEGACY_BUILD_ROOT/OUTPUT_SHA256SUMS.txt" | awk '{print $1}')" = \
  "$LEGACY_BUILD_OUTPUT_SHA256"
(
  cd "$LEGACY_BUILD_ROOT"
  sha256sum --check --quiet OUTPUT_SHA256SUMS.txt
)
grep -Fqx 'acceptance=build_only_no_numerical_oracle_acceptance' \
  "$LEGACY_BUILD_ROOT/PROVENANCE.txt"
grep -Fqx 'reader_mode=isolated_full_hartree_map' \
  "$LEGACY_BUILD_ROOT/PROVENANCE.txt"
grep -Fqx 'gw_vq_reader=distributed_row_unchanged' \
  "$LEGACY_BUILD_ROOT/PROVENANCE.txt"
grep -Fqx \
  'reader_state_restored=Vq_cut,n_irk_points,irk_points,irk_weight' \
  "$LEGACY_BUILD_ROOT/PROVENANCE.txt"

test -e "$FULLBZ_BUNDLE_ROOT/COMPLETE"
test ! -e "$FULLBZ_BUNDLE_ROOT/FAILED"
test -z "$(find "$FULLBZ_BUNDLE_ROOT" -type l -print -quit)"
test "$(sha256sum "$FULLBZ_BUNDLE_ROOT/PROVENANCE.txt" | awk '{print $1}')" = \
  "$FULLBZ_BUNDLE_PROVENANCE_SHA256"
test "$(sha256sum "$FULLBZ_BUNDLE_ROOT/OUTPUT_SHA256SUMS.txt" | awk '{print $1}')" = \
  "$FULLBZ_BUNDLE_OUTPUT_SHA256"
test "$(sha256sum "$FULLBZ_BUNDLE_ROOT/DATASET_SHA256SUMS.txt" | awk '{print $1}')" = \
  "$FULLBZ_DATASET_MANIFEST_SHA256"
test "$(sha256sum "$dataset/qsgw_input.contract" | awk '{print $1}')" = \
  "$GRID_BASE_CONTRACT_SHA256"
test "$(sha256sum "$dataset/qsgw_input.hartree-truncated.contract" | awk '{print $1}')" = \
  "$GRID_HARTREE_TRUNCATED_CONTRACT_SHA256"
(
  cd "$FULLBZ_BUNDLE_ROOT"
  sha256sum --check --quiet OUTPUT_SHA256SUMS.txt
  sha256sum --check --quiet DATASET_SHA256SUMS.txt
)
for expected in \
  'producer_commit=dd4216653386d32f79e3219f3ea5dd2d229c1c5a' \
  'grid=4x4x4' \
  'scf_kpoints=64' \
  'full_bz_kpoints=64' \
  'symmetry=-1_full_bz' \
  'reader_version=0'; do
  grep -Fqx "$expected" "$FULLBZ_BUNDLE_ROOT/PROVENANCE.txt"
done
test "$(awk 'NR == 2 {print $1, $2}' "$dataset/bz_sampling_out")" = '64 64'
test "$(find "$dataset" -maxdepth 1 -type f -name 'KS_eigenvector_*.dat' | wc -l)" -eq 64
test "$(find "$dataset" -maxdepth 1 -type f -name 'Cs_data_*.txt' | wc -l)" -ge 1
test "$(find "$dataset" -maxdepth 1 -type f -name 'coulomb_cut_*.txt' | wc -l)" -ge 1
grep -Fqx 'n_scf_kpoints 64' "$dataset/qsgw_input.contract"
grep -Fqx 'headwing_update none' "$dataset/qsgw_input.contract"
grep -Fqx 'hartree_update off' "$dataset/qsgw_input.contract"
grep -Fqx 'hartree_update delta_density' \
  "$dataset/qsgw_input.hartree-truncated.contract"

gate_source=$candidate_source/qsgw-rebase-evidence/remote/fish-gate-c-legacy-corrected-20260721
current_gate_source=$candidate_source/qsgw-rebase-evidence/remote/fish-gate-c-current-20260721
historical_source=$candidate_source/qsgw-rebase-evidence/remote/dongfang-gates-20260715-7d69a18c
runner_source=$gate_source/run_fish_gate_c_legacy_corrected_v1.sh
adapter_source=$gate_source/compare_qsgw_legacy_hartree_v4_current_v6.py
adapter_test_source=$gate_source/test_compare_qsgw_legacy_hartree_v4_current_v6.py
legacy_null_source=$gate_source/compare_legacy_hartree_null_delta_v1.py
legacy_null_test_source=$gate_source/test_compare_legacy_hartree_null_delta_v1.py
current_null_source=$current_gate_source/compare_qsgw_hartree_null_delta_v1.py
current_null_test_source=$current_gate_source/test_compare_qsgw_hartree_null_delta_v1.py
contraction_source=$current_gate_source/validate_qsgw_hartree_contraction_v1.py
contraction_test_source=$current_gate_source/test_validate_qsgw_hartree_contraction_v1.py
dump_source=$current_gate_source/validate_qsgw_hartree_dump_v2.py
trace_source=$current_gate_source/validate_qsgw_hartree_trace_v6.py
trace_test_source=$current_gate_source/test_validate_qsgw_hartree_trace_v6.py
base_comparator_source=$historical_source/compare_qsgw_component_traces_v3.py
recompute_source=$historical_source/recompute_qsgw_hartree_delta_v1.py
contract_parser_source=$candidate_source/regression_tests/backend/comparisons/cmp_qsgw.py
for path in "$runner_source" "$adapter_source" "$adapter_test_source" \
  "$legacy_null_source" "$legacy_null_test_source" \
  "$current_null_source" "$current_null_test_source" \
  "$contraction_source" "$contraction_test_source" "$dump_source" \
  "$trace_source" "$trace_test_source" "$base_comparator_source" \
  "$recompute_source" "$contract_parser_source"; do
  test -f "$path"
done
test "$(sha256sum "$runner_source" | awk '{print $1}')" = "$RUNNER_SHA256"

mkdir -p "$run_root/tools" "$run_root/unit-legacy-null" \
  "$run_root/unit-current-null" "$run_root/unit-contraction" \
  "$run_root/unit-current-trace"
printf '%s\n' "$RUNNER_SHA256" >"$run_root/runner-sha256.txt"
cp "$CANDIDATE_GATE0_ROOT/PROVENANCE.txt" "$run_root/gate0-PROVENANCE.txt"
cp "$LEGACY_BUILD_ROOT/PROVENANCE.txt" "$run_root/legacy-build-PROVENANCE.txt"
cp "$FULLBZ_BUNDLE_ROOT/PROVENANCE.txt" "$run_root/bundle-PROVENANCE.txt"
cp "$FULLBZ_BUNDLE_ROOT/DATASET_SHA256SUMS.txt" \
  "$run_root/bundle-DATASET_SHA256SUMS.txt"
cp "$adapter_source" "$run_root/tools/"
cp "$legacy_null_source" "$run_root/tools/"
cp "$current_null_source" "$run_root/tools/"
cp "$contraction_source" "$run_root/tools/"
cp "$dump_source" "$run_root/tools/"
cp "$trace_source" "$run_root/tools/"
cp "$base_comparator_source" "$run_root/tools/"
cp "$recompute_source" "$run_root/tools/"
cp "$contract_parser_source" "$run_root/tools/cmp_qsgw.py"

"$python" -B "$adapter_test_source" \
  >"$run_root/adapter-unit-test.stdout" \
  2>"$run_root/adapter-unit-test.stderr"
PYTHONPATH="$run_root/tools" \
LIBRPA_QSGW_LEGACY_HARTREE_NULL_TEST_TMP="$run_root/unit-legacy-null" \
  "$python" -B "$legacy_null_test_source" \
  >"$run_root/legacy-null-unit-test.stdout" \
  2>"$run_root/legacy-null-unit-test.stderr"
PYTHONPATH="$run_root/tools" \
LIBRPA_QSGW_HARTREE_NULL_TEST_TMP="$run_root/unit-current-null" \
  "$python" -B "$current_null_test_source" \
  >"$run_root/current-null-unit-test.stdout" \
  2>"$run_root/current-null-unit-test.stderr"
PYTHONPATH="$run_root/tools" \
LIBRPA_QSGW_HARTREE_CONTRACTION_TEST_TMP="$run_root/unit-contraction" \
  "$python" -B "$contraction_test_source" \
  >"$run_root/contraction-unit-test.stdout" \
  2>"$run_root/contraction-unit-test.stderr"
PYTHONPATH="$run_root/tools" \
LIBRPA_QSGW_HARTREE_TRACE_TEST_TMP="$run_root/unit-current-trace" \
  "$python" -B "$trace_test_source" \
  >"$run_root/current-trace-unit-test.stdout" \
  2>"$run_root/current-trace-unit-test.stderr"

set +u
source /opt/intel/oneapi/setvars.sh --force \
  >"$run_root/oneapi-setvars.stdout" 2>"$run_root/oneapi-setvars.stderr"
set -u
export OMP_NUM_THREADS=$omp_threads
export MKL_NUM_THREADS=$omp_threads
export OPENBLAS_NUM_THREADS=$omp_threads
export OMP_PROC_BIND=spread
export OMP_PLACES=cores
export I_MPI_PIN_DOMAIN=omp
export LIBRI_DETERMINISTIC_REDUCTION=1
base_ld_library_path=${LD_LIBRARY_PATH:-}

write_legacy_input() {
  local mode_root=$1
  local iterations=$2
  cat >"$mode_root/librpa.in" <<EOF
task = qsgw
input_dir = $dataset_input_dir
output_dir = .
constants_choice = internal
nfreq = 6
tfgrid_type = minimax
n_params_anacon = -1
parallel_routing = libri
vq_threshold = 0
sqrt_coulomb_threshold = 0
use_scalapack_gw_wc = true
use_shrink_abfs = false
use_shrink_chi = false
use_pyatb = false
replace_w_head = false
option_dielect_func = 0
use_fullcoul_exx = false
use_fullcoul_eps = true
use_fullcoul_wc = false
use_abacus_exx_symmetry = false
use_abacus_gw_symmetry = false
qsgw_iterative_headwing = false
max_iter = $iterations
EOF
}

write_current_input() {
  local mode_root=$1
  local iterations=$2
  local update_hartree=$3
  local contract=qsgw_input.contract
  if [[ "$update_hartree" == true ]]; then
    contract=qsgw_input.hartree-truncated.contract
  fi
  cat >"$mode_root/librpa.in" <<EOF
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
use_shrink_abfs = false
use_shrink_chi = false
use_pyatb = false
replace_w_head = false
option_dielect_func = 0
use_fullcoul_exx = false
use_fullcoul_eps = true
use_fullcoul_wc = false
use_abacus_exx_symmetry = false
use_abacus_gw_symmetry = false
use_symmetry_exx = false
use_symmetry_gw = false
use_symmetry_rpa = false
use_kpara_scf_eigvec = false
qsgw_input_contract = $contract
qsgw_mixer = linear
qsgw_mixing_beta = 0.2
qsgw_min_iter = $iterations
qsgw_max_iter = $iterations
qsgw_write_iteration_matrices = true
qsgw_update_hartree = $update_hartree
EOF
  if [[ "$update_hartree" == true ]]; then
    cat >>"$mode_root/librpa.in" <<'EOF'
qsgw_hartree_coulomb = truncated
qsgw_hartree_normalization = legacy_extra_inverse_nk
EOF
  fi
}

run_legacy() {
  local name=$1
  local iterations=$2
  local update_hartree=$3
  local mode_root=$run_root/$name
  mkdir -p "$mode_root"
  write_legacy_input "$mode_root" "$iterations"
  (
    cd "$mode_root"
    export QSGW_ORACLE_TRACE=1
    export QSGW_ORACLE_UPDATE_HARTREE="$update_hartree"
    export QSGW_HROUND_SCALE=0
    export LIBRPA_QSGW_MIXING_BETA=0.2
    export LD_LIBRARY_PATH="$LEGACY_BUILD_ROOT/build/qsgw:$LEGACY_BUILD_ROOT/build/src:$base_ld_library_path"
    printf 'started_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >runtime.txt
    timeout 21600 mpirun -np "$mpi_ranks" "$legacy_exe" \
      >librpa.stdout 2>librpa.stderr
    printf 'completed_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >>runtime.txt
  )
  test -s "$mode_root/qsgw_oracle_matrices.dat"
  test -s "$mode_root/homo_lumo_vs_iterations.dat"
  grep -Fqx '# qsgw_contract_version 4' "$mode_root/qsgw_oracle_matrices.dat"
  grep -Fqx '# qsgw_mixer linear' "$mode_root/qsgw_oracle_matrices.dat"
  grep -Fqx "# qsgw_update_hartree $update_hartree" \
    "$mode_root/qsgw_oracle_matrices.dat"
  grep -Fqx '# use_shrink_abfs 0' "$mode_root/qsgw_oracle_matrices.dat"
  grep -Fqx '# use_symmetry_gw 0' "$mode_root/qsgw_oracle_matrices.dat"
  grep -Fqx '# use_symmetry_exx 0' "$mode_root/qsgw_oracle_matrices.dat"
  test "$(awk 'NF && $1 !~ /^#/ {last=$1} END {print last}' \
    "$mode_root/homo_lumo_vs_iterations.dat")" = "$iterations"
  grep -Fq 'libRPA finished successfully' "$mode_root/librpa.stdout"
  if [[ "$update_hartree" == 1 ]]; then
    grep -Fq 'QSGW_LEGACY_HARTREE_READER mode=isolated_full gw_vq_restored=true reader_state_restored=true' \
      "$mode_root/librpa.stdout"
  else
    ! grep -Fq 'QSGW_LEGACY_HARTREE_READER' "$mode_root/librpa.stdout"
  fi
}

run_current() {
  local name=$1
  local iterations=$2
  local update_hartree=$3
  local mode_root=$run_root/$name
  local dump_root=$mode_root/hartree-dump
  mkdir -p "$mode_root" "$dump_root"
  write_current_input "$mode_root" "$iterations" "$update_hartree"
  (
    cd "$mode_root"
    unset QSGW_ORACLE_TRACE QSGW_ORACLE_UPDATE_HARTREE QSGW_HROUND_SCALE
    unset LIBRPA_QSGW_MIXING_BETA
    export LD_LIBRARY_PATH="$base_ld_library_path"
    if [[ "$update_hartree" == true ]]; then
      export LIBRPA_QSGW_HARTREE_DUMP_DIR="$dump_root"
    else
      unset LIBRPA_QSGW_HARTREE_DUMP_DIR
    fi
    printf 'started_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >runtime.txt
    timeout 21600 mpirun -np "$mpi_ranks" "$candidate_exe" \
      >librpa.stdout 2>librpa.stderr
    printf 'completed_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >>runtime.txt
  )
  for trace in qsgw_matrices.dat qsgw_eigenvalues.dat qsgw_iterations.dat; do
    test -s "$mode_root/$trace"
    grep -Fqx '# qsgw_contract_version 6' "$mode_root/$trace"
    grep -Fqx '# fixed_basis immutable_mf0' "$mode_root/$trace"
    grep -Fqx '# live_update eigenvalues_wfc' "$mode_root/$trace"
    grep -Fqx '# symmetry exx_off_gw_off_rpa_off' "$mode_root/$trace"
    grep -Fqx '# headwing disabled_stage1' "$mode_root/$trace"
    grep -Fqx '# qsgw_mixer linear' "$mode_root/$trace"
  done
  test "$(awk 'NF && $1 !~ /^#/ {last=$1} END {print last}' \
    "$mode_root/qsgw_iterations.dat")" = "$iterations"
  grep -Fq 'libRPA finished successfully' "$mode_root/librpa.stdout"
  if [[ "$update_hartree" == true ]]; then
    for trace in qsgw_matrices.dat qsgw_eigenvalues.dat qsgw_iterations.dat; do
      grep -Fqx '# hartree delta_density' "$mode_root/$trace"
      grep -Fqx '# hartree_coulomb truncated' "$mode_root/$trace"
      grep -Fqx '# hartree_normalization legacy_extra_inverse_nk' \
        "$mode_root/$trace"
      grep -Fqx "# qsgw_input_contract_sha256 $GRID_HARTREE_TRUNCATED_CONTRACT_SHA256" \
        "$mode_root/$trace"
    done
    test "$(find "$dump_root" -mindepth 1 -maxdepth 1 -type d -name 'call_*' | wc -l)" -eq "$iterations"
  else
    for trace in qsgw_matrices.dat qsgw_eigenvalues.dat qsgw_iterations.dat; do
      grep -Fqx '# hartree disabled_stage1' "$mode_root/$trace"
      grep -Fqx "# qsgw_input_contract_sha256 $GRID_BASE_CONTRACT_SHA256" \
        "$mode_root/$trace"
    done
  fi
}

run_legacy legacy-control-off 1 0
run_legacy legacy-control-on 1 1
PYTHONPATH="$run_root/tools" "$python" -B \
  "$run_root/tools/compare_legacy_hartree_null_delta_v1.py" \
  "$run_root/legacy-control-on/qsgw_oracle_matrices.dat" \
  "$run_root/legacy-control-off/qsgw_oracle_matrices.dat" \
  "$run_root/legacy-control-on/homo_lumo_vs_iterations.dat" \
  "$run_root/legacy-control-off/homo_lumo_vs_iterations.dat" \
  "$run_root/legacy-null-delta-comparison.json" \
  >"$run_root/legacy-null-delta-comparison.stdout" \
  2>"$run_root/legacy-null-delta-comparison.stderr"
grep -Fq '"passed": true' "$run_root/legacy-null-delta-comparison.json"

run_legacy legacy-parity-on 2 1
run_current current-control-off 1 false
run_current current-parity-on 2 true

PYTHONPATH="$run_root/tools" "$python" -B \
  "$run_root/tools/compare_qsgw_hartree_null_delta_v1.py" \
  "$run_root/current-parity-on" "$run_root/current-control-off" \
  "$run_root/current-null-delta-comparison.json" \
  >"$run_root/current-null-delta-comparison.stdout" \
  2>"$run_root/current-null-delta-comparison.stderr"
grep -Fq '"passed": true' "$run_root/current-null-delta-comparison.json"

PYTHONPATH="$run_root/tools" "$python" -B \
  "$run_root/tools/validate_qsgw_hartree_trace_v6.py" \
  "$run_root/current-parity-on/qsgw_matrices.dat" \
  "$run_root/current-parity-on/qsgw_iterations.dat" \
  "$run_root/current-trace-validation.json" --expected-iterations 2 \
  --expected-symmetry exx_off_gw_off_rpa_off \
  >"$run_root/current-trace-validation.stdout" \
  2>"$run_root/current-trace-validation.stderr"
grep -Fq '"passed": true' "$run_root/current-trace-validation.json"

PYTHONPATH="$run_root/tools" "$python" -B \
  "$run_root/tools/validate_qsgw_hartree_contraction_v1.py" \
  --input-dir "$dataset" \
  --dump-call "$run_root/current-parity-on/hartree-dump/call_002" \
  --expected-normalization legacy_extra_inverse_nk \
  --coulomb-prefix coulomb_cut_ \
  --output "$run_root/current-independent-contraction.json" \
  >"$run_root/current-independent-contraction.stdout" \
  2>"$run_root/current-independent-contraction.stderr"
grep -Fq '"passed": true' "$run_root/current-independent-contraction.json"

"$python" -B "$run_root/tools/compare_qsgw_legacy_hartree_v4_current_v6.py" \
  "$run_root/legacy-parity-on/qsgw_oracle_matrices.dat" \
  "$run_root/current-parity-on/qsgw_matrices.dat" \
  "$run_root/current-parity-on/qsgw_eigenvalues.dat" \
  "$run_root/current-parity-on/qsgw_iterations.dat" \
  "$run_root/legacy-current-parity.json" \
  --base-comparator "$run_root/tools/compare_qsgw_component_traces_v3.py" \
  --current-contract-parser "$run_root/tools/cmp_qsgw.py" \
  --iterations 0:2 --expected-mode linear \
  --expected-legacy-beta 0.2 --expected-current-beta 0.2 \
  --frequency-tolerance 1e-10 \
  --matrix-max-abs-tolerance-ha 1e-8 \
  --matrix-relative-tolerance 1e-8 \
  --eigenvalue-tolerance-ha 1e-6 \
  --gap-tolerance-ev 1e-5 \
  --degeneracy-tolerance-ha 1e-8 \
  --state-tolerance 1e-10 \
  --normalized-current-matrix "$run_root/current-v5-self-matrices.dat" \
  --normalized-current-eigenvalues "$run_root/current-v5-self-eigenvalues.dat" \
  --normalized-current-iterations "$run_root/current-v5-self-iterations.dat" \
  >"$run_root/legacy-current-parity.stdout" \
  2>"$run_root/legacy-current-parity.stderr"
grep -Fq '"passed": true' "$run_root/legacy-current-parity.json"

cat >"$run_root/ACCEPTANCE.txt" <<EOF
accepted=true
scope=corrected_legacy_same_dataset_hartree_parity
producer_commit=dd4216653386d32f79e3219f3ea5dd2d229c1c5a
grid=4x4x4_full_bz_64
iterations=0:2
legacy_reader=isolated_full_hartree_map
legacy_null_delta_side_effect_guard=true
current_null_delta_side_effect_guard=true
current_independent_contraction=true
hartree_coulomb=truncated
hartree_normalization=legacy_extra_inverse_nk
mixer=linear
mixing_beta=0.2
matrix_max_abs_tolerance_ha=1e-8
matrix_relative_frobenius_tolerance=1e-8
eigenvalue_tolerance_ha=1e-6
gap_tolerance_ev=1e-5
hermiticity_tolerance_ha=1e-10
EOF

cat >"$run_root/PROVENANCE.txt" <<EOF
gate=fish_gate_c_legacy_corrected_v1
acceptance=true
runner_sha256=$RUNNER_SHA256
run_tag=$RUN_TAG
candidate_commit=$CANDIDATE_COMMIT
candidate_executable=$candidate_exe
candidate_executable_sha256=$CANDIDATE_EXE_SHA256
legacy_build_root=$LEGACY_BUILD_ROOT
legacy_executable=$legacy_exe
legacy_executable_sha256=$LEGACY_EXE_SHA256
legacy_build_provenance_sha256=$LEGACY_BUILD_PROVENANCE_SHA256
fullbz_bundle_root=$FULLBZ_BUNDLE_ROOT
fullbz_bundle_provenance_sha256=$FULLBZ_BUNDLE_PROVENANCE_SHA256
dataset_manifest_sha256=$FULLBZ_DATASET_MANIFEST_SHA256
grid_base_contract_sha256=$GRID_BASE_CONTRACT_SHA256
grid_hartree_truncated_contract_sha256=$GRID_HARTREE_TRUNCATED_CONTRACT_SHA256
producer_commit=dd4216653386d32f79e3219f3ea5dd2d229c1c5a
legacy_control_iterations=0:1
legacy_parity_iterations=0:2
current_control_iterations=0:1
current_parity_iterations=0:2
symmetry=off_full_bz_64
headwing=off
hartree=delta_density
hartree_coulomb=truncated
hartree_normalization=legacy_extra_inverse_nk
mixer=linear
mixing_beta=0.2
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
    ! -name GREEN_CONFIRMED ! -name FAILED -print0 | \
    LC_ALL=C sort -z | xargs -0 sha256sum >OUTPUT_SHA256SUMS.txt
  sha256sum --check --quiet OUTPUT_SHA256SUMS.txt
)
touch "$run_root/GREEN_CONFIRMED"
trap - ERR
echo "FISH_GATE_C_LEGACY_CORRECTED_V1=PASS"
echo "run_root=$run_root"
